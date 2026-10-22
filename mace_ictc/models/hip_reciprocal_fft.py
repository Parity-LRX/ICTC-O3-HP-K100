# SPDX-License-Identifier: MIT
"""Real-FFT reciprocal algebra, preserving both members of every k pair.

At Nyquist indices in a triclinic cell, the folded mirror vector need not
equal -k. Both Green weights and both multipole amplitudes are evaluated;
blindly doubling the retained half would change the original functional.
"""
import math
import torch


def geometry(kernel, cell, dtype, *, exclude_k0, y_slice=None):
    from .long_range import (_effective_cell_for_boundary, _fft_integer_frequencies,
                             _inverse_3x3, _det_3x3, _safe_vector_norm)
    mx, my, mz = kernel.mesh_shape
    effective = _effective_cell_for_boundary(cell, boundary=kernel.boundary,
                    slab_padding_factor=kernel.slab_padding_factor, dtype=dtype)
    fx = _fft_integer_frequencies(mx, device=cell.device, dtype=dtype)
    fy = _fft_integer_frequencies(my, device=cell.device, dtype=dtype)
    if y_slice is not None:
        fy = fy[y_slice]
    z = torch.arange(mz // 2 + 1, device=cell.device, dtype=dtype)
    if mz % 2 == 0:
        z = torch.where(z == mz // 2, -z, z)
    xyz = torch.meshgrid(fx, fy, z, indexing='ij')
    positive = torch.stack(xyz, dim=-1).reshape(-1, 3)
    negative = -positive
    nyquist = positive.new_tensor([-(size // 2) if size % 2 == 0 else float('inf')
                                   for size in (mx, my, mz)])
    negative = torch.where(positive == nyquist, positive, negative)
    inverse = _inverse_3x3(effective).transpose(-1, -2)
    kp = 2. * math.pi * torch.einsum('kd,bdh->bkh', positive, inverse)
    kn = 2. * math.pi * torch.einsum('kd,bdh->bkh', negative, inverse)
    volume = _det_3x3(effective).abs().clamp_min(kernel.k_norm_floor)
    if y_slice is None:
        window = kernel._build_assignment_window(device=cell.device, dtype=dtype)[..., :mz//2+1]
    else:
        from .long_range import _assignment_window_1d
        windows = [_assignment_window_1d(mesh_size=size, assignment=kernel.assignment,
            device=cell.device, dtype=dtype) for size in (mx, my, mz)]
        window = (windows[0][:, None, None] * windows[1][y_slice][None, :, None]
                  * windows[2][:mz//2+1][None, None, :])
    deconv = window.reshape(1, -1).clamp_min(kernel.assignment_window_floor).square().reciprocal()
    real_cutoff = alpha = None
    if kernel.full_ewald:
        real_cutoff = kernel._estimate_real_cutoff_batched(cell.to(dtype))
        alpha = kernel._estimate_ewald_alpha(real_cutoff)
    def spectral(k):
        norms = _safe_vector_norm(k, dim=-1, floor=kernel.k_norm_floor)
        value = kernel.green_kernel(norms) / volume[:, None] * deconv
        if alpha is not None:
            value = value * torch.exp(-norms.square() / (4. * alpha[:, None].square()))
        if exclude_k0:
            value = torch.where(norms > kernel.k_norm_floor, value, torch.zeros_like(value))
        return value
    interior = torch.arange(mz//2+1, device=cell.device)
    interior = (interior > 0) & ((interior < mz//2) if mz % 2 == 0 else torch.ones_like(interior,dtype=torch.bool))
    interior = interior[None, None, :].expand(mx, fy.numel(), -1).reshape(1, -1)
    return kp, kn, spectral(kp), spectral(kn), interior, alpha, volume, real_cutoff


def scalar_potential(kernel, mesh, cell, *, reduce_spectrum=None):
    mx, my, mz = kernel.mesh_shape
    _, _, sp, sn, _, alpha, volume, cutoff = geometry(
        kernel, cell, mesh.dtype, exclude_k0=kernel.full_ewald or not kernel.include_k0)
    field = mesh.permute(0, 4, 1, 2, 3).contiguous()
    transformed = torch.fft.rfftn(field, dim=(-3, -2, -1))
    if reduce_spectrum is not None:
        transformed = torch.view_as_complex(reduce_spectrum(torch.view_as_real(transformed).contiguous()))
    # The original returns the real part of a full inverse FFT. Symmetrizing
    # the folded Green kernel reproduces that real linear map exactly.
    weights = (.5 * (sp + sn)).reshape(cell.shape[0], 1, mx, my, mz//2+1)
    filtered = torch.fft.irfftn(transformed * weights, s=(mx,my,mz), dim=(-3,-2,-1))
    return (filtered * float(mx*my*mz)).permute(0,2,3,4,1), alpha, volume, cutoff


def pack_moments(source, dipole, quadrupole, *, compact_quadrupole=False):
    """Pack Cartesian fields, optionally using the six exact Q combinations."""
    n, channels = source.shape
    fields = [source]
    if dipole is not None:
        fields.append(dipole.reshape(n, channels * 3))
    if quadrupole is not None:
        if compact_quadrupole:
            q = quadrupole
            compact = torch.stack((q[...,0,0],q[...,1,1],q[...,2,2],
                q[...,0,1]+q[...,1,0],q[...,0,2]+q[...,2,0],q[...,1,2]+q[...,2,1]),dim=-1)
            fields.append(compact.reshape(n, channels * 6))
        else:
            fields.append(quadrupole.reshape(n, channels * 9))
    return torch.cat(fields, dim=1) if len(fields)>1 else source


def multipole_energy(kernel, pos, batch, cell, source, dipole, quadrupole,
                     *, reduce_amplitudes=None, graph_counts=None, compact_quadrupole=False):
    from .long_range import _prepare_frac_for_boundary_batched
    frac = _prepare_frac_for_boundary_batched(pos, batch, cell,
                boundary=kernel.boundary, slab_padding_factor=kernel.slab_padding_factor)
    channels = source.shape[1]
    packed = pack_moments(source,dipole,quadrupole,compact_quadrupole=compact_quadrupole)
    # Assignment and its coordinate/mixed adjoints coalesce source channels.
    # A bounded grid transpose is cheaper than strided plane access per lane.
    mesh = kernel._spread(frac, batch, packed, cell.shape[0]).permute(0,4,1,2,3).contiguous()
    transformed = torch.fft.rfftn(mesh, dim=(-3,-2,-1)).flatten(2).transpose(1,2)
    graph_energy = multipole_spectral_energy(kernel, transformed, cell, channels,
        has_dipole=dipole is not None, has_quadrupole=quadrupole is not None,
        reduce_amplitudes=reduce_amplitudes,compact_quadrupole=compact_quadrupole)
    counts = kernel._graph_counts(batch, cell, dtype=source.dtype) if graph_counts is None else graph_counts
    return (graph_energy.index_select(0,batch)/counts.index_select(0,batch))[:,None]


def multipole_spectral_energy(kernel, transformed, cell, channels, *,
                             has_dipole, has_quadrupole,
                             reduce_amplitudes=None, y_slice=None, compact_quadrupole=False):
    """Energy of a complete spectrum or a disjoint slab along its y axis."""
    kp, kn, sp, sn, interior, _, _, _ = geometry(kernel, cell,
        transformed.real.dtype, exclude_k0=True, y_slice=y_slice)
    def amplitude(spectrum, k):
        scalar = spectrum[:, :, :channels]
        real, imag = scalar.real, scalar.imag
        if has_dipole:
            mu = spectrum[:, :, channels:4*channels].reshape(cell.shape[0],-1,channels,3)
            value = torch.einsum('bkx,bksx->bks',k.to(mu.dtype),mu)
            real, imag = real - value.imag, imag + value.real
        if has_quadrupole:
            offset = channels * (4 if has_dipole else 1)
            if compact_quadrupole:
                quad = spectrum[:, :, offset:offset+6*channels].reshape(cell.shape[0],-1,channels,6)
                x, y, z = k.unbind(-1)
                weights = torch.stack((x*x,y*y,z*z,x*y,x*z,y*z),dim=-1)
                value = torch.einsum('bkx,bksx->bks',weights.to(quad.dtype),quad)
            else:
                quad = spectrum[:, :, offset:offset+9*channels].reshape(cell.shape[0],-1,channels,3,3)
                kc = k.to(quad.dtype)
                value = torch.einsum('bkx,bksxy,bky->bks',kc,quad,kc)
            real, imag = real - .5 * value.real, imag - .5 * value.imag
        if reduce_amplitudes is not None:
            return torch.stack((real, imag), dim=-1)
        return real.square()+imag.square()
    positive = amplitude(transformed, kp)
    negative = amplitude(transformed.conj(), kn)
    if reduce_amplitudes is not None:
        # Communicate two folded complex source amplitudes, rather than every
        # Cartesian moment grid. Both Nyquist partners are still explicit.
        positive, negative = reduce_amplitudes(torch.stack((positive, negative))).unbind(0)
        positive = positive.square().sum(-1)
        negative = negative.square().sum(-1)
    density = sp[:, :, None] * positive
    density = density + (sn * interior)[:, :, None] * negative
    return .5 * density.sum(dim=(1,2))
