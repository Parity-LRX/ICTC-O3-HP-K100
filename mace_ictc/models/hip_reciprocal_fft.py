# SPDX-License-Identifier: MIT
"""Real-FFT reciprocal algebra, preserving both members of every k pair.

At Nyquist indices in a triclinic cell, the folded mirror vector need not
equal -k. Both Green weights and both multipole amplitudes are evaluated;
blindly doubling the retained half would change the original functional.
"""
import math
import torch


def geometry(kernel, cell, dtype, *, exclude_k0):
    from .long_range import (_effective_cell_for_boundary, _fft_integer_frequencies,
                             _inverse_3x3, _det_3x3, _safe_vector_norm)
    m = kernel.mesh_size
    effective = _effective_cell_for_boundary(cell, boundary=kernel.boundary,
                    slab_padding_factor=kernel.slab_padding_factor, dtype=dtype)
    freq = _fft_integer_frequencies(m, device=cell.device, dtype=dtype)
    z = torch.arange(m // 2 + 1, device=cell.device, dtype=dtype)
    if m % 2 == 0:
        z = torch.where(z == m // 2, -z, z)
    xyz = torch.meshgrid(freq, freq, z, indexing='ij')
    positive = torch.stack(xyz, dim=-1).reshape(-1, 3)
    negative = -positive
    if m % 2 == 0:
        negative = torch.where(positive == -m // 2, positive, negative)
    inverse = _inverse_3x3(effective).transpose(-1, -2)
    kp = 2. * math.pi * torch.einsum('kd,bdh->bkh', positive, inverse)
    kn = 2. * math.pi * torch.einsum('kd,bdh->bkh', negative, inverse)
    volume = _det_3x3(effective).abs().clamp_min(kernel.k_norm_floor)
    window = kernel._build_assignment_window(device=cell.device, dtype=dtype)[..., :m//2+1]
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
    interior = torch.arange(m//2+1, device=cell.device)
    interior = (interior > 0) & ((interior < m//2) if m % 2 == 0 else torch.ones_like(interior,dtype=torch.bool))
    interior = interior[None, None, :].expand(m, m, -1).reshape(1, -1)
    return kp, kn, spectral(kp), spectral(kn), interior, alpha, volume, real_cutoff


def scalar_potential(kernel, mesh, cell):
    m = kernel.mesh_size
    _, _, sp, sn, _, alpha, volume, cutoff = geometry(
        kernel, cell, mesh.dtype, exclude_k0=kernel.full_ewald or not kernel.include_k0)
    field = mesh.permute(0, 4, 1, 2, 3).contiguous()
    transformed = torch.fft.rfftn(field, dim=(-3, -2, -1))
    # The original returns the real part of a full inverse FFT. Symmetrizing
    # the folded Green kernel reproduces that real linear map exactly.
    weights = (.5 * (sp + sn)).reshape(cell.shape[0], 1, m, m, m//2+1)
    filtered = torch.fft.irfftn(transformed * weights, s=(m,m,m), dim=(-3,-2,-1))
    return (filtered * float(m**3)).permute(0,2,3,4,1), alpha, volume, cutoff


def multipole_energy(kernel, pos, batch, cell, source, dipole, quadrupole):
    from .long_range import _prepare_frac_for_boundary_batched
    frac = _prepare_frac_for_boundary_batched(pos, batch, cell,
                boundary=kernel.boundary, slab_padding_factor=kernel.slab_padding_factor)
    n, channels = source.shape
    fields = [source]
    if dipole is not None:
        fields.append(dipole.reshape(n, channels * 3))
    if quadrupole is not None:
        fields.append(quadrupole.reshape(n, channels * 9))
    packed = torch.cat(fields, dim=1) if len(fields)>1 else source
    # Assignment and its coordinate/mixed adjoints coalesce source channels.
    # A bounded grid transpose is cheaper than strided plane access per lane.
    mesh = kernel._spread(frac, batch, packed, cell.shape[0]).permute(0,4,1,2,3).contiguous()
    transformed = torch.fft.rfftn(mesh, dim=(-3,-2,-1)).flatten(2).transpose(1,2)
    kp, kn, sp, sn, interior, _, _, _ = geometry(kernel, cell, pos.dtype, exclude_k0=True)
    def amplitude(spectrum, k):
        scalar = spectrum[:, :, :channels]
        real, imag = scalar.real, scalar.imag
        if dipole is not None:
            mu = spectrum[:, :, channels:4*channels].reshape(cell.shape[0],-1,channels,3)
            value = torch.einsum('bkx,bksx->bks',k.to(mu.dtype),mu)
            real, imag = real - value.imag, imag + value.real
        if quadrupole is not None:
            offset = channels * (4 if dipole is not None else 1)
            quad = spectrum[:, :, offset:offset+9*channels].reshape(cell.shape[0],-1,channels,3,3)
            kc = k.to(quad.dtype)
            value = torch.einsum('bkx,bksxy,bky->bks',kc,quad,kc)
            real, imag = real - .5 * value.real, imag - .5 * value.imag
        return real.square()+imag.square()
    density = sp[:, :, None] * amplitude(transformed, kp)
    density = density + (sn * interior)[:, :, None] * amplitude(transformed.conj(), kn)
    graph_energy = .5 * density.sum(dim=(1,2))
    counts = kernel._graph_counts(batch, cell, dtype=source.dtype)
    return (graph_energy.index_select(0,batch)/counts.index_select(0,batch))[:,None]
