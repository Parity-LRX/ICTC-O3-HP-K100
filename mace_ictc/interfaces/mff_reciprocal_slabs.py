# SPDX-License-Identifier: MIT
"""Differentiable slab FFTs for the single-node native MFF process group.

Owned atoms first contribute to summed real-space x slabs. A transpose then
distributes y slabs for the final x transform and local spectral contraction.
Padding is used only in collective buffers, never in the physical FFT grid.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as functional


class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        size = dist.get_world_size()
        out = value.new_empty((value.shape[0] // size, *value.shape[1:]))
        dist.reduce_scatter_tensor(out, value.contiguous())
        return out

    @staticmethod
    def backward(ctx, gradient):
        return _AllGather.apply(gradient)


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        out = value.new_empty((value.shape[0] * dist.get_world_size(), *value.shape[1:]))
        dist.all_gather_into_tensor(out, value.contiguous())
        return out

    @staticmethod
    def backward(ctx, gradient):
        return _ReduceScatter.apply(gradient)


class _Transpose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        out = torch.empty_like(value, memory_format=torch.contiguous_format)
        dist.all_to_all_single(out, value.contiguous())
        return out

    @staticmethod
    def backward(ctx, gradient):
        return _Transpose.apply(gradient)


def transform(mesh, *, size, rank):
    """Return [B,C,mx,local_y,mz//2+1] and the global y slice.

    Input is the local contribution to a channels-first real grid. The caller
    must use one identical FFT shape and process group on every rank.
    """
    batch, channels, mx, my, mz = mesh.shape
    if size < 2 or my < size:
        raise ValueError('Slab FFT requires at least two ranks and my >= ranks')
    nx, ny = (mx + size - 1) // size, (my + size - 1) // size
    xpad = mesh if nx * size == mx else functional.pad(mesh, (0, 0, 0, 0, 0, nx * size - mx))
    pieces = xpad.reshape(batch, channels, size, nx, my, mz)
    pieces = pieces.permute(2, 0, 1, 3, 4, 5).reshape(size * batch, channels, nx, my, mz)
    local = _ReduceScatter.apply(pieces)
    yz = torch.fft.rfftn(local, dim=(-2, -1))
    if my % size == 0:
        pieces = yz.reshape(batch, channels, nx, size, ny, mz // 2 + 1)
        pieces = pieces.permute(3, 0, 1, 2, 4, 5).contiguous()
    else:
        # Balanced intervals keep every slab nonempty even when ceil(my/P)
        # would leave trailing ranks beyond the physical grid.
        intervals = [(r * my // size, (r + 1) * my // size) for r in range(size)]
        pieces = torch.stack([functional.pad(yz[..., begin:end, :],
            (0, 0, 0, ny - (end - begin))) for begin, end in intervals])
    received = torch.view_as_complex(_Transpose.apply(torch.view_as_real(pieces)))
    local = received.permute(1, 2, 0, 3, 4, 5).reshape(batch, channels, nx * size, ny, mz // 2 + 1)
    start, stop = rank * my // size, (rank + 1) * my // size
    local = local[:, :, :mx, :stop-start, :]
    return torch.fft.fft(local, dim=2), slice(start, stop)


def multipole_energy(kernel, pos, batch, cell, source, dipole, quadrupole,
                     *, size, rank, counts, reduce_energy):
    from mace_ictc.models.long_range import _prepare_frac_for_boundary_batched
    from mace_ictc.models.hip_reciprocal_fft import multipole_spectral_energy
    frac = _prepare_frac_for_boundary_batched(pos, batch, cell,
        boundary=kernel.boundary, slab_padding_factor=kernel.slab_padding_factor)
    n, channels = source.shape
    fields = [source]
    if dipole is not None:
        fields.append(dipole.reshape(n, channels * 3))
    if quadrupole is not None:
        fields.append(quadrupole.reshape(n, channels * 9))
    packed = torch.cat(fields, dim=1)
    # Coalesced assignment remains bounded in atom count and stencil width.
    mesh = kernel._spread(frac, batch, packed, cell.shape[0]).permute(0, 4, 1, 2, 3).contiguous()
    spectrum, y_slice = transform(mesh, size=size, rank=rank)
    energy = multipole_spectral_energy(kernel, spectrum.flatten(2).transpose(1, 2),
        cell, channels, has_dipole=dipole is not None,
        has_quadrupole=quadrupole is not None, y_slice=y_slice)
    energy = reduce_energy(energy)
    return (energy.index_select(0, batch) / counts.index_select(0, batch))[:, None]
