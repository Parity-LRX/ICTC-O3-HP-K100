# SPDX-License-Identifier: MIT
"""Public monopole/dipole/quadrupole reciprocal electrostatics on Hygon K100."""
import torch
from torch import nn


def configure_electrostatics(model, *, backend='auto', fuse_multipole_fft=True):
    """Select mesh execution without changing the functional or parameters.

    auto uses native gfx926 FP32/FP64 assignment and otherwise the differentiable
    Torch reference. hip requires that target when evaluated. reference retains
    the original mesh arithmetic. No compilation occurs during configuration.
    """
    if backend not in ('auto', 'hip', 'reference'):
        raise ValueError('Electrostatics backend must be auto, hip or reference')
    if not isinstance(fuse_multipole_fft, bool):
        raise TypeError('fuse_multipole_fft must be a bool')
    from mace_ictc.models.long_range import MeshLongRangeKernel3D
    names = []
    for name, module in model.named_modules():
        if isinstance(module, MeshLongRangeKernel3D):
            module.mesh_backend = backend
            module.fuse_multipole_fft = fuse_multipole_fft
            names.append(name)
    return dict(backend=backend, mesh_kernels=names,
                fuse_multipole_fft=fuse_multipole_fft,
                derivatives='energy, forces, cell gradients and mixed second derivatives')


class MeshElectrostatics(nn.Module):
    """Screened reciprocal correction compatible with the original ICTC model.

    forward(pos[N,3], batch[N], cell[B,3,3], q[N,S], mu[N,S,3], Q[N,S,3,3])
    returns atom energies [N,1]. Cell rows are lattice vectors. q is required;
    mu and Q are optional Cartesian moments. Scalar sources are neutralized per
    graph by default. The multipole route partitions graph energy uniformly.

    This is a reciprocal functional. It does not add real-space, self or
    background Ewald terms. The slab option retains the original padded-cell
    approximation. Native derivatives are implemented through second order;
    select reference for derivatives of higher order.

    smearing_sigma is a fixed length in coordinate units, with spectral factor
    exp(-sigma^2*k^2/2). The default is 1.0. None restores legacy box-dependent
    screening. Gaussian screening requires full_ewald=True; a bare kernel uses
    full_ewald=False together with smearing_sigma=None.
    """
    def __init__(self, *, mesh_size=32, assignment='pcs', boundary='periodic',
                 slab_padding_factor=2, neutralize=True, full_ewald=True,
                 green_mode='poisson', filter_hidden_dim=64,
                 energy_partition='potential', backend='auto', fuse_multipole_fft=True,
                 smearing_sigma=1.0):
        super().__init__()
        from mace_ictc.models.long_range import MeshLongRangeKernel3D
        if not isinstance(mesh_size, int) or isinstance(mesh_size, bool) or mesh_size < 1:
            raise ValueError('mesh_size must be a positive integer')
        self.neutralize = bool(neutralize)
        self.kernel = MeshLongRangeKernel3D(
            mesh_size=mesh_size, assignment=assignment, boundary=boundary,
            slab_padding_factor=slab_padding_factor, full_ewald=full_ewald,
            reciprocal_only=True, green_mode=green_mode,
            filter_hidden_dim=filter_hidden_dim, energy_partition=energy_partition,
            mesh_backend=backend, fuse_multipole_fft=fuse_multipole_fft,
            smearing_sigma=smearing_sigma)

    def forward(self, pos, batch, cell, monopole, dipole=None, quadrupole=None):
        n = pos.shape[0]
        if pos.ndim != 2 or pos.shape[1] != 3 or cell.ndim != 3 or cell.shape[1:] != (3, 3) or cell.shape[0] < 1:
            raise ValueError('Expected positions [N,3] and row-vector cells [B,3,3]')
        if monopole.ndim != 2 or monopole.shape[0] != n or monopole.shape[1] < 1:
            raise ValueError('Expected monopole [N,S] with S > 0')
        if pos.dtype not in (torch.float32, torch.float64):
            raise ValueError('Mesh electrostatics expects FP32 or FP64 inputs')
        if batch.shape != (n,) or batch.dtype != torch.int64 or batch.device != pos.device:
            raise ValueError('Expected int64 batch [N] on the coordinate device')
        for value, shape in ((cell, (cell.shape[0], 3, 3)),
                             (monopole, monopole.shape),
                             (dipole, (*monopole.shape, 3)),
                             (quadrupole, (*monopole.shape, 3, 3))):
            if value is not None and (value.shape != shape or value.device != pos.device or value.dtype != pos.dtype):
                raise ValueError('Positions, cells and moments must have compatible shapes and share dtype/device')
        source = monopole
        if self.neutralize:
            if cell.shape[0] == 1:
                source = source - source.sum(dim=0, keepdim=True) / max(n, 1)
            else:
                counts = source.new_zeros(cell.shape[0]).index_add(0, batch, source.new_ones(n)).clamp_min(1)
                sums = source.new_zeros((cell.shape[0], source.shape[1])).index_add(0, batch, source)
                source = source - (sums / counts[:, None]).index_select(0, batch)
        if dipole is None and quadrupole is None:
            return self.kernel(pos, batch, cell, source)
        return self.kernel.multipole_energy(pos, batch, cell, source, dipole, quadrupole)


def mesh_spread(frac, batch, source, *, mesh_size, num_graphs, assignment='pcs',
                boundary='periodic', backend='auto', channels_first=False):
    """B-spline spread without an [N,stencil,C] native temporary."""
    from mace_ictc.models.hip_electrostatics import supported, spread
    if backend not in ('auto', 'hip', 'reference'):
        raise ValueError('backend must be auto, hip or reference')
    if backend == 'hip' or (backend == 'auto' and supported(frac)):
        return spread(frac, batch, source, mesh_size=mesh_size, num_graphs=num_graphs,
                      assignment=assignment, boundary=boundary, channels_first=channels_first)
    from mace_ictc.models.long_range import _spread_source_to_mesh_batched, _build_assignment_offsets
    value = _spread_source_to_mesh_batched(
        frac, batch, source, mesh_size=mesh_size, num_graphs=num_graphs,
        assignment=assignment, boundary=boundary, assignment_offsets=_build_assignment_offsets(assignment))
    return value.permute(0, 4, 1, 2, 3).contiguous() if channels_first else value


def mesh_gather(frac, batch, mesh, *, mesh_size, num_graphs, assignment='pcs',
                boundary='periodic', backend='auto', channels_first=False):
    """Adjoint mesh gather, differentiable in both field and coordinates."""
    from mace_ictc.models.hip_electrostatics import supported, gather
    if backend not in ('auto', 'hip', 'reference'):
        raise ValueError('backend must be auto, hip or reference')
    if backend == 'hip' or (backend == 'auto' and supported(frac)):
        return gather(frac, batch, mesh, mesh_size=mesh_size, num_graphs=num_graphs,
                      assignment=assignment, boundary=boundary, channels_first=channels_first)
    from mace_ictc.models.long_range import _gather_source_from_mesh_batched, _build_assignment_offsets
    if channels_first:
        mesh = mesh.permute(0, 2, 3, 4, 1)
    return _gather_source_from_mesh_batched(
        frac, batch, mesh, mesh_size=mesh_size, assignment=assignment,
        boundary=boundary, assignment_offsets=_build_assignment_offsets(assignment))
