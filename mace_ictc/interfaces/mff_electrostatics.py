# SPDX-License-Identifier: MIT
"""Owned-source reciprocal electrostatics for the native MFF/RCCL runtime.

FFTs are replicated, while atoms, learned moments and their coordinate
derivatives remain partitioned. Only summed Fourier amplitudes travel between
GPUs. The reverse collective retains the learned-source force contribution.
"""
import torch
import torch.distributed as dist


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        out = value.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(out)
        return out

    @staticmethod
    def backward(ctx, gradient):
        # Each rank differentiates only its owned energy allocation. Summing
        # cotangents recovers the derivative of the complete physical energy.
        return _Sum.apply(gradient)


class NativeReciprocal:
    def __init__(self, module):
        from mace_ictc.models.long_range import LatentReciprocalLongRange, MeshLongRangeKernel3D
        if not isinstance(module, LatentReciprocalLongRange) or not isinstance(module.kernel, MeshLongRangeKernel3D):
            raise ValueError('Native MFF electrostatics requires reciprocal-spectral-v1 with mesh_fft')
        if module.kernel.full_ewald and not module.kernel.reciprocal_only:
            raise ValueError('Native MFF requires reciprocal-only electrostatics; exact real-space Ewald is unsupported')
        self.module = module
        self.calls = 0
        self.description = dict(backend='native-hip-reciprocal',
            mesh_size=module.kernel.mesh_size, assignment=module.kernel.assignment,
            smearing_sigma=module.kernel.smearing_sigma, boundary=module.kernel.boundary,
            source_channels=module.source_channels, counting='owned atoms only',
            transport='RCCL Fourier-amplitude sum and reverse sum', fft='replicated per rank')

    def __call__(self, pos, batch, cell, q, dipole, quadrupole, *, owned, size):
        from mace_ictc.models.long_range import _prepare_frac_for_boundary_batched
        from mace_ictc.models.hip_reciprocal_fft import scalar_potential, multipole_energy
        if cell.shape != (1, 3, 3) or q.shape[0] != owned:
            raise ValueError('Native MFF reciprocal evaluation requires one cell and owned sources')
        kernel = self.module.kernel
        # A domain can predict constant moments (for example, isolated atoms).
        # Retain the same reverse-collective graph on those domains as well.
        q = q + pos[:owned].sum() * 0.
        reduce = _Sum.apply if size > 1 else None
        counts = torch.tensor([owned], dtype=torch.int64, device=pos.device)
        if size > 1:
            dist.all_reduce(counts)
        counts = counts.to(q.dtype)
        if self.module.neutralize:
            total = q.sum(0, keepdim=True)
            if reduce is not None:
                total = reduce(total)
            q = q - total / counts[:, None]
        position, graph = pos[:owned], batch[:owned]
        if dipole is not None or quadrupole is not None:
            energy = multipole_energy(kernel, position, graph, cell, q, dipole, quadrupole,
                                      reduce_amplitudes=reduce, graph_counts=counts)
        else:
            frac = _prepare_frac_for_boundary_batched(position, graph, cell,
                boundary=kernel.boundary, slab_padding_factor=kernel.slab_padding_factor)
            mesh = kernel._spread(frac, graph, q, 1)
            potential, _, _, _ = scalar_potential(kernel, mesh, cell, reduce_spectrum=reduce)
            energy = .5 * (q * kernel._gather(frac, graph, potential)).sum(-1, keepdim=True)
            if kernel.energy_partition == 'uniform':
                total = energy.sum()
                if reduce is not None:
                    total = reduce(total)
                energy = (total / counts).expand(owned, 1)
        if self.module.energy_scale is not None:
            energy = self.module.energy_scale * energy
        self.calls += 1
        return torch.cat((energy, energy.new_zeros((pos.shape[0] - owned, 1))))
