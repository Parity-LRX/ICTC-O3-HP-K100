# SPDX-License-Identifier: MIT
"""Owned-source reciprocal electrostatics for the native MFF/RCCL runtime.

Supports slab-distributed multipole FFTs alongside the replicated baseline.
Atoms, learned moments and their coordinate derivatives stay partitioned.
Differentiable collectives retain the learned-source force contribution.
"""
import os
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
        from mace_ictc.models.long_range import LatentReciprocalLongRange
        from ictc_o3.electrostatics.functional import MeshLongRangeKernel3D
        if not isinstance(module, LatentReciprocalLongRange) or not isinstance(module.kernel, MeshLongRangeKernel3D):
            raise ValueError('Native MFF electrostatics requires reciprocal-spectral-v1 with mesh_fft')
        if module.kernel.full_ewald and not module.kernel.reciprocal_only:
            raise ValueError('Native MFF requires reciprocal-only electrostatics; exact real-space Ewald is unsupported')
        self.module = module
        self.calls = 0
        self.fft_policy = os.environ.get('ICTC_MFF_MULTIPOLE_FFT', 'auto')
        if self.fft_policy not in ('auto', 'replicated', 'slabs', 'particles'):
            raise ValueError('ICTC_MFF_MULTIPOLE_FFT must be auto, replicated, slabs or particles')
        self.description = dict(backend='native-hip-reciprocal',
            mesh_size=module.kernel.mesh_size, assignment=module.kernel.assignment,
            smearing_sigma=module.kernel.smearing_sigma, boundary=module.kernel.boundary,
            source_channels=module.source_channels, counting='owned atoms only',
            transport=('RCCL differentiable mesh and spectral collectives'
                       if self.fft_policy != 'replicated' else 'RCCL Fourier-amplitude sum and reverse sum'),
            fft=self.fft_policy,
            fft_min_elements=1 << 20,
            scalar_fft_min_elements=1 << 23,
            fft_fallback='single rank, small work in auto, or my < ranks uses replicated transforms')

    def __call__(self, pos, batch, cell, q, dipole, quadrupole, *, owned, size):
        from ictc_o3.electrostatics.functional import _prepare_frac_for_boundary_batched
        from ictc_o3.backends.k100.hip_reciprocal_fft import scalar_potential
        from ictc_o3.backends.k100.hip_reciprocal_fft import multipole_energy
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
        self.description['fft_selected'] = 'replicated'
        components = 1 + 3 * (dipole is not None) + 6 * (quadrupole is not None)
        mx, my, mz = kernel.mesh_shape
        # A scalar potential requires an inverse FFT and a second particle
        # exchange; use a higher threshold for that latency-sensitive path.
        threshold = (1 << 23) if components == 1 else (1 << 20)
        enough_work = mx * my * mz * q.shape[1] * components >= threshold
        use_particles = self.fft_policy == 'particles' or (self.fft_policy == 'auto' and enough_work)
        if use_particles and size > 1 and my >= size:
            from .mff_reciprocal_particles import energy as particle_energy
            self.description['fft_selected'] = 'particle-slabs'
            energy = particle_energy(kernel,position,graph,cell,q,dipole,quadrupole,
                size=size,rank=dist.get_rank(),counts=counts,reduce_energy=reduce)
        elif dipole is not None or quadrupole is not None:
            if self.fft_policy == 'slabs' and size > 1 and my >= size:
                from .mff_reciprocal_slabs import multipole_energy as slab_energy
                self.description['fft_selected'] = 'slabs'
                energy = slab_energy(kernel, position, graph, cell, q, dipole, quadrupole,
                    size=size, rank=dist.get_rank(), counts=counts, reduce_energy=reduce)
            else:
                energy = multipole_energy(kernel, position, graph, cell, q, dipole, quadrupole,
                    reduce_amplitudes=reduce,graph_counts=counts,
                    compact_quadrupole=self.fft_policy in ('auto','particles'))
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
