# SPDX-License-Identifier: MIT
"""Route owned sources to bounded FFT slabs, with differentiable return paths.

Only slabs touched by the B-spline stencil receive an atom. Coordinates and
learned moments remain differentiable across both source and potential traffic.
No full real-space mesh or global atom array is allocated on any rank.
"""
import torch
import torch.distributed as dist


class _Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, sent, received):
        ctx.sent, ctx.received = sent, received
        out = value.new_empty((sum(received), value.shape[1]))
        dist.all_to_all_single(out, value.contiguous(),
            output_split_sizes=received, input_split_sizes=sent)
        return out

    @staticmethod
    def backward(ctx, gradient):
        return _Route.apply(gradient, ctx.received, ctx.sent), None, None


class _Particles:
    def __init__(self, frac, *, mesh_shape, assignment, size, rank):
        self.owned = frac.shape[0]
        mx = mesh_shape[0]
        self.width = (mx + size - 1) // size
        self.start = rank * self.width
        order = {'cic': 2, 'tsc': 3, 'pcs': 4}[assignment]
        scaled = frac.detach()[:, 0] * mx
        base = torch.floor(scaled - .5) if order == 3 else torch.floor(scaled)
        if order == 4:
            base = base - 1
        index = base.long()[:, None] + torch.arange(order, device=frac.device)
        owners = torch.div(index.remainder(mx), self.width, rounding_mode='floor')
        # A short axis or a periodic wrap can touch one rank more than once.
        # Send that atom once; assignment retains every physical stencil term.
        previous = torch.ones((order, order), dtype=torch.bool, device=frac.device).tril(-1)
        unique = ~((owners[:, :, None] == owners[:, None, :]) & previous).any(-1)
        destination = owners[unique]
        rows = torch.arange(self.owned, device=frac.device)[:, None].expand(-1, order)[unique]
        self.rows = rows.index_select(0, torch.argsort(destination))
        counts = torch.bincount(destination, minlength=size)
        incoming = torch.empty_like(counts)
        dist.all_to_all_single(incoming, counts)
        self.sent, self.received = counts.cpu().tolist(), incoming.cpu().tolist()

    def forward(self, value):
        return _Route.apply(value.index_select(0, self.rows), self.sent, self.received)

    def reverse(self, value):
        parts = _Route.apply(value, self.received, self.sent)
        return value.new_zeros((self.owned, value.shape[1])).index_add(0, self.rows, parts)


def _pack(source, dipole, quadrupole):
    from ictc_o3.backends.k100.hip_reciprocal_fft import pack_moments
    return pack_moments(source,dipole,quadrupole,compact_quadrupole=True)


def energy(kernel, pos, batch, cell, source, dipole, quadrupole,
           *, size, rank, counts, reduce_energy):
    from ictc_o3.electrostatics.functional import _prepare_frac_for_boundary_batched
    from ictc_o3.backends.k100.hip_electrostatics import spread
    from ictc_o3.backends.k100.hip_electrostatics import gather
    from ictc_o3.backends.k100.hip_reciprocal_fft import geometry
    from ictc_o3.backends.k100.hip_reciprocal_fft import multipole_spectral_energy
    from .mff_reciprocal_slabs import transform_local, inverse_local
    if cell.shape[0] != 1:
        raise ValueError('Native particle routing requires one periodic cell')
    frac = _prepare_frac_for_boundary_batched(pos, batch, cell,
        boundary=kernel.boundary, slab_padding_factor=kernel.slab_padding_factor)
    plan = _Particles(frac, mesh_shape=kernel.mesh_shape, assignment=kernel.assignment,
                      size=size, rank=rank)
    packed = _pack(source, dipole, quadrupole)
    routed = plan.forward(torch.cat((frac, packed),dim=1))
    local_frac, local_source = routed[:, :3], routed[:, 3:]
    local_batch = torch.zeros(routed.shape[0], device=batch.device, dtype=torch.int64)
    options = dict(mesh_size=kernel.mesh_size,num_graphs=1,assignment=kernel.assignment,
        boundary=kernel.boundary,channels_first=True,x_start=plan.start,x_count=plan.width)
    mesh = spread(local_frac, local_batch, local_source, **options)
    spectrum, interval = transform_local(mesh, mesh_shape=kernel.mesh_shape, size=size, rank=rank)
    if dipole is not None or quadrupole is not None:
        total = multipole_spectral_energy(kernel,spectrum.flatten(2).transpose(1,2),cell,
            source.shape[1],has_dipole=dipole is not None,has_quadrupole=quadrupole is not None,
            y_slice=interval,compact_quadrupole=True)
        total = reduce_energy(total)
        return (total.index_select(0,batch)/counts.index_select(0,batch))[:,None]
    mx, my, mz = kernel.mesh_shape
    _, _, positive, negative, _, _, _, _ = geometry(kernel,cell,source.dtype,
        exclude_k0=kernel.full_ewald or not kernel.include_k0,y_slice=interval)
    width = interval.stop - interval.start
    weights = (.5*(positive+negative)).reshape(1,1,mx,width,mz//2+1)
    potential = inverse_local(spectrum*weights,mesh_shape=kernel.mesh_shape,size=size,rank=rank)
    potential = potential * float(mx*my*mz)
    partial = gather(local_frac,local_batch,potential,**options)
    owned_potential = plan.reverse(partial)
    result = .5*(source*owned_potential).sum(-1,keepdim=True)
    if kernel.energy_partition == 'uniform':
        total = reduce_energy(result.sum())
        result = (total/counts).expand(source.shape[0],1)
    return result
