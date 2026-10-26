"""Differentiable reciprocal and mesh electrostatics functionals."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mesh import mesh_shape, normalize_mesh_size

def _build_integer_k_lattice(kmax: int, include_k0: bool) -> torch.Tensor:
    values: list[list[float]] = []
    for i in range(-int(kmax), int(kmax) + 1):
        for j in range(-int(kmax), int(kmax) + 1):
            for k in range(-int(kmax), int(kmax) + 1):
                if not include_k0 and i == 0 and j == 0 and k == 0:
                    continue
                values.append([float(i), float(j), float(k)])
    if not values:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32)



def _fft_integer_frequencies(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.fft.fftfreq(size, d=1.0 / float(size), device=device).to(dtype=dtype)



def _safe_vector_norm(x: torch.Tensor, *, dim: int, floor: float) -> torch.Tensor:
    return torch.sqrt(x.square().sum(dim=dim).clamp_min(float(floor) * float(floor)))



def _inverse_3x3(cell: torch.Tensor, *, floor: float = 1.0e-12) -> torch.Tensor:
    squeeze = cell.dim() == 2
    m = cell.unsqueeze(0) if squeeze else cell
    a = m[..., 0, :]
    b = m[..., 1, :]
    c = m[..., 2, :]
    col0 = torch.cross(b, c, dim=-1)
    col1 = torch.cross(c, a, dim=-1)
    col2 = torch.cross(a, b, dim=-1)
    det = (a * col0).sum(dim=-1)
    det_floor = det.new_tensor(float(floor))
    det_safe = torch.where(
        det.abs() >= det_floor,
        det,
        torch.where(det >= 0.0, det_floor, -det_floor),
    )
    inv = torch.stack((col0, col1, col2), dim=-1) / det_safe[..., None, None]
    return inv.squeeze(0) if squeeze else inv



def _det_3x3(cell: torch.Tensor) -> torch.Tensor:
    squeeze = cell.dim() == 2
    m = cell.unsqueeze(0) if squeeze else cell
    det = (m[..., 0, :] * torch.cross(m[..., 1, :], m[..., 2, :], dim=-1)).sum(dim=-1)
    return det.squeeze(0) if squeeze else det



def _effective_cell_for_boundary(
    cell: torch.Tensor,
    *,
    boundary: str,
    slab_padding_factor: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    effective_cell = cell.to(dtype=dtype)
    if boundary == "slab":
        effective_cell = effective_cell.clone()
        if effective_cell.dim() == 2:
            effective_cell[2] = effective_cell[2] * float(max(int(slab_padding_factor), 1))
        else:
            effective_cell[:, 2] = effective_cell[:, 2] * float(max(int(slab_padding_factor), 1))
    return effective_cell



def _prepare_frac_for_boundary(
    pos: torch.Tensor,
    cell: torch.Tensor,
    *,
    boundary: str,
    slab_padding_factor: int,
) -> torch.Tensor:
    inv_cell = _inverse_3x3(cell)
    frac = torch.einsum("ni,ij->nj", pos, inv_cell)
    if boundary == "periodic":
        return frac - torch.floor(frac)

    frac = frac.clone()
    frac[:, :2] = frac[:, :2] - torch.floor(frac[:, :2])
    pad = float(max(int(slab_padding_factor), 1))
    z_offset = 0.5 * (pad - 1.0) / pad
    frac[:, 2] = frac[:, 2] / pad + z_offset
    return frac



def _prepare_frac_for_boundary_batched(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    *,
    boundary: str,
    slab_padding_factor: int,
) -> torch.Tensor:
    if cell.shape[0] == 1:
        return _prepare_frac_for_boundary(
            pos, cell[0], boundary=boundary, slab_padding_factor=slab_padding_factor)
    inv_cell = _inverse_3x3(cell)
    atom_inv_cell = inv_cell.index_select(0, batch)
    frac = torch.einsum("ni,nij->nj", pos, atom_inv_cell)
    if boundary == "periodic":
        return frac - torch.floor(frac)

    frac = frac.clone()
    frac[:, :2] = frac[:, :2] - torch.floor(frac[:, :2])
    pad = float(max(int(slab_padding_factor), 1))
    z_offset = 0.5 * (pad - 1.0) / pad
    frac[:, 2] = frac[:, 2] / pad + z_offset
    return frac



def _assignment_stencil_size(assignment: str) -> int:
    if assignment == "cic":
        return 2
    if assignment == "tsc":
        return 3
    if assignment == "pcs":
        return 4
    raise ValueError(f"Unsupported mesh assignment: {assignment!r}")



def _build_assignment_offsets(assignment: str) -> torch.Tensor:
    stencil = _assignment_stencil_size(assignment)
    values = torch.arange(stencil, dtype=torch.long)
    return torch.cartesian_prod(values, values, values)



def _assignment_kernel_1d(scaled: torch.Tensor, assignment: str) -> tuple[torch.Tensor, torch.Tensor]:
    if assignment == "cic":
        base = torch.floor(scaled).to(dtype=torch.long)
        frac = scaled - base.to(dtype=scaled.dtype)
        weights = torch.stack([1.0 - frac, frac], dim=-1)
        return base, weights

    if assignment == "tsc":
        shifted = scaled - 0.5
        base = torch.floor(shifted).to(dtype=torch.long)
        local = scaled - base.to(dtype=scaled.dtype)
        weights = torch.stack(
            [
                0.5 * (1.5 - local).square(),
                0.75 - (local - 1.0).square(),
                0.5 * (local - 0.5).square(),
            ],
            dim=-1,
        )
        return base, weights

    if assignment == "pcs":
        floor_scaled = torch.floor(scaled).to(dtype=torch.long)
        base = floor_scaled - 1
        frac = scaled - floor_scaled.to(dtype=scaled.dtype)
        frac2 = frac * frac
        frac3 = frac2 * frac
        weights = torch.stack(
            [
                ((1.0 - frac).pow(3)) / 6.0,
                (3.0 * frac3 - 6.0 * frac2 + 4.0) / 6.0,
                (-3.0 * frac3 + 3.0 * frac2 + 3.0 * frac + 1.0) / 6.0,
                frac3 / 6.0,
            ],
            dim=-1,
        )
        return base, weights

    raise ValueError(f"Unsupported mesh assignment: {assignment!r}")



def _assignment_weights_from_scaled(scaled: torch.Tensor, assignment: str) -> tuple[torch.Tensor, torch.Tensor]:
    base_components: list[torch.Tensor] = []
    weight_components: list[torch.Tensor] = []
    for dim in range(3):
        base_dim, weights_dim = _assignment_kernel_1d(scaled[:, dim], assignment)
        base_components.append(base_dim)
        weight_components.append(weights_dim)
    base = torch.stack(base_components, dim=-1)
    weights = (
        weight_components[0][:, :, None, None]
        * weight_components[1][:, None, :, None]
        * weight_components[2][:, None, None, :]
    ).reshape(scaled.size(0), _assignment_stencil_size(assignment) ** 3)
    return base, weights



def _assignment_window_1d(
    *,
    mesh_size: int,
    assignment: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    exponent = _assignment_stencil_size(assignment)
    freq = _fft_integer_frequencies(mesh_size, device=device, dtype=dtype)
    return torch.sinc(freq / float(mesh_size)).pow(exponent)



def _apply_mesh_boundary(idx: torch.Tensor, *, mesh_size: int, boundary: str) -> torch.Tensor:
    shape = mesh_shape(mesh_size)
    if boundary == "periodic":
        return torch.remainder(idx, idx.new_tensor(shape))
    idx_wrapped = idx.clone()
    idx_wrapped[..., 0] = torch.remainder(idx_wrapped[..., 0], shape[0])
    idx_wrapped[..., 1] = torch.remainder(idx_wrapped[..., 1], shape[1])
    idx_wrapped[..., 2] = idx_wrapped[..., 2].clamp(0, shape[2] - 1)
    return idx_wrapped



def _spread_source_to_mesh(
    frac: torch.Tensor,
    source: torch.Tensor,
    *,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    boundary: str,
) -> torch.Tensor:
    channels = int(source.size(1))
    shape = mesh_shape(mesh_size)
    mesh = source.new_zeros((*shape, channels))
    flat_mesh = mesh.view(-1, channels)
    scaled = frac * frac.new_tensor(shape)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * shape[1]) + idx[..., 1]) * shape[2] + idx[..., 2]
    flat_mesh.scatter_add_(
        0,
        flat_idx.reshape(-1, 1).expand(-1, channels),
        (source.unsqueeze(1) * stencil_weights.unsqueeze(-1)).reshape(-1, channels),
    )
    return mesh



def _spread_source_to_mesh_batched(
    frac: torch.Tensor,
    batch: torch.Tensor,
    source: torch.Tensor,
    *,
    num_graphs: int,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    boundary: str,
) -> torch.Tensor:
    channels = int(source.size(1))
    shape = mesh_shape(mesh_size)
    mesh_points = math.prod(shape)
    flat_mesh = source.new_zeros((num_graphs * mesh_points, channels))
    scaled = frac * frac.new_tensor(shape)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * shape[1]) + idx[..., 1]) * shape[2] + idx[..., 2]
    graph_offset = batch.to(dtype=flat_idx.dtype).unsqueeze(-1) * mesh_points
    flat_idx = flat_idx + graph_offset
    flat_mesh.scatter_add_(
        0,
        flat_idx.reshape(-1, 1).expand(-1, channels),
        (source.unsqueeze(1) * stencil_weights.unsqueeze(-1)).reshape(-1, channels),
    )
    return flat_mesh.view(num_graphs, *shape, channels)



def _gather_source_from_mesh(
    frac: torch.Tensor,
    mesh: torch.Tensor,
    *,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    boundary: str,
) -> torch.Tensor:
    channels = int(mesh.size(-1))
    shape = mesh_shape(mesh_size)
    flat_mesh = mesh.view(-1, channels)
    scaled = frac * frac.new_tensor(shape)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * shape[1]) + idx[..., 1]) * shape[2] + idx[..., 2]
    gathered = flat_mesh.index_select(0, flat_idx.reshape(-1)).reshape(frac.size(0), _assignment_stencil_size(assignment) ** 3, channels)
    return (gathered * stencil_weights.unsqueeze(-1)).sum(dim=1)



def _gather_source_from_mesh_batched(
    frac: torch.Tensor,
    batch: torch.Tensor,
    mesh: torch.Tensor,
    *,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    boundary: str,
) -> torch.Tensor:
    channels = int(mesh.size(-1))
    shape = mesh_shape(mesh_size)
    mesh_points = math.prod(shape)
    flat_mesh = mesh.reshape(-1, channels)
    scaled = frac * frac.new_tensor(shape)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * shape[1]) + idx[..., 1]) * shape[2] + idx[..., 2]
    graph_offset = batch.to(dtype=flat_idx.dtype).unsqueeze(-1) * mesh_points
    flat_idx = flat_idx + graph_offset
    gathered = flat_mesh.index_select(0, flat_idx.reshape(-1)).reshape(frac.size(0), _assignment_stencil_size(assignment) ** 3, channels)
    return (gathered * stencil_weights.unsqueeze(-1)).sum(dim=1)



class RadialSpectralFilter(nn.Module):
    """Learnable radial filter that modulates a Coulomb-like k-space kernel."""

    def __init__(self, hidden_dim: int, k_norm_floor: float = 1.0e-6):
        super().__init__()
        self.k_norm_floor = float(k_norm_floor)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, k_norms: torch.Tensor) -> torch.Tensor:
        safe_k = k_norms.clamp_min(self.k_norm_floor)
        x0 = torch.log1p(safe_k)
        x1 = x0 * x0
        x2 = torch.reciprocal(safe_k)
        x = torch.stack([x0, x1, x2], dim=-1)
        learned_scale = torch.nn.functional.softplus(self.net(x).squeeze(-1))
        base_kernel = 4.0 * math.pi / (safe_k * safe_k)
        return base_kernel * learned_scale



class ReciprocalGreenKernel(nn.Module):
    """Base Poisson kernel with an optional learnable radial modifier."""

    def __init__(self, *, green_mode: str, hidden_dim: int, k_norm_floor: float = 1.0e-6):
        super().__init__()
        if green_mode not in {"poisson", "learned_poisson"}:
            raise ValueError(f"Unsupported long-range green mode: {green_mode!r}")
        self.green_mode = str(green_mode)
        self.k_norm_floor = float(k_norm_floor)
        self.learned_filter = (
            RadialSpectralFilter(hidden_dim=hidden_dim, k_norm_floor=k_norm_floor)
            if self.green_mode == "learned_poisson"
            else None
        )

    def forward(self, k_norms: torch.Tensor) -> torch.Tensor:
        if self.learned_filter is not None:
            return self.learned_filter(k_norms)
        safe_k = k_norms.clamp_min(self.k_norm_floor)
        return 4.0 * math.pi / (safe_k * safe_k)



class ReciprocalSpectralKernel3D(nn.Module):
    """Direct k-space prototype for periodic reciprocal-space long-range energy."""

    def __init__(
        self,
        *,
        kmax: int = 2,
        filter_hidden_dim: int = 64,
        include_k0: bool = False,
        reciprocal_backend: str = "direct_kspace",
        energy_partition: str = "potential",
        k_norm_floor: float = 1.0e-6,
    ):
        super().__init__()
        if int(kmax) < 0:
            raise ValueError(f"kmax must be >= 0, got {kmax}")
        if reciprocal_backend != "direct_kspace":
            raise ValueError(f"Unsupported reciprocal backend: {reciprocal_backend!r}")
        if energy_partition not in ("potential", "uniform"):
            raise ValueError(f"Unsupported reciprocal energy partition: {energy_partition!r}")
        self.kmax = int(kmax)
        self.include_k0 = bool(include_k0)
        self.reciprocal_backend = str(reciprocal_backend)
        self.energy_partition = str(energy_partition)
        self.k_norm_floor = float(k_norm_floor)
        self.register_buffer("integer_k_lattice", _build_integer_k_lattice(self.kmax, self.include_k0))
        if self.integer_k_lattice.shape[0] == 0:
            raise ValueError("ReciprocalSpectralKernel3D produced an empty k-lattice; increase kmax or enable include_k0")
        self.spectral_filter = RadialSpectralFilter(hidden_dim=filter_hidden_dim, k_norm_floor=k_norm_floor)

    @property
    def num_k(self) -> int:
        return int(self.integer_k_lattice.shape[0])

    def build_k_lattice(self, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_cells = _inverse_3x3(cell)
        k_lattice = self.integer_k_lattice.to(device=cell.device, dtype=cell.dtype)
        k_cart = 2.0 * math.pi * torch.einsum("kd,bdh->bkh", k_lattice, inv_cells.transpose(-1, -2))
        k_norms = _safe_vector_norm(k_cart, dim=-1, floor=self.k_norm_floor)
        volumes = torch.abs(_det_3x3(cell)).clamp_min(self.k_norm_floor)
        return k_cart, k_norms, volumes

    def compute_structure_factor(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_cells = _inverse_3x3(cell)
        atom_inv_cells = inv_cells.index_select(0, batch)
        frac = torch.einsum("ni,nij->nj", pos, atom_inv_cells)
        k_lattice = self.integer_k_lattice.to(device=pos.device, dtype=pos.dtype)
        phases = 2.0 * math.pi * torch.matmul(frac, k_lattice.transpose(0, 1))
        cos_phase = torch.cos(phases)
        sin_phase = torch.sin(phases)

        graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=pos.dtype)
        weighted_cos = source.unsqueeze(1) * cos_phase.unsqueeze(-1)
        weighted_sin = source.unsqueeze(1) * sin_phase.unsqueeze(-1)
        structure_cos = torch.einsum("nb,nkc->bkc", graph_mask, weighted_cos)
        structure_sin = torch.einsum("nb,nkc->bkc", graph_mask, weighted_sin)
        return structure_cos, structure_sin, cos_phase, sin_phase

    def apply_spectral_filter(self, structure_cos: torch.Tensor, structure_sin: torch.Tensor, k_norms: torch.Tensor) -> torch.Tensor:
        spectral_weights = self.spectral_filter(k_norms)
        if not self.include_k0:
            spectral_weights = torch.where(
                k_norms > self.k_norm_floor,
                spectral_weights,
                torch.zeros_like(spectral_weights),
            )
        return spectral_weights

    def forward(self, pos: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(batch, minlength=cell.size(0)).to(dtype=pos.dtype).clamp_min(1.0)
        k_cart, k_norms, volumes = self.build_k_lattice(cell)
        structure_cos, structure_sin, cos_phase, sin_phase = self.compute_structure_factor(pos, batch, cell, source)
        del k_cart
        spectral_weights = self.apply_spectral_filter(structure_cos, structure_sin, k_norms)

        source_cos = structure_cos.index_select(0, batch)
        source_sin = structure_sin.index_select(0, batch)
        spectral_per_atom = spectral_weights.index_select(0, batch)
        volume_per_atom = volumes.index_select(0, batch).unsqueeze(-1)
        potential = (
            spectral_per_atom.unsqueeze(-1)
            * (source_cos * cos_phase.unsqueeze(-1) + source_sin * sin_phase.unsqueeze(-1))
        ).sum(dim=1) / volume_per_atom

        if self.energy_partition == "potential":
            atom_energy = 0.5 * (source * potential).sum(dim=-1, keepdim=True)
            return atom_energy

        graph_total = 0.5 * (
            spectral_weights.unsqueeze(-1) * (structure_cos.square() + structure_sin.square())
        ).sum(dim=(1, 2)) / volumes
        atom_energy = graph_total.index_select(0, batch).unsqueeze(-1) / counts.index_select(0, batch).unsqueeze(-1)
        return atom_energy



class MeshLongRangeKernel3D(nn.Module):
    """Mesh/FFT reciprocal kernel with periodic/slab boundary support."""

    def __init__(
        self,
        *,
        mesh_size: int | tuple[int, int, int] = 16,
        filter_hidden_dim: int = 64,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        energy_partition: str = "potential",
        green_mode: str = "poisson",
        assignment: str = "pcs",
        full_ewald: bool = False,
        reciprocal_only: bool = False,
        k_norm_floor: float = 1.0e-6,
        mesh_backend: str = "auto",
        fuse_multipole_fft: bool = True,
        smearing_sigma: float | None = None,
    ):
        super().__init__()
        if boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported reciprocal mesh boundary: {boundary!r}")
        if energy_partition not in {"potential", "uniform"}:
            raise ValueError(f"Unsupported reciprocal energy partition: {energy_partition!r}")
        if assignment not in {"cic", "tsc", "pcs"}:
            raise ValueError(f"Unsupported mesh assignment: {assignment!r}")
        self.mesh_size = normalize_mesh_size(mesh_size)
        if mesh_backend not in {"auto", "hip", "reference"}:
            raise ValueError("mesh_backend must be auto, hip or reference")
        self.mesh_backend = mesh_backend
        self.fuse_multipole_fft = bool(fuse_multipole_fft)
        self.boundary = str(boundary)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.energy_partition = str(energy_partition)
        self.green_mode = str(green_mode)
        self.assignment = str(assignment)
        self.full_ewald = bool(full_ewald)
        if smearing_sigma is not None:
            smearing_sigma = float(smearing_sigma)
            if not math.isfinite(smearing_sigma) or smearing_sigma <= 0.0:
                raise ValueError("smearing_sigma must be finite and positive, or None for legacy box-dependent screening")
            if not self.full_ewald:
                raise ValueError("smearing_sigma requires full_ewald=True to enable Gaussian screening")
        self.smearing_sigma = smearing_sigma
        # A learned reciprocal correction lets the local model fit complementary
        # terms. An exact Coulomb sum also needs real-space, self and background
        # terms; the reciprocal correction alone does not reproduce that sum.
        self.reciprocal_only = bool(reciprocal_only)
        self.k_norm_floor = float(k_norm_floor)
        self.ewald_alpha_prefactor = 5.0
        self.assignment_window_floor = 1.0e-6
        self.green_kernel = ReciprocalGreenKernel(
            green_mode=self.green_mode,
            hidden_dim=int(filter_hidden_dim),
            k_norm_floor=self.k_norm_floor,
        )
        assignment_offsets = _build_assignment_offsets(self.assignment)
        self.register_buffer("assignment_offsets", assignment_offsets, persistent=False)
        self.register_buffer("real_space_shift_index", self._init_real_space_shifts(), persistent=False)
        periodic_axis_index = torch.tensor(
            [axis for axis, enabled in enumerate(self._periodic_axes()) if enabled],
            dtype=torch.long,
        )
        self.register_buffer("periodic_axis_index", periodic_axis_index, persistent=False)
        self.register_buffer("_assignment_window_cache", torch.empty(0), persistent=False)

    @property
    def mesh_shape(self) -> tuple[int, int, int]:
        return mesh_shape(self.mesh_size)

    @property
    def num_k(self) -> int:
        total = math.prod(self.mesh_shape)
        return total if self.include_k0 else max(total - 1, 0)

    def _periodic_axes(self) -> tuple[bool, bool, bool]:
        if self.boundary == "periodic":
            return True, True, True
        return True, True, False

    def _estimate_real_cutoff_batched(self, cell: torch.Tensor) -> torch.Tensor:
        periodic_vectors = cell.index_select(1, self.periodic_axis_index.to(device=cell.device))
        periodic_lengths = torch.linalg.vector_norm(periodic_vectors, dim=-1)
        return 0.5 * periodic_lengths.min(dim=-1).values.clamp_min(self.k_norm_floor)

    def _estimate_ewald_alpha(self, real_cutoff: torch.Tensor) -> torch.Tensor:
        # exp(-k^2/(4 alpha^2)) = exp(-sigma^2 k^2/2). A fixed
        # physical sigma makes replication leave the reciprocal functional
        # unchanged. None preserves existing checkpoint behavior.
        sigma = getattr(self, "smearing_sigma", None)
        if sigma is not None:
            return torch.full_like(real_cutoff, 1.0 / (math.sqrt(2.0) * sigma))
        return real_cutoff.new_tensor(self.ewald_alpha_prefactor) / real_cutoff.clamp_min(self.k_norm_floor)

    def _init_real_space_shifts(self) -> torch.Tensor:
        ranges: list[torch.Tensor] = []
        for is_periodic in self._periodic_axes():
            values = [-1, 0, 1] if is_periodic else [0]
            ranges.append(torch.tensor(values, dtype=torch.long))
        return torch.cartesian_prod(*ranges)

    def _build_assignment_window(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = self._assignment_window_cache
        if tuple(cached.shape) == self.mesh_shape and cached.device == device and cached.dtype == dtype:
            return cached
        windows = [_assignment_window_1d(mesh_size=size, assignment=self.assignment,
                                         device=device, dtype=dtype)
                   for size in self.mesh_shape]
        wx, wy, wz = torch.meshgrid(*windows, indexing="ij")
        window = wx * wy * wz
        self._assignment_window_cache = window
        return window

    def build_k_norms_batched(self, cell: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        effective_cell = _effective_cell_for_boundary(
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            dtype=dtype,
        )
        frequencies = [_fft_integer_frequencies(size, device=cell.device, dtype=dtype)
                       for size in self.mesh_shape]
        kx, ky, kz = torch.meshgrid(*frequencies, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        inv_cell = _inverse_3x3(effective_cell)
        k_cart = 2.0 * math.pi * torch.einsum("kd,bdh->bkh", integer_k, inv_cell.transpose(-1, -2))
        k_norms = _safe_vector_norm(k_cart, dim=-1, floor=self.k_norm_floor).reshape(
            cell.size(0), *self.mesh_shape
        )
        volume = torch.abs(_det_3x3(effective_cell)).clamp_min(self.k_norm_floor)
        return k_norms, volume

    def _build_k_cart_flat_batched(
        self,
        cell: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        effective_cell = _effective_cell_for_boundary(
            cell, boundary=self.boundary, slab_padding_factor=self.slab_padding_factor, dtype=dtype
        )
        frequencies = [_fft_integer_frequencies(size, device=cell.device, dtype=dtype)
                       for size in self.mesh_shape]
        kx, ky, kz = torch.meshgrid(*frequencies, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        inv_cell = _inverse_3x3(effective_cell)
        k_cart = 2.0 * math.pi * torch.einsum("kd,bdh->bkh", integer_k, inv_cell.transpose(-1, -2))
        k_norms = _safe_vector_norm(k_cart, dim=-1, floor=self.k_norm_floor)
        volume = torch.abs(_det_3x3(effective_cell)).clamp_min(self.k_norm_floor)
        return k_cart, k_norms, volume

    def _graph_counts(self, batch: torch.Tensor, cell: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if cell.shape[0] == 1:
            return torch.full((1,), max(batch.shape[0], 1), device=batch.device, dtype=dtype)
        return torch.zeros(cell.size(0), device=batch.device, dtype=dtype).index_add(
            0, batch, torch.ones(batch.shape[0], device=batch.device, dtype=dtype)
        ).clamp_min(1.0)

    def _native_mesh(self, value: torch.Tensor) -> bool:
        backend = getattr(self, "mesh_backend", "auto")
        if backend == "reference":
            return False
        from ictc_o3.backends.k100.hip_electrostatics import supported
        enabled = supported(value)
        if backend == "hip" and not enabled:
            raise ValueError("HIP electrostatics requires K100 gfx926 FP32 or FP64")
        return enabled

    def _spread(self, frac, batch, source, num_graphs, *, channels_first=False):
        if self._native_mesh(source):
            from ictc_o3.backends.k100.hip_electrostatics import spread
            return spread(frac, batch, source, mesh_size=self.mesh_size,
                          num_graphs=num_graphs, assignment=self.assignment,
                          boundary=self.boundary, channels_first=channels_first)
        mesh = _spread_source_to_mesh_batched(
            frac, batch, source, num_graphs=num_graphs, mesh_size=self.mesh_size,
            assignment=self.assignment, assignment_offsets=self.assignment_offsets,
            boundary=self.boundary)
        return mesh.permute(0, 4, 1, 2, 3).contiguous() if channels_first else mesh

    def _gather(self, frac, batch, mesh):
        if self._native_mesh(mesh):
            from ictc_o3.backends.k100.hip_electrostatics import gather
            return gather(frac, batch, mesh, mesh_size=self.mesh_size,
                          num_graphs=mesh.shape[0], assignment=self.assignment,
                          boundary=self.boundary)
        return _gather_source_from_mesh_batched(
            frac, batch, mesh, mesh_size=self.mesh_size, assignment=self.assignment,
            assignment_offsets=self.assignment_offsets, boundary=self.boundary)

    def multipole_energy(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        source: torch.Tensor,
        dipole: torch.Tensor | None,
        quadrupole: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reciprocal energy from latent monopole(+dipole)(+quadrupole) via the |S(k)|^2 PME route.

        S(k) = FFT(spread q) + i k . FFT(spread mu) - 1/2 k . FFT(spread Q) . k
        E    = (1/2V) sum_{k!=0} [green(k) / |W(k)|^2] |S(k)|^2          (W = assignment window)

        This route computes the energy directly in k-space (no iFFT), so it is free of the
        iFFT 1/N normalization that the potential route needs compensated, and it reduces to the
        plain reciprocal monopole sum when dipole/quadrupole are None. Per-graph, uniform partition.
        """
        return self._multipole_energy_batched(pos, batch, cell, source, dipole, quadrupole)

    def _multipole_energy_batched(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        source: torch.Tensor,
        dipole: torch.Tensor | None,
        quadrupole: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._native_mesh(source) and getattr(self, "fuse_multipole_fft", True):
            from ictc_o3.backends.k100.hip_reciprocal_fft import multipole_energy
            return multipole_energy(self, pos, batch, cell, source, dipole, quadrupole,
                                    compact_quadrupole=True)
        src_c = source.size(1)
        frac = _prepare_frac_for_boundary_batched(
            pos,
            batch,
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
        )
        k_cart, k_norms_flat, volume = self._build_k_cart_flat_batched(cell, dtype=pos.dtype)
        green = self.green_kernel(k_norms_flat)
        window = self._build_assignment_window(device=pos.device, dtype=pos.dtype).reshape(1, -1)
        wdeconv = torch.reciprocal(window.clamp_min(self.assignment_window_floor).square())
        spectral = green / volume.unsqueeze(-1) * wdeconv
        if self.full_ewald:
            real_cutoff = self._estimate_real_cutoff_batched(cell.to(dtype=pos.dtype))
            alpha = self._estimate_ewald_alpha(real_cutoff)
            spectral = spectral * torch.exp(-(k_norms_flat.square()) / (4.0 * alpha.unsqueeze(-1).square()))
        spectral = torch.where(k_norms_flat > self.k_norm_floor, spectral, torch.zeros_like(spectral))

        def _spread_fft(field: torch.Tensor) -> torch.Tensor:
            mesh = self._spread(frac, batch, field, cell.size(0))
            return torch.fft.fftn(mesh, dim=(1, 2, 3)).reshape(cell.size(0), -1, field.size(1)).contiguous()

        S = _spread_fft(source).reshape(cell.size(0), -1, src_c)
        S_real = S.real
        S_imag = S.imag
        if dipole is not None:
            mut = _spread_fft(dipole.reshape(source.size(0), src_c * 3)).reshape(cell.size(0), -1, src_c, 3)
            dipole_term = torch.einsum("bkx,bksx->bks", k_cart.to(mut.dtype), mut)
            S_real = S_real - dipole_term.imag
            S_imag = S_imag + dipole_term.real
        if quadrupole is not None:
            qt = _spread_fft(quadrupole.reshape(source.size(0), src_c * 9)).reshape(cell.size(0), -1, src_c, 3, 3)
            k_complex = k_cart.to(qt.dtype)
            quadrupole_term = torch.einsum("bkx,bksxy,bky->bks", k_complex, qt, k_complex)
            S_real = S_real - 0.5 * quadrupole_term.real
            S_imag = S_imag - 0.5 * quadrupole_term.imag
        e_graph = 0.5 * (spectral.unsqueeze(-1) * (S_real.square() + S_imag.square())).sum(dim=(1, 2))
        counts = self._graph_counts(batch, cell, dtype=source.dtype)
        return (e_graph.index_select(0, batch) / counts.index_select(0, batch)).unsqueeze(-1)

    def apply_green_kernel_batched(
        self,
        mesh: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        if self._native_mesh(mesh):
            from ictc_o3.backends.k100.hip_reciprocal_fft import scalar_potential
            return scalar_potential(self, mesh, cell)
        mesh_dtype = mesh.dtype
        mesh_complex = torch.fft.fftn(mesh, dim=(1, 2, 3))
        k_norms, volume = self.build_k_norms_batched(cell, dtype=mesh_dtype)
        spectral_weights = self.green_kernel(k_norms) / volume.view(-1, 1, 1, 1)
        real_cutoff = None
        alpha = None
        if self.full_ewald:
            real_cutoff = self._estimate_real_cutoff_batched(cell.to(dtype=mesh_dtype))
            alpha = self._estimate_ewald_alpha(real_cutoff)
            spectral_weights = spectral_weights * torch.exp(
                -(k_norms.square()) / (4.0 * alpha.view(-1, 1, 1, 1).square())
            )
        assignment_window = self._build_assignment_window(device=cell.device, dtype=mesh_dtype)
        assignment_scale = torch.reciprocal(assignment_window.clamp_min(self.assignment_window_floor).square())
        spectral_weights = spectral_weights * assignment_scale.unsqueeze(0)
        if self.full_ewald or (not self.include_k0):
            spectral_weights = torch.where(
                k_norms > self.k_norm_floor,
                spectral_weights,
                torch.zeros_like(spectral_weights),
            )
        filtered = torch.fft.ifftn(mesh_complex * spectral_weights.unsqueeze(-1), dim=(1, 2, 3))
        # Compensate torch.fft.ifftn's 1/N normalization (N = Mx*My*Mz). The forward FFT of
        # the spread charges already yields the structure factor S(k), so the iFFT's 1/N factor
        # is spurious here.
        filtered = filtered * float(math.prod(self.mesh_shape))
        return filtered.real, alpha, volume, real_cutoff

    def _compute_real_space_potential_batched(
        self,
        pos: torch.Tensor,
        source: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        *,
        alpha: torch.Tensor,
        real_cutoff: torch.Tensor,
    ) -> torch.Tensor:
        shift_index = self.real_space_shift_index.to(device=cell.device)
        atom_cell = cell.index_select(0, batch).to(dtype=pos.dtype)
        shift_cart = torch.einsum("sd,ndh->nsh", shift_index.to(dtype=pos.dtype), atom_cell)
        disp = pos.unsqueeze(1).unsqueeze(2) - pos.unsqueeze(0).unsqueeze(2) - shift_cart.unsqueeze(1)
        distance = _safe_vector_norm(disp, dim=-1, floor=self.k_norm_floor)
        alpha_atom = alpha.index_select(0, batch).to(dtype=pos.dtype).view(-1, 1, 1)
        cutoff_atom = real_cutoff.index_select(0, batch).to(dtype=pos.dtype).view(-1, 1, 1)
        kernel = torch.special.erfc(alpha_atom * distance) / distance.clamp_min(self.k_norm_floor)
        same_graph = (batch.unsqueeze(1) == batch.unsqueeze(0)).unsqueeze(-1)
        valid = (distance <= cutoff_atom) & same_graph
        atom_ids = torch.arange(pos.size(0), device=pos.device)
        zero_shift = (shift_index == 0).all(dim=1)
        self_mask = (atom_ids.unsqueeze(1) == atom_ids.unsqueeze(0)).unsqueeze(-1) & zero_shift.view(1, 1, -1)
        kernel = kernel * (valid & (~self_mask)).to(dtype=source.dtype)
        pair_kernel = kernel.sum(dim=-1)
        return torch.matmul(pair_kernel, source)

    def forward(self, pos: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self._forward_batched(pos, batch, cell, source)

    def _forward_batched(self, pos: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        frac = _prepare_frac_for_boundary_batched(
            pos,
            batch,
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
        )
        mesh = self._spread(frac, batch, source, cell.size(0))
        potential_mesh, alpha, effective_volume, real_cutoff = self.apply_green_kernel_batched(mesh, cell)
        reciprocal_potential = self._gather(frac, batch, potential_mesh)
        total_potential = reciprocal_potential
        if self.full_ewald and not self.reciprocal_only:
            assert alpha is not None and real_cutoff is not None
            real_space_potential = self._compute_real_space_potential_batched(
                pos,
                source,
                batch,
                cell,
                alpha=alpha,
                real_cutoff=real_cutoff,
            )
            alpha_atom = alpha.index_select(0, batch).to(dtype=source.dtype).unsqueeze(-1)
            self_potential = (-2.0 * alpha_atom / math.sqrt(math.pi)) * source
            graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
            graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=source.dtype)
            net_source = torch.einsum("nb,nc->bc", graph_mask, source)
            background_graph = (
                -math.pi
                * net_source
                / (alpha.to(dtype=source.dtype).square().unsqueeze(-1) * effective_volume.to(dtype=source.dtype).unsqueeze(-1))
            )
            background_potential = background_graph.index_select(0, batch)
            total_potential = reciprocal_potential + real_space_potential + self_potential + background_potential
        atom_energy = 0.5 * (source * total_potential).sum(dim=-1, keepdim=True)
        if self.energy_partition == "uniform":
            graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
            graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=source.dtype)
            graph_total = torch.einsum("nb,nc->bc", graph_mask, atom_energy).squeeze(-1)
            counts = graph_mask.sum(dim=0).clamp_min(1.0)
            atom_energy = (graph_total.index_select(0, batch) / counts.index_select(0, batch)).unsqueeze(-1)
        return atom_energy

