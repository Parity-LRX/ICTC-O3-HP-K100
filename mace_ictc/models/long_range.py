from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mace_ictc.utils.checkpoint_metadata import (
    derive_long_range_far_max_radius_multiplier,
)


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


class LatentSourceHead(nn.Module):
    """Map node invariant features to latent reciprocal-space sources."""

    def __init__(self, feature_dim: int, hidden_dim: int, source_channels: int = 1):
        super().__init__()
        self.source_channels = int(source_channels)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.source_channels),
        )

    def forward(self, invariant_features: torch.Tensor) -> torch.Tensor:
        return self.net(invariant_features)


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
    ).reshape(scaled.size(0), -1)
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
    if boundary == "periodic":
        return torch.remainder(idx, mesh_size)
    idx_wrapped = idx.clone()
    idx_wrapped[..., 0] = torch.remainder(idx_wrapped[..., 0], mesh_size)
    idx_wrapped[..., 1] = torch.remainder(idx_wrapped[..., 1], mesh_size)
    idx_wrapped[..., 2] = idx_wrapped[..., 2].clamp(0, mesh_size - 1)
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
    mesh = source.new_zeros((mesh_size, mesh_size, mesh_size, channels))
    flat_mesh = mesh.view(-1, channels)
    scaled = frac * float(mesh_size)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * mesh_size) + idx[..., 1]) * mesh_size + idx[..., 2]
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
    mesh_points = mesh_size * mesh_size * mesh_size
    flat_mesh = source.new_zeros((num_graphs * mesh_points, channels))
    scaled = frac * float(mesh_size)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * mesh_size) + idx[..., 1]) * mesh_size + idx[..., 2]
    graph_offset = batch.to(dtype=flat_idx.dtype).unsqueeze(-1) * mesh_points
    flat_idx = flat_idx + graph_offset
    flat_mesh.scatter_add_(
        0,
        flat_idx.reshape(-1, 1).expand(-1, channels),
        (source.unsqueeze(1) * stencil_weights.unsqueeze(-1)).reshape(-1, channels),
    )
    return flat_mesh.view(num_graphs, mesh_size, mesh_size, mesh_size, channels)


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
    flat_mesh = mesh.view(-1, channels)
    scaled = frac * float(mesh_size)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * mesh_size) + idx[..., 1]) * mesh_size + idx[..., 2]
    gathered = flat_mesh.index_select(0, flat_idx.reshape(-1)).reshape(frac.size(0), -1, channels)
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
    mesh_points = mesh_size * mesh_size * mesh_size
    flat_mesh = mesh.reshape(-1, channels)
    scaled = frac * float(mesh_size)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary=boundary,
    )
    flat_idx = ((idx[..., 0] * mesh_size) + idx[..., 1]) * mesh_size + idx[..., 2]
    graph_offset = batch.to(dtype=flat_idx.dtype).unsqueeze(-1) * mesh_points
    flat_idx = flat_idx + graph_offset
    gathered = flat_mesh.index_select(0, flat_idx.reshape(-1)).reshape(frac.size(0), -1, channels)
    return (gathered * stencil_weights.unsqueeze(-1)).sum(dim=1)


def _dipole_pme_kernel_mesh(
    *,
    mesh_size: int,
    k_cart: torch.Tensor,
    k2: torch.Tensor,
    spectral: torch.Tensor,
    k_norm_floor: float,
) -> torch.Tensor:
    k_complex = k_cart.to(torch.complex128 if k_cart.dtype == torch.float64 else torch.complex64)
    k2_safe = k2.clamp_min(float(k_norm_floor) * float(k_norm_floor)).to(k_complex.dtype)
    kernel_fft = (
        spectral.to(k_complex.dtype).unsqueeze(-1).unsqueeze(-1)
        * k_complex.unsqueeze(-1)
        * k_complex.unsqueeze(-2)
        / k2_safe.unsqueeze(-1).unsqueeze(-1)
    )
    return torch.fft.ifftn(
        kernel_fft.reshape(mesh_size, mesh_size, mesh_size, 9),
        dim=(0, 1, 2),
    ).real.reshape(mesh_size * mesh_size * mesh_size, 3, 3) * (float(mesh_size) ** 3)


def _dipole_pme_kernel_mesh_batched(
    *,
    mesh_size: int,
    k_cart: torch.Tensor,
    k2: torch.Tensor,
    spectral: torch.Tensor,
    k_norm_floor: float,
) -> torch.Tensor:
    k_complex = k_cart.to(torch.complex128 if k_cart.dtype == torch.float64 else torch.complex64)
    k2_safe = k2.clamp_min(float(k_norm_floor) * float(k_norm_floor)).to(k_complex.dtype)
    kernel_fft = (
        spectral.to(k_complex.dtype).unsqueeze(-1).unsqueeze(-1)
        * k_complex.unsqueeze(-1)
        * k_complex.unsqueeze(-2)
        / k2_safe.unsqueeze(-1).unsqueeze(-1)
    )
    num_graphs = int(k_cart.size(0))
    return torch.fft.ifftn(
        kernel_fft.reshape(num_graphs, mesh_size, mesh_size, mesh_size, 9),
        dim=(1, 2, 3),
    ).real.reshape(num_graphs * mesh_size * mesh_size * mesh_size, 3, 3) * (float(mesh_size) ** 3)


def _dipole_pme_self_blocks(
    frac: torch.Tensor,
    *,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    kernel_mesh_flat: torch.Tensor,
    graph_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact discrete spread/filter/gather self block for the PME dipole tensor."""
    if frac.numel() == 0:
        return frac.new_zeros(0, 3, 3)
    scaled = frac * float(mesh_size)
    base, stencil_weights = _assignment_weights_from_scaled(scaled, assignment)
    idx = _apply_mesh_boundary(
        base.unsqueeze(1) + assignment_offsets.to(device=base.device).unsqueeze(0),
        mesh_size=mesh_size,
        boundary="periodic",
    )
    ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
    stencil = int(stencil_weights.size(1))
    max_pairs_per_chunk = 262144
    chunk = max(1, min(int(frac.size(0)), max_pairs_per_chunk // max(stencil * stencil, 1)))
    blocks: list[torch.Tensor] = []
    mesh_points = mesh_size * mesh_size * mesh_size
    for start in range(0, int(frac.size(0)), chunk):
        stop = min(start + chunk, int(frac.size(0)))
        ix_c = ix[start:stop]
        iy_c = iy[start:stop]
        iz_c = iz[start:stop]
        dx = torch.remainder(ix_c.unsqueeze(2) - ix_c.unsqueeze(1), mesh_size)
        dy = torch.remainder(iy_c.unsqueeze(2) - iy_c.unsqueeze(1), mesh_size)
        dz = torch.remainder(iz_c.unsqueeze(2) - iz_c.unsqueeze(1), mesh_size)
        flat = (dx * mesh_size + dy) * mesh_size + dz
        if graph_offset is not None:
            flat = flat + graph_offset[start:stop].view(-1, 1, 1) * mesh_points
        values = kernel_mesh_flat.index_select(0, flat.reshape(-1)).reshape(stop - start, stencil, stencil, 3, 3)
        weights = stencil_weights[start:stop]
        pair_weights = weights.unsqueeze(2) * weights.unsqueeze(1)
        blocks.append((values * pair_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=(1, 2)))
    return torch.cat(blocks, dim=0)


def build_periodic_dipole_pme_kernel(
    *,
    cell: torch.Tensor,
    mesh_size: int,
    assignment: str,
    device: torch.device,
    dtype: torch.dtype,
    k_norm_floor: float = 1.0e-6,
    assignment_window_floor: float = 1.0e-6,
    ewald_alpha_prefactor: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute the periodic PME dipole-tensor spectral kernel for repeated matvecs."""
    mesh_size = int(mesh_size)
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")

    freq = _fft_integer_frequencies(mesh_size, device=device, dtype=dtype)
    kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
    integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
    inv_cell = _inverse_3x3(cell.to(dtype=dtype))
    k_cart = 2.0 * math.pi * torch.einsum("kd,dh->kh", integer_k, inv_cell.transpose(-1, -2))
    k_norm = _safe_vector_norm(k_cart, dim=-1, floor=k_norm_floor)
    k2 = k_norm.square()
    volume = _det_3x3(cell.to(dtype=dtype)).abs().clamp_min(k_norm_floor)

    window_1d = _assignment_window_1d(
        mesh_size=mesh_size,
        assignment=assignment,
        device=device,
        dtype=dtype,
    )
    wx, wy, wz = torch.meshgrid(window_1d, window_1d, window_1d, indexing="ij")
    window = (wx * wy * wz).reshape(-1)
    wdeconv = torch.reciprocal(window.clamp_min(assignment_window_floor).square())
    real_cutoff = (0.5 * torch.linalg.vector_norm(cell.to(dtype=dtype), dim=-1).min()).clamp_min(k_norm_floor)
    ewald_alpha = real_cutoff.new_tensor(float(ewald_alpha_prefactor)) / real_cutoff
    # MBD uses the Hessian convention T = grad grad (1/r) = (3 rr - I)/r^3.
    # With the FFT exp(-ik.r) structure-factor convention, the smooth reciprocal
    # dipole tensor carries a negative k_a k_b/k^2 sign.
    spectral = -(4.0 * math.pi / volume) * wdeconv * torch.exp(-k2 / (4.0 * ewald_alpha.square()))
    spectral = torch.where(k_norm > k_norm_floor, spectral, torch.zeros_like(spectral))
    return k_cart, k2, spectral


def build_periodic_dipole_pme_kernel_batched(
    *,
    cell: torch.Tensor,
    mesh_size: int,
    assignment: str,
    device: torch.device,
    dtype: torch.dtype,
    k_norm_floor: float = 1.0e-6,
    assignment_window_floor: float = 1.0e-6,
    ewald_alpha_prefactor: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched variant of :func:`build_periodic_dipole_pme_kernel` for multi-graph FFTs."""
    mesh_size = int(mesh_size)
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")

    cell_b = cell.reshape(-1, 3, 3).to(device=device, dtype=dtype)
    freq = _fft_integer_frequencies(mesh_size, device=device, dtype=dtype)
    kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
    integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
    inv_cell = _inverse_3x3(cell_b)
    k_cart = 2.0 * math.pi * torch.einsum("kd,bdh->bkh", integer_k, inv_cell.transpose(-1, -2))
    k_norm = _safe_vector_norm(k_cart, dim=-1, floor=k_norm_floor)
    k2 = k_norm.square()
    volume = _det_3x3(cell_b).abs().clamp_min(k_norm_floor)

    window_1d = _assignment_window_1d(
        mesh_size=mesh_size,
        assignment=assignment,
        device=device,
        dtype=dtype,
    )
    wx, wy, wz = torch.meshgrid(window_1d, window_1d, window_1d, indexing="ij")
    window = (wx * wy * wz).reshape(1, -1)
    wdeconv = torch.reciprocal(window.clamp_min(assignment_window_floor).square())
    real_cutoff = (0.5 * torch.linalg.vector_norm(cell_b, dim=-1).min(dim=-1).values).clamp_min(k_norm_floor)
    ewald_alpha = real_cutoff.new_tensor(float(ewald_alpha_prefactor)) / real_cutoff
    # See build_periodic_dipole_pme_kernel: negative sign matches
    # T = grad grad (1/r), consistent with edge_sparse and mbd.dipole_field.
    spectral = -(4.0 * math.pi / volume).unsqueeze(-1) * wdeconv
    spectral = spectral * torch.exp(-k2 / (4.0 * ewald_alpha.unsqueeze(-1).square()))
    spectral = torch.where(k_norm > k_norm_floor, spectral, torch.zeros_like(spectral))
    return k_cart, k2, spectral


def apply_periodic_dipole_pme_field(
    frac: torch.Tensor,
    dipoles: torch.Tensor,
    *,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    k_cart: torch.Tensor,
    k2: torch.Tensor,
    spectral: torch.Tensor,
    k_norm_floor: float = 1.0e-6,
    kernel_mesh_flat: torch.Tensor | None = None,
    self_blocks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a precomputed periodic PME dipole-tensor kernel to vector dipoles."""
    if dipoles.dim() != 3 or dipoles.size(-1) != 3:
        raise ValueError("apply_periodic_dipole_pme_field expects dipoles with shape (N, C, 3)")
    n_atoms, channels, _ = dipoles.shape
    mesh = _spread_source_to_mesh(
        frac,
        dipoles.reshape(n_atoms, channels * 3),
        mesh_size=mesh_size,
        assignment=assignment,
        assignment_offsets=assignment_offsets,
        boundary="periodic",
    )
    mesh_fft = torch.fft.fftn(mesh, dim=(0, 1, 2)).reshape(-1, channels, 3)
    k_complex = k_cart.to(mesh_fft.dtype)
    k_dot_m = (mesh_fft * k_complex.unsqueeze(1)).sum(dim=-1)
    k2_safe = k2.clamp_min(float(k_norm_floor) * float(k_norm_floor)).to(mesh_fft.dtype).view(-1, 1)
    field_fft = (
        spectral.to(mesh_fft.dtype).view(-1, 1, 1)
        * k_complex.unsqueeze(1)
        * (k_dot_m / k2_safe).unsqueeze(-1)
    )
    field_mesh = torch.fft.ifftn(
        field_fft.reshape(mesh_size, mesh_size, mesh_size, channels * 3),
        dim=(0, 1, 2),
    ).real * (float(mesh_size) ** 3)
    gathered = _gather_source_from_mesh(
        frac,
        field_mesh,
        mesh_size=mesh_size,
        assignment=assignment,
        assignment_offsets=assignment_offsets,
        boundary="periodic",
    ).reshape(n_atoms, channels, 3)
    if self_blocks is None:
        if kernel_mesh_flat is None:
            kernel_mesh_flat = _dipole_pme_kernel_mesh(
                mesh_size=mesh_size,
                k_cart=k_cart,
                k2=k2,
                spectral=spectral,
                k_norm_floor=k_norm_floor,
            )
        self_blocks = _dipole_pme_self_blocks(
            frac,
            mesh_size=mesh_size,
            assignment=assignment,
            assignment_offsets=assignment_offsets,
            kernel_mesh_flat=kernel_mesh_flat,
        )
    self_field = torch.einsum("nab,ncb->nca", self_blocks, dipoles)
    return gathered - self_field


def apply_periodic_dipole_pme_field_batched(
    frac: torch.Tensor,
    batch: torch.Tensor,
    dipoles: torch.Tensor,
    *,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    k_cart: torch.Tensor,
    k2: torch.Tensor,
    spectral: torch.Tensor,
    k_norm_floor: float = 1.0e-6,
    kernel_mesh_flat: torch.Tensor | None = None,
    self_blocks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply precomputed batched periodic PME dipole-tensor kernels."""
    if dipoles.dim() != 3 or dipoles.size(-1) != 3:
        raise ValueError("apply_periodic_dipole_pme_field_batched expects dipoles with shape (N, C, 3)")
    n_atoms, channels, _ = dipoles.shape
    num_graphs = int(k_cart.size(0))
    mesh = _spread_source_to_mesh_batched(
        frac,
        batch,
        dipoles.reshape(n_atoms, channels * 3),
        num_graphs=num_graphs,
        mesh_size=mesh_size,
        assignment=assignment,
        assignment_offsets=assignment_offsets,
        boundary="periodic",
    )
    mesh_fft = torch.fft.fftn(mesh, dim=(1, 2, 3)).reshape(num_graphs, -1, channels, 3)
    k_complex = k_cart.to(mesh_fft.dtype)
    k_dot_m = (mesh_fft * k_complex.unsqueeze(2)).sum(dim=-1)
    k2_safe = k2.clamp_min(float(k_norm_floor) * float(k_norm_floor)).to(mesh_fft.dtype).unsqueeze(-1)
    field_fft = (
        spectral.to(mesh_fft.dtype).unsqueeze(-1).unsqueeze(-1)
        * k_complex.unsqueeze(2)
        * (k_dot_m / k2_safe).unsqueeze(-1)
    )
    field_mesh = torch.fft.ifftn(
        field_fft.reshape(num_graphs, mesh_size, mesh_size, mesh_size, channels * 3),
        dim=(1, 2, 3),
    ).real * (float(mesh_size) ** 3)
    gathered = _gather_source_from_mesh_batched(
        frac,
        batch,
        field_mesh,
        mesh_size=mesh_size,
        assignment=assignment,
        assignment_offsets=assignment_offsets,
        boundary="periodic",
    ).reshape(n_atoms, channels, 3)
    if self_blocks is None:
        if kernel_mesh_flat is None:
            kernel_mesh_flat = _dipole_pme_kernel_mesh_batched(
                mesh_size=mesh_size,
                k_cart=k_cart,
                k2=k2,
                spectral=spectral,
                k_norm_floor=k_norm_floor,
            )
        self_blocks = _dipole_pme_self_blocks(
            frac,
            mesh_size=mesh_size,
            assignment=assignment,
            assignment_offsets=assignment_offsets,
            kernel_mesh_flat=kernel_mesh_flat,
            graph_offset=batch,
        )
    self_field = torch.einsum("nab,ncb->nca", self_blocks, dipoles)
    return gathered - self_field


def periodic_dipole_pme_field(
    frac: torch.Tensor,
    dipoles: torch.Tensor,
    *,
    cell: torch.Tensor,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    k_norm_floor: float = 1.0e-6,
    assignment_window_floor: float = 1.0e-6,
    ewald_alpha_prefactor: float = 5.0,
) -> torch.Tensor:
    """Periodic PME dipole-tensor field shared by electrostatics and MBD prototypes.

    ``dipoles`` has shape ``(N, C, 3)`` and the returned field has the same shape.
    The helper intentionally covers only the periodic smooth reciprocal operator;
    short-range damping and SLQ/logdet logic remain with the caller. The helper
    removes the exact discrete spread/filter/gather self block so the operator
    has no diagonal self action.
    """
    if dipoles.dim() != 3 or dipoles.size(-1) != 3:
        raise ValueError("periodic_dipole_pme_field expects dipoles with shape (N, C, 3)")
    k_cart, k2, spectral = build_periodic_dipole_pme_kernel(
        cell=cell,
        mesh_size=mesh_size,
        assignment=assignment,
        device=dipoles.device,
        dtype=dipoles.dtype,
        k_norm_floor=k_norm_floor,
        assignment_window_floor=assignment_window_floor,
        ewald_alpha_prefactor=ewald_alpha_prefactor,
    )
    return apply_periodic_dipole_pme_field(
        frac,
        dipoles,
        mesh_size=mesh_size,
        assignment=assignment,
        assignment_offsets=assignment_offsets,
        k_cart=k_cart,
        k2=k2,
        spectral=spectral,
        k_norm_floor=k_norm_floor,
    )


def periodic_dipole_pme_field_batched(
    frac: torch.Tensor,
    batch: torch.Tensor,
    dipoles: torch.Tensor,
    *,
    cell: torch.Tensor,
    mesh_size: int,
    assignment: str,
    assignment_offsets: torch.Tensor,
    k_norm_floor: float = 1.0e-6,
    assignment_window_floor: float = 1.0e-6,
    ewald_alpha_prefactor: float = 5.0,
) -> torch.Tensor:
    """Batched periodic PME dipole-tensor field for multi-graph long-range prototypes."""
    if dipoles.dim() != 3 or dipoles.size(-1) != 3:
        raise ValueError("periodic_dipole_pme_field_batched expects dipoles with shape (N, C, 3)")
    k_cart, k2, spectral = build_periodic_dipole_pme_kernel_batched(
        cell=cell,
        mesh_size=mesh_size,
        assignment=assignment,
        device=dipoles.device,
        dtype=dipoles.dtype,
        k_norm_floor=k_norm_floor,
        assignment_window_floor=assignment_window_floor,
        ewald_alpha_prefactor=ewald_alpha_prefactor,
    )
    return apply_periodic_dipole_pme_field_batched(
        frac,
        batch,
        dipoles,
        mesh_size=mesh_size,
        assignment=assignment,
        assignment_offsets=assignment_offsets,
        k_cart=k_cart,
        k2=k2,
        spectral=spectral,
        k_norm_floor=k_norm_floor,
    )


class FeatureSpectralFilterGrid(nn.Module):
    """Apply a learnable radial spectral filter on a regular periodic/slab mesh."""

    def __init__(
        self,
        *,
        mesh_size: int,
        channels: int,
        hidden_dim: int,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        k_norm_floor: float = 1.0e-6,
    ):
        super().__init__()
        self.mesh_size = int(mesh_size)
        self.channels = int(channels)
        self.boundary = str(boundary)
        if self.boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported feature spectral boundary: {self.boundary!r}")
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.k_norm_floor = float(k_norm_floor)
        self.radial_filter = RadialSpectralFilter(hidden_dim=hidden_dim, k_norm_floor=k_norm_floor)
        self.channel_scale_raw = nn.Parameter(torch.zeros(self.channels))

    def _effective_cell(self, cell: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return _effective_cell_for_boundary(
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            dtype=dtype,
        )

    def build_k_norms(self, cell: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        freq = _fft_integer_frequencies(self.mesh_size, device=cell.device, dtype=dtype)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        effective_cell = self._effective_cell(cell, dtype=dtype)
        inv_cell = _inverse_3x3(effective_cell)
        # k = 2*pi * m @ inv(cell)^T (transpose required for O(3) equivariance on non-orthogonal
        # cells; without it |k| -- hence the radial filter -- is not rotation-invariant). No-op for
        # orthogonal cells. Matches the long-range build_k_norms / _build_k_cart_flat convention.
        k_cart = 2.0 * math.pi * torch.matmul(integer_k, inv_cell.transpose(-1, -2))
        if cell.dim() == 2:
            return _safe_vector_norm(k_cart, dim=-1, floor=self.k_norm_floor).reshape(
                self.mesh_size, self.mesh_size, self.mesh_size
            )
        return _safe_vector_norm(k_cart, dim=-1, floor=self.k_norm_floor).reshape(
            cell.size(0), self.mesh_size, self.mesh_size, self.mesh_size
        )

    def forward(self, mesh: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        mesh_dtype = mesh.dtype
        fft_dims = (0, 1, 2) if mesh.dim() == 4 else (1, 2, 3)
        mesh_complex = torch.fft.fftn(mesh, dim=fft_dims)
        k_norms = self.build_k_norms(cell, dtype=mesh_dtype)
        spectral_weights = self.radial_filter(k_norms)
        if not self.include_k0:
            spectral_weights = torch.where(
                k_norms > self.k_norm_floor,
                spectral_weights,
                torch.zeros_like(spectral_weights),
            )
        channel_scale = torch.nn.functional.softplus(self.channel_scale_raw).to(dtype=mesh_dtype)
        channel_view = channel_scale.view(*([1] * (mesh.dim() - 1)), -1)
        filtered = torch.fft.ifftn(mesh_complex * spectral_weights.unsqueeze(-1) * channel_view, dim=fft_dims)
        return filtered.real


class FeatureSpectralResidualBlock(nn.Module):
    """Low-rank feature-space spectral filter with FFT mesh projection."""

    def __init__(
        self,
        *,
        feature_dim: int,
        bottleneck_dim: int = 8,
        mesh_size: int = 16,
        filter_hidden_dim: int = 64,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        neutralize: bool = True,
        include_k0: bool = False,
        assignment: str = "pcs",
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.mesh_size = int(mesh_size)
        self.boundary = str(boundary)
        if self.boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported feature spectral boundary: {self.boundary!r}")
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.neutralize = bool(neutralize)
        self.include_k0 = bool(include_k0)
        self.assignment = str(assignment)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.in_proj = nn.Linear(self.feature_dim, self.bottleneck_dim)
        self.out_proj = nn.Linear(self.bottleneck_dim, self.feature_dim, bias=False)
        self.mesh_filter = FeatureSpectralFilterGrid(
            mesh_size=self.mesh_size,
            channels=self.bottleneck_dim,
            hidden_dim=int(filter_hidden_dim),
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            include_k0=self.include_k0,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        assignment_offsets = _build_assignment_offsets(self.assignment)
        self.register_buffer("assignment_offsets", assignment_offsets, persistent=False)

    def _neutralize_source_batched(self, source: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        if not self.neutralize:
            return source
        graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=source.dtype)
        counts = graph_mask.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        graph_mean = torch.einsum("nb,nc->bc", graph_mask, source) / counts
        return source - graph_mean.index_select(0, batch)

    def _filter_batched(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        frac = _prepare_frac_for_boundary_batched(
            pos,
            batch,
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
        )
        mesh = _spread_source_to_mesh_batched(
            frac,
            batch,
            source,
            num_graphs=cell.size(0),
            mesh_size=self.mesh_size,
            assignment=self.assignment,
            assignment_offsets=self.assignment_offsets,
            boundary=self.boundary,
        )
        filtered_mesh = self.mesh_filter(mesh, cell)
        return _gather_source_from_mesh_batched(
            frac,
            batch,
            filtered_mesh,
            mesh_size=self.mesh_size,
            assignment=self.assignment,
            assignment_offsets=self.assignment_offsets,
            boundary=self.boundary,
        )

    def forward(
        self,
        invariant_features: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.in_proj(self.input_norm(invariant_features))
        filtered_source = self._filter_batched(pos, batch, cell, self._neutralize_source_batched(source, batch, cell))

        residual = self.out_proj(filtered_source)
        gated_residual = torch.tanh(self.gate).to(dtype=residual.dtype) * residual
        return invariant_features + gated_residual, source


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
        mesh_size: int = 16,
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
    ):
        super().__init__()
        if boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported reciprocal mesh boundary: {boundary!r}")
        if energy_partition not in {"potential", "uniform"}:
            raise ValueError(f"Unsupported reciprocal energy partition: {energy_partition!r}")
        if assignment not in {"cic", "tsc", "pcs"}:
            raise ValueError(f"Unsupported mesh assignment: {assignment!r}")
        self.mesh_size = int(mesh_size)
        self.boundary = str(boundary)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.energy_partition = str(energy_partition)
        self.green_mode = str(green_mode)
        self.assignment = str(assignment)
        self.full_ewald = bool(full_ewald)
        # Latent-Ewald / LES-style: keep the (screened) reciprocal but SKIP the real-space erfc +
        # self + background -- the network absorbs the real-space part. Much cheaper (no O(N^2) erfc
        # loop), and a learnable long-range feature rather than an exact Coulomb sum.
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
    def num_k(self) -> int:
        total = self.mesh_size * self.mesh_size * self.mesh_size
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
        return real_cutoff.new_tensor(self.ewald_alpha_prefactor) / real_cutoff.clamp_min(self.k_norm_floor)

    def _init_real_space_shifts(self) -> torch.Tensor:
        ranges: list[torch.Tensor] = []
        for is_periodic in self._periodic_axes():
            values = [-1, 0, 1] if is_periodic else [0]
            ranges.append(torch.tensor(values, dtype=torch.long))
        return torch.cartesian_prod(*ranges)

    def _build_assignment_window(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = self._assignment_window_cache
        expected_numel = self.mesh_size * self.mesh_size * self.mesh_size
        if cached.numel() == expected_numel and cached.device == device and cached.dtype == dtype:
            return cached
        window_1d = _assignment_window_1d(
            mesh_size=self.mesh_size,
            assignment=self.assignment,
            device=device,
            dtype=dtype,
        )
        wx, wy, wz = torch.meshgrid(window_1d, window_1d, window_1d, indexing="ij")
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
        freq = _fft_integer_frequencies(self.mesh_size, device=cell.device, dtype=dtype)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        inv_cell = _inverse_3x3(effective_cell)
        k_cart = 2.0 * math.pi * torch.einsum("kd,bdh->bkh", integer_k, inv_cell.transpose(-1, -2))
        k_norms = _safe_vector_norm(k_cart, dim=-1, floor=self.k_norm_floor).reshape(
            cell.size(0), self.mesh_size, self.mesh_size, self.mesh_size
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
        freq = _fft_integer_frequencies(self.mesh_size, device=cell.device, dtype=dtype)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        inv_cell = _inverse_3x3(effective_cell)
        k_cart = 2.0 * math.pi * torch.einsum("kd,bdh->bkh", integer_k, inv_cell.transpose(-1, -2))
        k_norms = _safe_vector_norm(k_cart, dim=-1, floor=self.k_norm_floor)
        volume = torch.abs(_det_3x3(effective_cell)).clamp_min(self.k_norm_floor)
        return k_cart, k_norms, volume

    def _graph_counts(self, batch: torch.Tensor, cell: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=dtype)
        return graph_mask.sum(dim=0).clamp_min(1.0)

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
            mesh = _spread_source_to_mesh_batched(
                frac,
                batch,
                field,
                num_graphs=cell.size(0),
                mesh_size=self.mesh_size,
                assignment=self.assignment,
                assignment_offsets=self.assignment_offsets,
                boundary=self.boundary,
            )
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
        # Compensate torch.fft.ifftn's 1/N normalization (N = mesh_size**3). The forward FFT of
        # the spread charges already yields the structure factor S(k), so the iFFT's 1/N factor
        # is spurious here.
        filtered = filtered * (float(self.mesh_size) ** 3)
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
        mesh = _spread_source_to_mesh_batched(
            frac,
            batch,
            source,
            num_graphs=cell.size(0),
            mesh_size=self.mesh_size,
            assignment=self.assignment,
            assignment_offsets=self.assignment_offsets,
            boundary=self.boundary,
        )
        potential_mesh, alpha, effective_volume, real_cutoff = self.apply_green_kernel_batched(mesh, cell)
        reciprocal_potential = _gather_source_from_mesh_batched(
            frac,
            batch,
            potential_mesh,
            mesh_size=self.mesh_size,
            assignment=self.assignment,
            assignment_offsets=self.assignment_offsets,
            boundary=self.boundary,
        )
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


class LatentReciprocalLongRange(nn.Module):
    """Periodic reciprocal-space prototype closer to a learnable Green's function."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        *,
        source_channels: int = 1,
        boundary: str = "periodic",
        neutralize: bool = True,
        kmax: int = 2,
        mesh_size: int = 16,
        filter_hidden_dim: int = 64,
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        reciprocal_backend: str = "direct_kspace",
        energy_partition: str = "potential",
        green_mode: str = "poisson",
        assignment: str = "pcs",
        mesh_fft_full_ewald: bool = False,
        mesh_fft_reciprocal_only: bool = False,
    ):
        super().__init__()
        if reciprocal_backend not in {"direct_kspace", "mesh_fft"}:
            raise ValueError(f"Unsupported reciprocal backend: {reciprocal_backend!r}")
        if reciprocal_backend == "direct_kspace" and boundary != "periodic":
            raise ValueError("direct_kspace reciprocal backend currently requires boundary='periodic'")
        if boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported long-range boundary mode: {boundary!r}")
        self.source_channels = int(source_channels)
        self.boundary = str(boundary)
        self.neutralize = bool(neutralize)
        self.kmax = int(kmax)
        self.mesh_size = int(mesh_size)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.reciprocal_backend = str(reciprocal_backend)
        self.energy_partition = str(energy_partition)
        self.green_mode = str(green_mode)
        self.assignment = str(assignment)
        self.mesh_fft_full_ewald = bool(mesh_fft_full_ewald)
        self.mesh_fft_reciprocal_only = bool(mesh_fft_reciprocal_only)
        self.source_kind = "latent_charge"
        self.source_layout = "channels_last"
        self.runtime_backend = "mesh_fft" if reciprocal_backend == "mesh_fft" else "none"
        self.source_head = LatentSourceHead(feature_dim, hidden_dim, source_channels=self.source_channels)
        if self.reciprocal_backend == "mesh_fft":
            self.kernel = MeshLongRangeKernel3D(
                mesh_size=self.mesh_size,
                filter_hidden_dim=int(filter_hidden_dim),
                boundary=self.boundary,
                slab_padding_factor=self.slab_padding_factor,
                include_k0=self.include_k0,
                energy_partition=self.energy_partition,
                green_mode=self.green_mode,
                assignment=self.assignment,
                full_ewald=self.mesh_fft_full_ewald,
                reciprocal_only=self.mesh_fft_reciprocal_only,
            )
            self.exports_reciprocal_source = True
            self.energy_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.kernel = ReciprocalSpectralKernel3D(
                kmax=self.kmax,
                filter_hidden_dim=int(filter_hidden_dim),
                include_k0=self.include_k0,
                reciprocal_backend=self.reciprocal_backend,
                energy_partition=self.energy_partition,
            )
            self.exports_reciprocal_source = False
            # Keep the default contribution near zero so the module can be enabled
            # in existing workflows without destabilizing outputs before training.
            self.energy_scale = nn.Parameter(torch.tensor(0.0))

    @property
    def num_k(self) -> int:
        return self.kernel.num_k

    def _neutralize_source(self, source: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        if not self.neutralize:
            return source
        graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=source.dtype)
        counts = graph_mask.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        graph_mean = torch.einsum("nb,nc->bc", graph_mask, source) / counts
        return source - graph_mean.index_select(0, batch)

    def forward(
        self,
        invariant_features: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        *,
        edge_src: torch.Tensor | None = None,
        edge_dst: torch.Tensor | None = None,
        return_source: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        source = self.source_head(invariant_features)
        source = self._neutralize_source(source, batch, cell)
        atom_energy = self.kernel(pos, batch, cell, source)
        if self.energy_scale is not None:
            atom_energy = self.energy_scale * atom_energy
        if return_source:
            return atom_energy, source
        return atom_energy

    def emit_source(self, invariant_features: torch.Tensor) -> torch.Tensor:
        """Latent monopole (0e charge) source for the C++ reciprocal solver at deploy time: the raw
        source_head output only -- the reciprocal energy AND neutralization are deferred to the C++
        solver (which neutralizes per long_range_neutralize). Unlike ``forward(return_source=True)``
        this does NOT call the mesh kernel, so it avoids the kernel's per-graph ``torch.nonzero`` (a
        data-dependent op that make_fx/AOTI cannot trace), mirroring the multipole pack-source emit."""
        return self.source_head(invariant_features)

    def forward_multipole(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        monopole: torch.Tensor,
        dipole: torch.Tensor | None = None,
        quadrupole: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Long-range energy from pre-computed equivariant Cartesian multipole sources.

        Routes through the mesh-FFT kernel's ``multipole_energy``
        (S(k) = q + i k.mu - 1/2 k.Q.k). Only the monopole is charge-neutralized
        per graph; dipole/quadrupole carry no net-charge constraint.
        """
        if not hasattr(self.kernel, "multipole_energy"):
            raise ValueError(
                "multipole long-range requires reciprocal_backend='mesh_fft' "
                "(only the mesh-FFT kernel exposes multipole_energy)"
            )
        source = self._neutralize_source(monopole, batch, cell)
        atom_energy = self.kernel.multipole_energy(pos, batch, cell, source, dipole, quadrupole)
        if self.energy_scale is not None:
            atom_energy = self.energy_scale * atom_energy
        return atom_energy


def build_feature_spectral_module(
    *,
    mode: str,
    feature_dim: int,
    bottleneck_dim: int = 8,
    mesh_size: int = 16,
    filter_hidden_dim: int = 64,
    boundary: str = "periodic",
    slab_padding_factor: int = 2,
    neutralize: bool = True,
    include_k0: bool = False,
    assignment: str = "pcs",
    gate_init: float = 0.0,
) -> nn.Module | None:
    if mode == "none":
        return None
    if mode == "fft":
        return FeatureSpectralResidualBlock(
            feature_dim=feature_dim,
            bottleneck_dim=bottleneck_dim,
            mesh_size=mesh_size,
            filter_hidden_dim=filter_hidden_dim,
            boundary=boundary,
            slab_padding_factor=slab_padding_factor,
            neutralize=neutralize,
            include_k0=include_k0,
            assignment=assignment,
            gate_init=gate_init,
        )
    raise ValueError(f"Unsupported feature_spectral_mode: {mode!r}")


def build_long_range_module(
    *,
    mode: str,
    feature_dim: int,
    hidden_dim: int = 64,
    boundary: str = "nonperiodic",
    neutralize: bool = True,
    filter_hidden_dim: int = 64,
    kmax: int = 2,
    mesh_size: int = 16,
    slab_padding_factor: int = 2,
    include_k0: bool = False,
    source_channels: int = 1,
    backend: str = "dense_pairwise",
    reciprocal_backend: str = "direct_kspace",
    energy_partition: str = "potential",
    green_mode: str = "poisson",
    assignment: str = "pcs",
    mesh_fft_full_ewald: bool = False,
    mesh_fft_reciprocal_only: bool = False,
    theta: float = 0.5,
    leaf_size: int = 32,
    multipole_order: int = 0,
    far_source_dim: int = 16,
    far_num_shells: int = 3,
    far_shell_growth: float = 2.0,
    far_tail: bool = True,
    far_tail_bins: int = 2,
    far_stats: str = "mean,count,mean_r,rms_r",
    far_max_radius_multiplier: float | None = None,
    far_source_norm: bool = True,
    far_gate_init: float = 0.0,
    cutoff_radius: float = 5.0,
    max_multipole_l: int = 0,
    multipole_feature_channels: int = 0,
) -> nn.Module | None:
    if mode == "none":
        return None
    if mode == "reciprocal-spectral-v1":
        return LatentReciprocalLongRange(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            source_channels=source_channels,
            boundary=boundary,
            neutralize=neutralize,
            kmax=kmax,
            mesh_size=mesh_size,
            filter_hidden_dim=filter_hidden_dim,
            slab_padding_factor=slab_padding_factor,
            include_k0=include_k0,
            reciprocal_backend=reciprocal_backend,
            energy_partition=energy_partition,
            green_mode=green_mode,
            assignment=assignment,
            mesh_fft_full_ewald=mesh_fft_full_ewald,
            mesh_fft_reciprocal_only=mesh_fft_reciprocal_only,
        )
    raise ValueError(f"Unsupported long_range_mode: {mode!r}")


def configure_long_range_modules(
    owner: nn.Module,
    *,
    feature_dim: int,
    cutoff_radius: float,
    long_range_mode: str = "none",
    long_range_hidden_dim: int = 64,
    long_range_boundary: str = "nonperiodic",
    long_range_neutralize: bool = True,
    long_range_filter_hidden_dim: int = 64,
    long_range_kmax: int = 2,
    long_range_mesh_size: int = 16,
    long_range_slab_padding_factor: int = 2,
    long_range_include_k0: bool = False,
    long_range_source_channels: int = 1,
    long_range_backend: str = "dense_pairwise",
    long_range_reciprocal_backend: str = "direct_kspace",
    long_range_energy_partition: str = "potential",
    long_range_green_mode: str = "poisson",
    long_range_assignment: str = "pcs",
    long_range_mesh_fft_full_ewald: bool = False,
    long_range_mesh_fft_reciprocal_only: bool = False,
    long_range_theta: float = 0.5,
    long_range_leaf_size: int = 32,
    long_range_multipole_order: int = 0,
    long_range_far_source_dim: int = 16,
    long_range_far_num_shells: int = 3,
    long_range_far_shell_growth: float = 2.0,
    long_range_far_tail: bool = True,
    long_range_far_tail_bins: int = 2,
    long_range_far_stats: str = "mean,count,mean_r,rms_r",
    long_range_far_max_radius_multiplier: float | None = None,
    long_range_far_source_norm: bool = True,
    long_range_far_gate_init: float = 0.0,
    feature_spectral_mode: str = "none",
    feature_spectral_bottleneck_dim: int = 8,
    feature_spectral_mesh_size: int = 16,
    feature_spectral_filter_hidden_dim: int = 64,
    feature_spectral_boundary: str = "periodic",
    feature_spectral_slab_padding_factor: int = 2,
    feature_spectral_neutralize: bool = True,
    feature_spectral_include_k0: bool = False,
    feature_spectral_assignment: str = "pcs",
    feature_spectral_gate_init: float = 0.0,
) -> None:
    owner.long_range_mode = str(long_range_mode)
    owner.long_range_hidden_dim = int(long_range_hidden_dim)
    owner.long_range_boundary = str(long_range_boundary)
    owner.long_range_neutralize = bool(long_range_neutralize)
    owner.long_range_filter_hidden_dim = int(long_range_filter_hidden_dim)
    owner.long_range_kmax = int(long_range_kmax)
    owner.long_range_mesh_size = int(long_range_mesh_size)
    owner.long_range_slab_padding_factor = int(long_range_slab_padding_factor)
    owner.long_range_include_k0 = bool(long_range_include_k0)
    owner.long_range_source_channels = int(long_range_source_channels)
    owner.long_range_backend = str(long_range_backend)
    owner.long_range_reciprocal_backend = str(long_range_reciprocal_backend)
    owner.long_range_energy_partition = str(long_range_energy_partition)
    owner.long_range_green_mode = str(long_range_green_mode)
    owner.long_range_assignment = str(long_range_assignment)
    owner.long_range_mesh_fft_full_ewald = bool(long_range_mesh_fft_full_ewald)
    owner.long_range_mesh_fft_reciprocal_only = bool(long_range_mesh_fft_reciprocal_only)
    owner.long_range_theta = float(long_range_theta)
    owner.long_range_leaf_size = int(long_range_leaf_size)
    owner.long_range_multipole_order = int(long_range_multipole_order)
    owner.long_range_far_source_dim = int(long_range_far_source_dim)
    owner.long_range_far_num_shells = int(long_range_far_num_shells)
    owner.long_range_far_shell_growth = float(long_range_far_shell_growth)
    owner.long_range_far_tail = bool(long_range_far_tail)
    owner.long_range_far_tail_bins = int(long_range_far_tail_bins)
    owner.long_range_far_stats = str(long_range_far_stats)
    owner.long_range_far_max_radius_multiplier = (
        None if long_range_far_max_radius_multiplier is None else float(long_range_far_max_radius_multiplier)
    )
    owner.long_range_far_source_norm = bool(long_range_far_source_norm)
    owner.long_range_far_gate_init = float(long_range_far_gate_init)
    owner.feature_spectral_mode = str(feature_spectral_mode)
    owner.feature_spectral_bottleneck_dim = int(feature_spectral_bottleneck_dim)
    owner.feature_spectral_mesh_size = int(feature_spectral_mesh_size)
    owner.feature_spectral_filter_hidden_dim = int(feature_spectral_filter_hidden_dim)
    owner.feature_spectral_boundary = str(feature_spectral_boundary)
    owner.feature_spectral_slab_padding_factor = int(feature_spectral_slab_padding_factor)
    owner.feature_spectral_neutralize = bool(feature_spectral_neutralize)
    owner.feature_spectral_include_k0 = bool(feature_spectral_include_k0)
    owner.feature_spectral_assignment = str(feature_spectral_assignment)
    owner.feature_spectral_gate_init = float(feature_spectral_gate_init)

    owner.long_range_module = build_long_range_module(
        mode=owner.long_range_mode,
        feature_dim=int(feature_dim),
        hidden_dim=owner.long_range_hidden_dim,
        boundary=owner.long_range_boundary,
        neutralize=owner.long_range_neutralize,
        filter_hidden_dim=owner.long_range_filter_hidden_dim,
        kmax=owner.long_range_kmax,
        mesh_size=owner.long_range_mesh_size,
        slab_padding_factor=owner.long_range_slab_padding_factor,
        include_k0=owner.long_range_include_k0,
        source_channels=owner.long_range_source_channels,
        backend=owner.long_range_backend,
        reciprocal_backend=owner.long_range_reciprocal_backend,
        energy_partition=owner.long_range_energy_partition,
        green_mode=owner.long_range_green_mode,
        assignment=owner.long_range_assignment,
        mesh_fft_full_ewald=owner.long_range_mesh_fft_full_ewald,
        mesh_fft_reciprocal_only=owner.long_range_mesh_fft_reciprocal_only,
        theta=owner.long_range_theta,
        leaf_size=owner.long_range_leaf_size,
        multipole_order=owner.long_range_multipole_order,
        far_source_dim=owner.long_range_far_source_dim,
        far_num_shells=owner.long_range_far_num_shells,
        far_shell_growth=owner.long_range_far_shell_growth,
        far_tail=owner.long_range_far_tail,
        far_tail_bins=owner.long_range_far_tail_bins,
        far_stats=owner.long_range_far_stats,
        far_max_radius_multiplier=owner.long_range_far_max_radius_multiplier,
        far_source_norm=owner.long_range_far_source_norm,
        far_gate_init=owner.long_range_far_gate_init,
        cutoff_radius=float(cutoff_radius),
    )
    owner.long_range_num_k = (
        getattr(owner.long_range_module, "num_k", None) if owner.long_range_module is not None else None
    )
    owner.feature_spectral_module = build_feature_spectral_module(
        mode=owner.feature_spectral_mode,
        feature_dim=int(feature_dim),
        bottleneck_dim=owner.feature_spectral_bottleneck_dim,
        mesh_size=owner.feature_spectral_mesh_size,
        filter_hidden_dim=owner.feature_spectral_filter_hidden_dim,
        boundary=owner.feature_spectral_boundary,
        slab_padding_factor=owner.feature_spectral_slab_padding_factor,
        neutralize=owner.feature_spectral_neutralize,
        include_k0=owner.feature_spectral_include_k0,
        assignment=owner.feature_spectral_assignment,
        gate_init=owner.feature_spectral_gate_init,
    )
    owner.long_range_runtime_backend = "none"
    owner.long_range_runtime_source_kind = "none"
    owner.long_range_runtime_source_channels = 0
    owner.long_range_runtime_source_layout = "none"
    owner.long_range_runtime_source_boundary = owner.long_range_boundary
    owner.long_range_runtime_source_slab_padding_factor = owner.long_range_slab_padding_factor
    if owner.long_range_module is not None and bool(getattr(owner.long_range_module, "exports_reciprocal_source", False)):
        owner.long_range_runtime_backend = str(getattr(owner.long_range_module, "runtime_backend", "none"))
        owner.long_range_runtime_source_kind = str(getattr(owner.long_range_module, "source_kind", "latent_charge"))
        owner.long_range_runtime_source_channels = int(
            getattr(owner.long_range_module, "source_channels", owner.long_range_source_channels)
        )
        owner.long_range_runtime_source_layout = str(getattr(owner.long_range_module, "source_layout", "channels_last"))
        owner.long_range_runtime_source_boundary = owner.long_range_boundary
        owner.long_range_runtime_source_slab_padding_factor = owner.long_range_slab_padding_factor
        owner.reciprocal_source_channels = owner.long_range_runtime_source_channels
        owner.reciprocal_source_boundary = owner.long_range_runtime_source_boundary
        owner.reciprocal_source_slab_padding_factor = owner.long_range_runtime_source_slab_padding_factor
    elif owner.feature_spectral_module is not None:
        owner.long_range_runtime_backend = "mesh_fft"
        owner.long_range_runtime_source_kind = "feature_bottleneck"
        owner.long_range_runtime_source_channels = owner.feature_spectral_bottleneck_dim
        owner.long_range_runtime_source_layout = "channels_last"
        owner.long_range_runtime_source_boundary = owner.feature_spectral_boundary
        owner.long_range_runtime_source_slab_padding_factor = owner.feature_spectral_slab_padding_factor
        owner.reciprocal_source_channels = owner.feature_spectral_bottleneck_dim
        owner.reciprocal_source_boundary = owner.feature_spectral_boundary
        owner.reciprocal_source_slab_padding_factor = owner.feature_spectral_slab_padding_factor
    else:
        owner.reciprocal_source_channels = 0
        owner.reciprocal_source_boundary = "periodic"
        owner.reciprocal_source_slab_padding_factor = 2


def apply_long_range_modules(
    owner: nn.Module,
    invariant_features: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    *,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    return_reciprocal_source: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    feature_reciprocal_source = None
    if getattr(owner, "feature_spectral_module", None) is not None:
        invariant_features, feature_reciprocal_source = owner.feature_spectral_module(
            invariant_features,
            pos,
            batch,
            cell,
        )

    long_range_energy = None
    reciprocal_source = None
    defer_long_range_to_runtime = False
    if getattr(owner, "long_range_module", None) is not None:
        if return_reciprocal_source and bool(getattr(owner.long_range_module, "exports_reciprocal_source", False)):
            long_range_energy, reciprocal_source = owner.long_range_module(
                invariant_features,
                pos,
                batch,
                cell,
                edge_src=edge_src,
                edge_dst=edge_dst,
                return_source=True,
            )
            defer_long_range_to_runtime = reciprocal_source.numel() > 0
        else:
            long_range_energy = owner.long_range_module(
                invariant_features,
                pos,
                batch,
                cell,
                edge_src=edge_src,
                edge_dst=edge_dst,
            )
    if reciprocal_source is None:
        reciprocal_source = feature_reciprocal_source
    return invariant_features, long_range_energy, reciprocal_source, defer_long_range_to_runtime
