from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ictc_o3.electrostatics.mesh import mesh_shape
from ictc_o3.electrostatics.mesh import normalize_mesh_size
from ictc_o3.electrostatics.cutoff import derive_long_range_far_max_radius_multiplier


from ictc_o3.electrostatics.functional import _build_integer_k_lattice


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


from ictc_o3.electrostatics.functional import _fft_integer_frequencies


from ictc_o3.electrostatics.functional import _safe_vector_norm


from ictc_o3.electrostatics.functional import _inverse_3x3


from ictc_o3.electrostatics.functional import _det_3x3


from ictc_o3.electrostatics.functional import _effective_cell_for_boundary


from ictc_o3.electrostatics.functional import _prepare_frac_for_boundary


from ictc_o3.electrostatics.functional import _prepare_frac_for_boundary_batched


from ictc_o3.electrostatics.functional import _assignment_stencil_size


from ictc_o3.electrostatics.functional import _build_assignment_offsets


from ictc_o3.electrostatics.functional import _assignment_kernel_1d


from ictc_o3.electrostatics.functional import _assignment_weights_from_scaled


from ictc_o3.electrostatics.functional import _assignment_window_1d


from ictc_o3.electrostatics.functional import _apply_mesh_boundary


from ictc_o3.electrostatics.functional import _spread_source_to_mesh


from ictc_o3.electrostatics.functional import _spread_source_to_mesh_batched


from ictc_o3.electrostatics.functional import _gather_source_from_mesh


from ictc_o3.electrostatics.functional import _gather_source_from_mesh_batched


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


from ictc_o3.electrostatics.functional import RadialSpectralFilter


from ictc_o3.electrostatics.functional import ReciprocalGreenKernel


from ictc_o3.electrostatics.functional import ReciprocalSpectralKernel3D


from ictc_o3.electrostatics.functional import MeshLongRangeKernel3D


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
        mesh_size: int | tuple[int, int, int] = 16,
        filter_hidden_dim: int = 64,
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        reciprocal_backend: str = "direct_kspace",
        energy_partition: str = "potential",
        green_mode: str = "poisson",
        assignment: str = "pcs",
        mesh_fft_full_ewald: bool = False,
        mesh_fft_reciprocal_only: bool = False,
        smearing_sigma: float | None = None,
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
        self.mesh_size = normalize_mesh_size(mesh_size)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.reciprocal_backend = str(reciprocal_backend)
        self.energy_partition = str(energy_partition)
        self.green_mode = str(green_mode)
        self.assignment = str(assignment)
        self.mesh_fft_full_ewald = bool(mesh_fft_full_ewald)
        self.mesh_fft_reciprocal_only = bool(mesh_fft_reciprocal_only)
        self.smearing_sigma = smearing_sigma
        if smearing_sigma is not None and reciprocal_backend != "mesh_fft":
            raise ValueError("Fixed smearing_sigma currently requires reciprocal_backend='mesh_fft'")
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
                smearing_sigma=smearing_sigma,
            )
            # The external reciprocal-only solver cannot reproduce a learned
            # Green filter or optional real/self/background Ewald corrections.
            # Fixed-sigma and three-axis-grid models keep their energy inline
            # until the external C++ solver supports the same functional/grid.
            self.exports_reciprocal_source = self.green_mode == "poisson" and smearing_sigma is None and isinstance(self.mesh_size, int) and (
                not self.mesh_fft_full_ewald or self.mesh_fft_reciprocal_only)
            self.runtime_backend = "mesh_fft" if self.exports_reciprocal_source else "none"
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
        if cell.shape[0] == 1:
            return source - source.sum(dim=0, keepdim=True) / max(source.shape[0], 1)
        counts = source.new_zeros(cell.size(0)).index_add(0, batch, source.new_ones(source.shape[0])).clamp_min(1).unsqueeze(-1)
        graph_mean = source.new_zeros((cell.size(0), source.shape[1])).index_add(0, batch, source) / counts
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
    mesh_size: int | tuple[int, int, int] = 16,
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
    smearing_sigma: float | None = None,
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
            smearing_sigma=smearing_sigma,
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
    long_range_mesh_size: int | tuple[int, int, int] = 16,
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
    long_range_smearing_sigma: float | None = None,
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
    owner.long_range_mesh_size = normalize_mesh_size(long_range_mesh_size)
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
    owner.long_range_smearing_sigma = long_range_smearing_sigma
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
        smearing_sigma=owner.long_range_smearing_sigma,
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
