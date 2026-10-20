"""Shared PME / cuFFT reciprocal-space operator backend.

This factors the grid operators already used by :class:`MeshLongRangeKernel3D` into ONE reusable
interface -- GridSpec / B-spline assignment / FFT plan / reciprocal kernel / gather -- so that the
two long-range physics share the *backend* (not the physics):

    ReciprocalBackend (grid + assignment + FFT + gather)
      |- electrostatics : scalar charge source  + scalar kernel  4*pi/k^2          -> potential
      |- MBD            : vector dipole source   + tensor kernel  4*pi k_a k_b/k^2  -> field  (in SLQ matvec)

The scalar path here is REGRESSION-IDENTICAL to the existing reciprocal potential (same spread, same
FFT, same mesh^3 normalization, same gather); only the *kernel* and the *source channels* differ
between electrostatics and MBD. The MBD tensor kernel + spectral (SLQ) solver are layered on top
(see ``dipole_field`` / the MBD module), not merged into a scalar Poisson solve.
"""

from __future__ import annotations

import torch
from torch import nn

from mace_ictc.models.long_range import (
    _build_assignment_offsets,
    _effective_cell_for_boundary,
    _fft_integer_frequencies,
    _gather_source_from_mesh,
    _inverse_3x3,
    _prepare_frac_for_boundary,
    _safe_vector_norm,
    _spread_source_to_mesh,
)


class ReciprocalBackend(nn.Module):
    """GridSpec + B-spline assignment + FFT + reciprocal-kernel + gather, channel-agnostic.

    All ops accept an arbitrary source channel count C, so electrostatics (C=1 charge) and MBD
    (C=3 dipole, or 3*n_probes batched SLQ vectors) use the identical grid machinery.
    """

    def __init__(
        self,
        *,
        mesh_size: int = 16,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        assignment: str = "pcs",
        include_k0: bool = False,
        k_norm_floor: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported boundary: {boundary!r}")
        if assignment not in {"cic", "tsc", "pcs"}:
            raise ValueError(f"Unsupported assignment: {assignment!r}")
        self.mesh_size = int(mesh_size)
        self.boundary = str(boundary)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.assignment = str(assignment)
        self.include_k0 = bool(include_k0)
        self.k_norm_floor = float(k_norm_floor)
        self.register_buffer("assignment_offsets", _build_assignment_offsets(self.assignment), persistent=False)

    # ---- GridSpec ----
    def effective_cell(self, cell: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return _effective_cell_for_boundary(
            cell, boundary=self.boundary, slab_padding_factor=self.slab_padding_factor, dtype=dtype
        )

    def frac(self, pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        return _prepare_frac_for_boundary(
            pos, cell, boundary=self.boundary, slab_padding_factor=self.slab_padding_factor
        )

    def k_grid(self, cell: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reciprocal lattice vectors for a single cell [3,3].

        Returns ``(k_cart [K,3], k_norm [K], volume scalar)`` with k = 2*pi * m @ inv(cell)^T over the
        FFT integer-frequency grid m (K = mesh^3). The transpose makes k.mu / k.Q.k equivariant and
        matches the C++ mff_reciprocal_solver convention.
        """
        eff = self.effective_cell(cell, dtype=dtype)
        m = self.mesh_size
        freq = _fft_integer_frequencies(m, device=eff.device, dtype=dtype)  # [M]
        grids = torch.meshgrid(freq, freq, freq, indexing="ij")
        integer_k = torch.stack(grids, dim=-1).reshape(-1, 3)  # [K,3]
        inv_cell = _inverse_3x3(eff)
        k_cart = 2.0 * torch.pi * torch.matmul(integer_k, inv_cell.transpose(-1, -2))
        k_norm = _safe_vector_norm(k_cart, dim=-1, floor=self.k_norm_floor)
        volume = torch.det(eff).abs()
        return k_cart, k_norm, volume

    def assignment_window(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """1/|W(k)|^2 deconvolution weights, flat [K]. W = prod_axes sinc(m/mesh)^stencil."""
        from mace_ictc.models.long_range import _assignment_window_1d

        w1d = _assignment_window_1d(mesh_size=self.mesh_size, assignment=self.assignment, device=device, dtype=dtype)
        m = self.mesh_size
        window = (w1d.view(m, 1, 1) * w1d.view(1, m, 1) * w1d.view(1, 1, m)).reshape(-1)
        return torch.reciprocal(window.clamp_min(1.0e-6).square())

    # ---- channel-agnostic grid ops ----
    def spread(self, frac: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Atoms -> mesh [M,M,M,C] (B-spline assignment); C = source.size(1)."""
        return _spread_source_to_mesh(
            frac, source, mesh_size=self.mesh_size, assignment=self.assignment,
            assignment_offsets=self.assignment_offsets, boundary=self.boundary,
        )

    def gather(self, frac: torch.Tensor, mesh: torch.Tensor) -> torch.Tensor:
        """Mesh [M,M,M,C] -> atoms [N,C] (adjoint of spread)."""
        return _gather_source_from_mesh(
            frac, mesh, mesh_size=self.mesh_size, assignment=self.assignment,
            assignment_offsets=self.assignment_offsets, boundary=self.boundary,
        )

    @staticmethod
    def fftn(mesh: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(mesh, dim=(0, 1, 2))

    @staticmethod
    def ifftn(mesh_complex: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifftn(mesh_complex, dim=(0, 1, 2))

    # ---- scalar reciprocal route (electrostatics) ----
    def scalar_potential(self, frac: torch.Tensor, source: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        """spread -> FFT -> x scalar spectral weight -> iFFT -> (mesh^3 norm) -> gather.

        ``spectral`` is the per-k scalar green/window weight [K] or [M,M,M] (caller supplies the
        physics: 4*pi/k^2 / V * 1/|W|^2 [* Ewald screening]). Regression-identical to the existing
        MeshLongRangeKernel3D.apply_green_kernel + gather. Returns per-atom potential [N,C].
        """
        m = self.mesh_size
        mesh = self.spread(frac, source)
        mesh_c = self.fftn(mesh)
        sw = spectral.reshape(m, m, m, 1).to(mesh_c.dtype)
        filtered = self.ifftn(mesh_c * sw).real * (float(m) ** 3)
        return self.gather(frac, filtered)
