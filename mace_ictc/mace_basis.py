"""Orthogonal ICTC <-> original-MACE (e3nn) basis change.

The baseline ICTC-MACE stores a degree-``l`` feature as the ``2l+1`` components of an *irreducible
Cartesian tensor*; original MACE / e3nn stores it as real spherical-harmonic components. The two
conventions are related, per degree, by a fixed **orthogonal** matrix ``Q_l`` such that

    direction_harmonics(n, l) @ Q_l == Y_l(n)              (machine precision)

so an arbitrary equivariant feature ``x`` (ICTC basis) becomes its original-MACE counterpart by a
single right-multiply ``x @ Q`` with the block-diagonal ``Q = diag(Q_0, ..., Q_lmax)``.

Important: energy, forces and the virial are SO(3) invariants / physical Cartesian tensors and are
**unchanged** by ``Q`` (it is an orthogonal change of the angular basis). ``Q`` only re-expresses
equivariant (``l>=1``) features. These helpers let you put any equivariant tensor into the
original-MACE convention consistently across Python, the exported model, and LAMMPS (see
``lammps_user_mfftorch/src/USER-MFFTORCH/mff_mace_basis.h`` for the matching C++ constants).
"""
from __future__ import annotations

import functools

import torch
from e3nn import o3

from mace_ictc.models.ictd_irreps import direction_harmonics_all

_Q_FIT_SEED = 20260426
_Q_FIT_SAMPLES = 8192


@functools.lru_cache(maxsize=None)
def _q_per_l_f64(lmax: int) -> tuple[torch.Tensor, ...]:
    """Per-degree orthogonal Q_l (float64), deterministic. ICTC harmonics @ Q_l == e3nn SH."""
    g = torch.Generator(device="cpu").manual_seed(_Q_FIT_SEED)
    dirs = torch.randn(_Q_FIT_SAMPLES, 3, generator=g, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y_ictd = direction_harmonics_all(dirs, int(lmax))
    y_e3nn = o3.spherical_harmonics(
        o3.Irreps.spherical_harmonics(int(lmax)), dirs, normalize=True, normalization="component"
    )
    out = []
    off = 0
    for l in range(int(lmax) + 1):
        w = 2 * l + 1
        a = y_ictd[l].to(torch.float64)
        b = y_e3nn[:, off : off + w].to(torch.float64)
        off += w
        # orthogonal Procrustes: argmin_Q ||a Q - b||, Q = U Vᵀ from SVD(aᵀb)
        u, _, vh = torch.linalg.svd(a.T @ b)
        out.append(u @ vh)
    return tuple(out)


def orthogonal_Q_blocks(lmax: int, *, dtype=torch.float64, device="cpu") -> list[torch.Tensor]:
    """List of per-degree orthogonal matrices ``[Q_0, ..., Q_lmax]`` (each ``(2l+1, 2l+1)``)."""
    return [q.to(dtype=dtype, device=device) for q in _q_per_l_f64(int(lmax))]


def orthogonal_Q(lmax: int, *, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Block-diagonal orthogonal ``Q`` (size ``sum_l (2l+1) = (lmax+1)**2``), ICTC -> MACE/e3nn."""
    return torch.block_diag(*orthogonal_Q_blocks(int(lmax), dtype=dtype, device=device))


def to_mace_basis(x: torch.Tensor, lmax: int) -> torch.Tensor:
    """ICTC -> original-MACE/e3nn. ``x`` has the ``(lmax+1)**2`` angular components in its last axis."""
    Q = orthogonal_Q(int(lmax), dtype=x.dtype, device=x.device)
    return x @ Q


def to_ictd_basis(x: torch.Tensor, lmax: int) -> torch.Tensor:
    """original-MACE/e3nn -> ICTC (inverse of :func:`to_mace_basis`; ``Q`` is orthogonal so Qᵀ)."""
    Q = orthogonal_Q(int(lmax), dtype=x.dtype, device=x.device)
    return x @ Q.transpose(-1, -2)


def to_mace_basis_blocks(blocks: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Per-degree dict ``{l: [..., 2l+1]}`` (ICTC) -> original-MACE/e3nn convention."""
    lmax = max(blocks)
    Qs = orthogonal_Q_blocks(lmax)
    return {l: t @ Qs[l].to(dtype=t.dtype, device=t.device) for l, t in blocks.items()}
