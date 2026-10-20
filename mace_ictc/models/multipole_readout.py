"""Equivariant multipole readout for long-range electrostatics.

A degree-l irreducible ICTC carrier *is* a rank-l multipole moment. Given the
full-SO(3) node features of the last interaction layer (l = 0..lmax, the
"fusion route" tap rather than the scalar-only readout), this module produces
per-atom Cartesian multipole sources -- monopole (l=0), dipole (l=1),
quadrupole (l=2) -- ready for the reciprocal-space ``multipole_energy`` PME path
in :mod:`mace_ictc.models.long_range`.

Equivariance: the channel mixers act on the channel axis only (shared across the
2l+1 angular components), so they commute with rotations; the ICTC-block ->
Cartesian decode is the existing convention-faithful
:class:`PhysicalTensorICTDRecovery`. Hence the monopole is invariant, the dipole
rotates by R, and the quadrupole by R Q R^T.
"""

from __future__ import annotations

import torch
from torch import nn

from mace_ictc.models.pure_cartesian_ictd_layers import (
    PhysicalTensorICTDRecovery,
    _split_irreps,
)


def pack_multipole_source(
    monopole: torch.Tensor,
    dipole: torch.Tensor | None = None,
    quadrupole: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack Cartesian multipoles into the LAMMPS reciprocal-solver source layout
    ``[ q (N, S) | dipole (N, 3S) | quadrupole (N, 9S) ]`` -> ``(N, S*(1 + 3[mu] + 9[Q]))``.

    Channel-major within each block (per channel: dipole x,y,z; quadrupole 3x3 row-major),
    matching mff_reciprocal_solver.cpp's ``narrow(1, off, k*C).reshape(-1, C, ...)`` decode
    (q at [0,C), mu at [C,4C) -> reshape(C,3), Q at [4C,13C) -> reshape(C,3,3)).
    """
    n, s = monopole.shape[0], monopole.shape[1]
    parts = [monopole.contiguous()]
    if dipole is not None:
        parts.append(dipole.contiguous().flatten(start_dim=1))
    if quadrupole is not None:
        parts.append(quadrupole.contiguous().flatten(start_dim=1))
    return torch.cat(parts, dim=1)


class MultipoleReadout(nn.Module):
    def __init__(
        self,
        channels: int,
        lmax: int,
        max_multipole_l: int,
        source_channels: int = 1,
        source_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.max_multipole_l = int(max_multipole_l)
        self.source_channels = int(source_channels)
        if self.max_multipole_l < 0:
            raise ValueError(f"max_multipole_l must be >= 0, got {self.max_multipole_l}")
        if self.max_multipole_l > self.lmax:
            raise ValueError(
                f"max_multipole_l={self.max_multipole_l} exceeds model lmax={self.lmax}; "
                "the last layer must expose at least that degree"
            )
        if self.max_multipole_l > 2:
            raise NotImplementedError(
                "multipole long-range currently supports max_multipole_l <= 2 "
                "(monopole/dipole/quadrupole)"
            )
        # Per-degree channel mixers: C -> source_channels, acting ONLY on the
        # channel axis (so equivariance is preserved). No bias (would break l>0).
        self.mix = nn.ModuleList(
            [nn.Linear(self.channels, self.source_channels, bias=False) for _ in range(self.max_multipole_l + 1)]
        )
        self.dipole_recovery = (
            PhysicalTensorICTDRecovery(
                rank=1, channels_in=self.source_channels, lmax_in=1, include_trace_chain=False
            )
            if self.max_multipole_l >= 1
            else None
        )
        self.quad_recovery = (
            PhysicalTensorICTDRecovery(
                rank=2,
                channels_in=self.source_channels,
                lmax_in=2,
                include_trace_chain=False,
                rank2_mode="symmetric",
            )
            if self.max_multipole_l >= 2
            else None
        )
        # Gentle-start gate. A learnable scalar that multiplies every emitted multipole
        # source (q, mu, Q). The reciprocal long-range energy is QUADRATIC in the sources
        # (E ~ |S(k)|^2), so a freshly-added multipole head started at the default mix
        # init injects a non-trivial energy from step 1 -- fine for from-scratch training
        # but destabilizing when WARM-STARTING a trained backbone (--finetune). Initializing
        # the scale small makes the contribution start ~`source_scale_init`^2 of the ungated
        # value (gentle), while it stays trainable because the sources themselves are
        # non-zero (d E / d source_scale != 0, so it bootstraps -- a zero-source start would
        # be a zero-gradient fixed point of the quadratic energy). Crucially the gate is
        # applied INSIDE forward, before BOTH the in-model `forward_multipole` energy AND the
        # exported `pack_multipole_source`, so train and deploy stay byte-consistent with no
        # change to the C++ reciprocal solver (it just receives the gated sources). This is
        # the multipole analog of the scalar route's `energy_scale=0` warm-start gate.
        self.source_scale = nn.Parameter(torch.tensor(float(source_scale_init)))

    def _chan_mix(self, block_l: torch.Tensor, l: int) -> torch.Tensor:
        # block_l: [N, C, 2l+1] -> [N, source_channels, 2l+1]
        return self.mix[l](block_l.transpose(-1, -2)).transpose(-1, -2)

    def forward(
        self, node_feats_so3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """node_feats_so3: ``[N, channels*(lmax+1)**2]`` (concat over l=0..lmax of
        ``[N, channels, 2l+1]``). Returns ``(monopole [N, S], dipole [N, S, 3] or
        None, quadrupole [N, S, 3, 3] or None)`` with ``S = source_channels``."""
        blocks = _split_irreps(node_feats_so3, self.channels, self.lmax)
        scale = self.source_scale.to(dtype=blocks[0].dtype)
        monopole = self._chan_mix(blocks[0], 0).squeeze(-1) * scale  # [N, S]
        dipole = None
        quadrupole = None
        if self.max_multipole_l >= 1:
            d_block = self._chan_mix(blocks[1], 1)  # [N, S, 3]
            dipole = self.dipole_recovery({1: d_block}, squeeze_channel=False) * scale  # [N, S, 3]
        if self.max_multipole_l >= 2:
            q_block = self._chan_mix(blocks[2], 2)  # [N, S, 5]
            quadrupole = self.quad_recovery({2: q_block}, squeeze_channel=False) * scale  # [N, S, 3, 3]
        return monopole, dipole, quadrupole
