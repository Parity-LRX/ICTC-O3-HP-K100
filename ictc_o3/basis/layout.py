"""Shared block layouts; no model construction or runtime configuration."""
from __future__ import annotations
import torch

def _split_irreps(x: torch.Tensor, channels: int, lmax: int) -> dict[int, torch.Tensor]:
    widths = [channels * (2 * l + 1) for l in range(lmax + 1)]
    # One split joins cotangents once, instead of creating and adding a
    # full-width zero-padded gradient for every independently sliced degree.
    # Preserve front-slicing when a caller requests only the lower degrees.
    blocks = x.split([*widths, x.shape[-1] - sum(widths)], dim=-1)
    return {l: blocks[l].view(*x.shape[:-1], channels, 2 * l + 1)
            for l in range(lmax + 1)}



def _merge_irreps(blocks: dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    parts = []
    for l in range(lmax + 1):
        parts.append(blocks[l].reshape(*blocks[l].shape[:-2], channels * (2 * l + 1)))
    return torch.cat(parts, dim=-1)

