"""Fixed representation bridges and tensor block conventions."""
from __future__ import annotations
from typing import Dict
import torch
from torch import nn
from e3nn import o3
from .layout import _split_irreps
from .ictd_irreps import direction_harmonics_all

def _hidden_irreps(channels: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{int(channels)}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(int(lmax) + 1)))



def _so3_flat_to_mace_features(x: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    blocks = _split_irreps(x, int(channels), int(lmax))
    return torch.cat([blocks[l] for l in range(int(lmax) + 1)], dim=-1)



def _merge_blocks_subset(blocks: Dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    return torch.cat([blocks[l].reshape(blocks[l].shape[0], int(channels) * (2 * l + 1)) for l in range(int(lmax) + 1)], dim=-1)



class SO3ToE3NNBasisBridge(nn.Module):
    """
    Fixed per-l orthogonal bridge between the ICTC SO3 basis and e3nn/MACE basis.

    `direction_harmonics_all` now matches e3nn component normalization in RMS, but
    each l block can still differ by an orthogonal basis convention. Native MACE
    contraction assumes the e3nn convention, while ICTC interaction emits ICTC
    convention features. The bridge uses a deterministic least-squares/SVD fit
    from sampled directions to construct Q_l such that:

        Y_ictd_l @ Q_l ~= Y_e3nn_l
    """

    def __init__(self, channels: int, lmax: int, num_samples: int = 8192):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260426)
        dirs = torch.randn(int(num_samples), 3, generator=generator, dtype=torch.float64)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y_ictd = direction_harmonics_all(dirs, self.lmax)
        y_e3nn = o3.spherical_harmonics(
            o3.Irreps.spherical_harmonics(self.lmax),
            dirs,
            normalize=True,
            normalization="component",
        )
        offset = 0
        for l in range(self.lmax + 1):
            width = 2 * l + 1
            a = y_ictd[l].to(dtype=torch.float64)
            b = y_e3nn[:, offset : offset + width].to(dtype=torch.float64)
            offset += width
            u, _, vh = torch.linalg.svd(a.T @ b)
            q = (u @ vh).to(dtype=torch.get_default_dtype())
            self.register_buffer(f"q_{l}", q, persistent=True)

    def _q(self, l: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return getattr(self, f"q_{int(l)}").to(dtype=dtype, device=device)

    def ictd_flat_to_e3nn_blocks(self, x: torch.Tensor, lmax: int) -> Dict[int, torch.Tensor]:
        blocks = _split_irreps(x, self.channels, int(lmax))
        out: Dict[int, torch.Tensor] = {}
        for l, block in blocks.items():
            out[l] = torch.einsum("ncm,mp->ncp", block, self._q(l, dtype=x.dtype, device=x.device))
        return out

    def e3nn_flat_to_ictd_blocks(self, x: torch.Tensor, lmax: int) -> Dict[int, torch.Tensor]:
        blocks = _split_irreps(x, self.channels, int(lmax))
        out: Dict[int, torch.Tensor] = {}
        for l, block in blocks.items():
            out[l] = torch.einsum("ncm,pm->ncp", block, self._q(l, dtype=x.dtype, device=x.device))
        return out

    def ictd_flat_to_e3nn_features(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.ictd_flat_to_e3nn_blocks(x, int(lmax))
        return torch.cat([blocks[l] for l in range(int(lmax) + 1)], dim=-1)

    def ictd_flat_to_e3nn_flat(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.ictd_flat_to_e3nn_blocks(x, int(lmax))
        return _merge_blocks_subset(blocks, self.channels, int(lmax))

    def e3nn_flat_to_ictd_flat(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.e3nn_flat_to_ictd_blocks(x, int(lmax))
        return _merge_blocks_subset(blocks, self.channels, int(lmax))

