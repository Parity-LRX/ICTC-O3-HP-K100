"""Energy readouts and the multipole feature handoff contract."""
from __future__ import annotations
import torch
import torch.nn as nn



from .common import _init_mace_style_linear_
from .layers import _Normalize2MomSiLU


class MACEStyleScalarReadoutSO3(nn.Module):
    """Native MACE final readout shape: Cx0e -> MLP_irreps scalar width -> 1x0e."""

    def __init__(self, channels: int, hidden_channels: int = 16, output_init_std: float = 0.003):
        """output_init_std: small value (0.003) so initial energy ≈ 0 (MLIP standard practice)."""
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.linear_1 = nn.Linear(self.channels, self.hidden_channels, bias=False)
        self.activation = _Normalize2MomSiLU()
        self.linear_2 = nn.Linear(self.hidden_channels, 1, bias=False)
        _init_mace_style_linear_(self.linear_1)
        _init_mace_style_linear_(self.linear_2)

    def forward(self, scalar_feats: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.activation(self.linear_1(scalar_feats)))


def multipole_feature_tap(last_full_so3_features, *, owned_count=None):
    """Return the final interaction features before scalar product truncation.

    The last product can emit only l=0. Multipole heads require the complete
    interaction representation, including l=1/2, even for a one-layer model.
    A wider edge_lmax representation is valid: the head selects its own blocks.
    Owner slicing is applied only when the simulation callback consumes sources.
    No feature is copied or detached here.
    """
    if last_full_so3_features is None:
        raise RuntimeError("multipole long-range needs a full-SO(3) node-feature layer; none found")
    return last_full_so3_features if owned_count is None else last_full_so3_features[:owned_count]

