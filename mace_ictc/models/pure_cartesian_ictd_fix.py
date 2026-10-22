from __future__ import annotations

import math
from numbers import Integral
from typing import Dict, List

import opt_einsum_fx
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from mace_ictc.utils.mesh import normalize_mesh_size
from mace_ictc.models.ictd_irreps import (
    EdgeWeightedPathPreservingTensorProduct,
    EquivariantChannelLinearSO3,
    EquivariantChannelLinearSO3Rect,
    HarmonicElementwiseProduct,
    HarmonicPathWeightedTensorProduct,
    direction_harmonics_all,
    ictd_u_matrix_so3,
)
from mace_ictc.models.radial_basis import mace_radial_embedding, mace_polynomial_cutoff
from mace_ictc.models.pure_cartesian_ictd_layers import (
    EquivariantScalarReadoutSO3,
    SO3BlockRMSNorm,
    _irreps_total_dim,
    _merge_irreps,
    _split_irreps,
    apply_channel_adapter_per_l,
    resolve_save_multiple_mix_channels,
)
from mace_ictc.models._mace_symmetric_contraction import MaceSymmetricContraction
from mace_ictc.models.mlp import MainNet
from mace_ictc.utils.scatter import scatter
from mace_ictc.models.long_range import build_long_range_module
from mace_ictc.models.dispersion import build_long_range_dispersion, normalize_dispersion_mode


_CONTRACTION_BATCH_EXAMPLE = 10
_CONTRACTION_ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


def _radial_mlp_widths(value) -> tuple[int, ...]:
    if value is None:
        return (64, 64, 64)
    if not isinstance(value, (list, tuple)):
        raise TypeError("radial_mlp_hidden must be a list or tuple of positive integers")
    if any(isinstance(width, bool) or not isinstance(width, Integral) or width < 1 for width in value):
        raise ValueError("radial_mlp_hidden must contain positive integers")
    return tuple(int(width) for width in value)


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


def _node_type_indices(node_attrs: torch.Tensor) -> torch.Tensor:
    if node_attrs.dim() == 1:
        return node_attrs.long()
    return node_attrs.argmax(dim=-1).long()


def _init_contraction_basis_logits_(logits: torch.Tensor, first_order_logit: float = 4.0) -> None:
    """Start from a stable order-1-dominant contraction instead of free large mixing."""
    with torch.no_grad():
        logits.zero_()
        logits[:, 0, :].fill_(float(first_order_logit))


def _init_contraction_basis_weight_(weight: torch.Tensor, higher_order_std: float = 0.02) -> None:
    """Free ablation: order-1 starts as passthrough, higher orders start small."""
    with torch.no_grad():
        weight.zero_()
        weight[:, 0, :].fill_(1.0)
        if weight.shape[1] > 1 and higher_order_std > 0:
            weight[:, 1:, :].normal_(mean=0.0, std=float(higher_order_std))


def _init_contraction_path_weight_(weight: torch.Tensor, std: float = 0.02) -> None:
    with torch.no_grad():
        weight.fill_(1.0)
        if std > 0:
            weight.add_(torch.randn_like(weight) * float(std))


def _init_path_tp_weight_to_one_(module: nn.Module | None) -> None:
    """ICTC path TPs multiply radial/contraction weights; do not start them near zero."""
    if module is not None and hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        with torch.no_grad():
            module.weight.fill_(1.0)


def _init_linear_identity_(module: nn.Module | None) -> None:
    if not isinstance(module, nn.Linear):
        return
    if module.weight.shape[0] != module.weight.shape[1]:
        return
    with torch.no_grad():
        nn.init.eye_(module.weight)
        if module.bias is not None:
            module.bias.zero_()


def _add_product_self_connection(
    out: torch.Tensor,
    sc: torch.Tensor,
    *,
    channels: int,
    target_lmax: int,
    input_lmax: int,
) -> torch.Tensor:
    """Add a MACE-style skip path to an SO(3) flattened product output.

    MACE's first residual interaction can produce an l=0-only self connection from
    scalar element embeddings, while the product output may still contain higher-l
    blocks. In that case the skip is added only to the output l=0 block.
    """
    if sc.shape[-1] == out.shape[-1]:
        return out + sc
    if sc.shape[-1] == channels:
        if target_lmax == 0:
            return out + sc
        out_blocks = _split_irreps(out, channels, target_lmax)
        out_blocks[0] = out_blocks[0] + sc.reshape(*sc.shape[:-1], channels, 1)
        return _merge_irreps(out_blocks, channels, target_lmax)
    if target_lmax == 0:
        return out + _split_irreps(sc, channels, input_lmax)[0].squeeze(-1)
    raise ValueError(f"Cannot add sc shape {tuple(sc.shape)} to product output {tuple(out.shape)}")


def _init_so3_linear_identity_(module: nn.Module | None) -> None:
    adapters = getattr(module, "adapters", None)
    if adapters is None:
        return
    for adapter in adapters.values():
        _init_linear_identity_(adapter)


def _init_so3_linear_mace_style_(module: nn.Module | None) -> None:
    adapters = getattr(module, "adapters", None)
    if adapters is None:
        return
    for adapter in adapters.values():
        with torch.no_grad():
            nn.init.normal_(adapter.weight, mean=0.0, std=1.0 / math.sqrt(float(adapter.in_features)))
            if adapter.bias is not None:
                adapter.bias.zero_()


def _init_element_conditioned_identity_(module: nn.Module | None) -> None:
    reset = getattr(module, "reset_identity_parameters", None)
    if callable(reset):
        reset()
        return
    weights = getattr(module, "weights", None)
    if weights is None:
        return
    with torch.no_grad():
        for weight in weights.values():
            if weight.shape[-2] != weight.shape[-1]:
                continue
            weight.zero_()
            eye = torch.eye(weight.shape[-1], dtype=weight.dtype, device=weight.device)
            weight.copy_(eye.unsqueeze(0).expand_as(weight))
        bias = getattr(module, "bias", None)
        if bias is not None:
            for value in bias.values():
                value.zero_()


def _init_element_conditioned_mace_style_(module: nn.Module | None) -> None:
    reset = getattr(module, "reset_mace_style_parameters", None)
    if callable(reset):
        reset()
        return
    weights = getattr(module, "weights", None)
    if weights is None:
        return
    with torch.no_grad():
        for weight in weights.values():
            fan_in = float(weight.shape[-1])
            nn.init.normal_(weight, mean=0.0, std=1.0 / math.sqrt(max(fan_in, 1.0)))
        bias = getattr(module, "bias", None)
        if bias is not None:
            for value in bias.values():
                value.zero_()


def _hidden_irreps(channels: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{int(channels)}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(int(lmax) + 1)))


def _mace_like_conv_tp_path_scales(
    tp: EdgeWeightedPathPreservingTensorProduct,
    *,
    channels: int,
    input_lmax: int,
    edge_lmax: int,
    n_probe: int = 6,
    seed: int = 12345,
) -> list[float]:
    """Recover per-path scalars mapping ICTC conv-TP paths to e3nn/MACE paths.

    MACE's convolution tensor product keeps every `(l_node, l_edge) -> l_out`
    path as a separate multiplicity block and applies e3nn's internal path
    normalization. ICTC's irreducible Cartesian CG tensors are equivalent but not
    necessarily in the same scalar convention. The converter calibrates the same
    constants from an actual MACE block; this helper builds the matching e3nn
    TensorProduct directly so from-scratch ICTC training can start in the same
    effective path scale.
    """
    from mace_ictc.mace_basis import orthogonal_Q_blocks

    param_dtype = tp.weight.dtype
    calib_dtype = torch.float64
    device = tp.weight.device
    C = int(channels)
    paths = [tuple(int(v) for v in p) for p in tp.paths]
    edge_lmax = int(edge_lmax)
    input_lmax = int(input_lmax)

    irreps_in1 = _hidden_irreps(C, input_lmax)
    irreps_in2 = o3.Irreps.spherical_harmonics(edge_lmax)
    irreps_out = o3.Irreps(
        [
            (
                C,
                o3.Irrep(int(l3), 1 if int(l3) % 2 == 0 else -1),
            )
            for _l1, _l2, l3 in paths
        ]
    )
    instructions = []
    for out_idx, (l1, l2, _l3) in enumerate(paths):
        if l1 > input_lmax:
            continue
        instructions.append((int(l1), int(l2), int(out_idx), "uvu", True))
    ref_tp = o3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        irrep_normalization="component",
        path_normalization="element",
        internal_weights=False,
        shared_weights=False,
    ).to(device=device, dtype=calib_dtype)

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    N = int(n_probe)
    q_blocks = orthogonal_Q_blocks(edge_lmax, dtype=calib_dtype, device=device)
    h_ictd = {
        l: torch.randn(N, C, 2 * l + 1, generator=g, dtype=calib_dtype).to(device)
        for l in range(input_lmax + 1)
    }
    h_e3nn = torch.cat(
        [
            torch.einsum("ncm,mp->ncp", h_ictd[l], q_blocks[l]).reshape(N, C * (2 * l + 1))
            for l in range(input_lmax + 1)
        ],
        dim=-1,
    )
    ndir = torch.randn(N, 3, generator=g, dtype=calib_dtype).to(device)
    ndir = ndir / ndir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y_ictd = direction_harmonics_all(ndir, edge_lmax)
    y_e3nn = torch.cat([y_ictd[l] @ q_blocks[l] for l in range(edge_lmax + 1)], dim=-1)

    weights = torch.randn(N, len(paths), C, generator=g, dtype=calib_dtype).to(device)
    ref_weights = weights.reshape(N, len(paths) * C)
    old_weight = tp.weight.detach().clone()
    old_compute_dtype = tp.internal_compute_dtype
    with torch.no_grad():
        try:
            tp.internal_compute_dtype = calib_dtype
            tp.weight.fill_(1.0)
            ref_out = ref_tp(h_e3nn, y_e3nn, ref_weights)
            edge_attrs = {l: y_ictd[l].unsqueeze(-2) for l in range(edge_lmax + 1)}
            ictd_out = tp(h_ictd, edge_attrs, ref_weights)
        finally:
            tp.weight.copy_(old_weight.to(dtype=param_dtype, device=device))
            tp.internal_compute_dtype = old_compute_dtype
            tp._cg_cache_by_dev_dtype.clear()
            tp._proj_group_cache_by_dev_dtype.clear()
            tp._proj_group_view_cache_by_dev_dtype.clear()

    scales = [1.0 for _ in paths]
    idx = 0
    max_resid = 0.0
    for path_idx, (_l1, _l2, l3) in enumerate(paths):
        d = C * (2 * int(l3) + 1)
        if int(_l1) > input_lmax:
            idx += d
            continue
        ref_block = ref_out[:, idx : idx + d].reshape(N, C, 2 * int(l3) + 1)
        off = int(tp.path_offset[path_idx])
        ictd_block = ictd_out[int(l3)][:, off * C : (off + 1) * C, :]
        ictd_block = torch.einsum("ncm,mp->ncp", ictd_block, q_blocks[int(l3)])
        num = (ref_block * ictd_block).sum()
        den = (ictd_block * ictd_block).sum().clamp_min(1e-30)
        scale = float((num / den).item())
        scales[path_idx] = scale
        max_resid = max(max_resid, float((ref_block - scale * ictd_block).abs().max().item()))
        idx += d
    if max_resid > 1e-6:
        raise RuntimeError(
            f"e3nn conv-TP path calibration residual too large ({max_resid:.2e}); "
            "ICTC and e3nn path conventions do not match."
        )
    return scales


def _so3_flat_to_mace_features(x: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    blocks = _split_irreps(x, int(channels), int(lmax))
    return torch.cat([blocks[l] for l in range(int(lmax) + 1)], dim=-1)


def _merge_blocks_subset(blocks: Dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    return torch.cat([blocks[l].reshape(blocks[l].shape[0], int(channels) * (2 * l + 1)) for l in range(int(lmax) + 1)], dim=-1)


def _concat_so3_states_by_l(states: List[torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    """
    Concatenate multiple SO3-flat states by channel within each l-block.

    Each state is laid out as [l0 | l1 | ...]. Directly concatenating states
    along the flat dimension would produce [s0_l0 | s0_l1 | ... | s1_l0 | ...],
    which is not a valid SO3-flat layout for a larger channel count. Equivariant
    operators expect [all_l0_channels | all_l1_channels | ...].
    """
    if len(states) == 0:
        raise ValueError("states must contain at least one SO3-flat tensor")
    split_states = [_split_irreps(state, int(channels), int(lmax)) for state in states]
    parts = []
    for l in range(int(lmax) + 1):
        block = torch.cat([split_state[l] for split_state in split_states], dim=-2)
        parts.append(block.reshape(*block.shape[:-2], block.shape[-2] * block.shape[-1]))
    return torch.cat(parts, dim=-1)


def _so3_block_rmsnorm(
    x: torch.Tensor,
    channels: int,
    lmax: int,
    gamma: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Apply an equivariant RMS normalization independently inside each l block."""
    blocks = _split_irreps(x, int(channels), int(lmax))
    parts = []
    gamma = gamma.to(dtype=x.dtype, device=x.device)
    for l in range(int(lmax) + 1):
        block = blocks[l]
        rms = block.square().mean(dim=(-2, -1), keepdim=True).add(float(eps)).sqrt()
        block = block / rms * gamma[l].view(*([1] * (block.ndim - 2)), 1, 1)
        parts.append(block.reshape(*block.shape[:-2], block.shape[-2] * block.shape[-1]))
    return torch.cat(parts, dim=-1)


def _tp_allowed_paths_from_target_lmax(lmax_in1: int, lmax_in2: int, lmax_target: int) -> List[tuple[int, int, int]]:
    """
    Mirror MACE/e3nn instruction pruning at the SO3 level:
    keep only paths (l1, l2, l3) whose output irrep l3 is present in the target set.

    For our current ICTC fix baseline, target irreps are exactly all l=0..lmax_target.
    This helper still makes the path set explicit and keeps interaction TP
    aligned with the same contract used by MACE TensorProduct instructions.
    """
    paths: List[tuple[int, int, int]] = []
    target_ls = set(range(int(lmax_target) + 1))
    for l1 in range(int(lmax_in1) + 1):
        for l2 in range(int(lmax_in2) + 1):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                if l3 not in target_ls:
                    continue
                if l3 > int(lmax_target):
                    continue
                if (l1 + l2 + l3) % 2 == 1:
                    continue
                paths.append((l1, l2, l3))
    return paths


def _tp_allowed_paths_to_output_l(lmax_in1: int, lmax_in2: int, output_l: int) -> List[tuple[int, int, int]]:
    paths: List[tuple[int, int, int]] = []
    l3 = int(output_l)
    for l1 in range(int(lmax_in1) + 1):
        for l2 in range(int(lmax_in2) + 1):
            if not (abs(l1 - l2) <= l3 <= l1 + l2):
                continue
            if (l1 + l2 + l3) % 2 == 1:
                continue
            paths.append((l1, l2, l3))
    return paths


class ElementConditionedLinearSO3(nn.Module):
    def __init__(self, num_elements: int, channels: int, lmax: int, bias: bool = False):
        super().__init__()
        self.num_elements = int(num_elements)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.dim = _irreps_total_dim(self.channels, self.lmax)
        self.weights = nn.ParameterDict(
            {
                str(l): nn.Parameter(torch.randn(self.num_elements, self.channels, self.channels) * 0.02)
                for l in range(self.lmax + 1)
            }
        )
        if bias:
            self.bias = nn.ParameterDict(
                {
                    str(l): nn.Parameter(torch.zeros(self.num_elements, self.channels))
                    for l in range(self.lmax + 1)
                }
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        attrs = node_attrs.to(dtype=x.dtype)
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            weight = self.weights[str(l)].to(dtype=x.dtype)
            mixed_weight = torch.einsum("ne,eoi->noi", attrs, weight)
            out_block = torch.einsum("noi,nid->nod", mixed_weight, blocks[l])
            if self.bias is not None:
                mixed_bias = torch.einsum("ne,eo->no", attrs, self.bias[str(l)].to(dtype=x.dtype))
                out_block = out_block + mixed_bias.unsqueeze(-1)
            out_blocks[l] = out_block
        return _merge_irreps(out_blocks, self.channels, self.lmax)

    def forward_type_idx(self, x: torch.Tensor, node_type_idx: torch.Tensor) -> torch.Tensor:
        idx = node_type_idx.to(device=x.device, dtype=torch.long)
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: Dict[int, torch.Tensor] = {}
        groups = None
        if getattr(self, "_grouped_species_gemm", False):
            from .species_linear import species_groups, grouped_species_linear
            groups = species_groups(idx, self.num_elements)
        for l in range(self.lmax + 1):
            weight = self.weights[str(l)].to(dtype=x.dtype, device=x.device)
            if groups is not None:
                bias = None if self.bias is None else self.bias[str(l)].to(dtype=x.dtype, device=x.device)
                out_blocks[l] = grouped_species_linear(blocks[l], weight, idx, bias, groups=groups)
                continue
            if getattr(self,"_shared_species_gemm",False):
                from .species_linear import shared_species_linear,supported
                if supported(blocks[l],weight):
                    bias=None if self.bias is None else self.bias[str(l)].to(dtype=x.dtype,device=x.device)
                    out_blocks[l]=shared_species_linear(blocks[l],weight,idx,bias)
                    continue
            mixed_weight = weight.index_select(0, idx)
            out_block = torch.einsum("noi,nid->nod", mixed_weight, blocks[l])
            if self.bias is not None:
                bias = self.bias[str(l)].to(dtype=x.dtype, device=x.device)
                mixed_bias = bias.index_select(0, idx)
                out_block = out_block + mixed_bias.unsqueeze(-1)
            out_blocks[l] = out_block
        return _merge_irreps(out_blocks, self.channels, self.lmax)


class LowRankElementConditionedLinearSO3(nn.Module):
    """Element-conditioned equivariant linear map with a shared residual core.

    For every spatial degree, the element-specific channel matrix is

    ``diag(g_z) + A_z @ B``.

    The diagonal gate preserves an identity-like information highway while the
    rank-``r`` correction provides element-specific cross-channel mixing.  This
    removes both the parameter cost and the per-node ``channels**2`` multiply of
    a fully independent matrix for every species.
    """

    def __init__(
        self,
        num_elements: int,
        channels: int,
        lmax: int,
        rank: int,
        bias: bool = False,
    ):
        super().__init__()
        self.num_elements = int(num_elements)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.dim = _irreps_total_dim(self.channels, self.lmax)
        self.element_gates = nn.ParameterDict(
            {
                str(l): nn.Parameter(torch.empty(self.num_elements, self.channels))
                for l in range(self.lmax + 1)
            }
        )
        self.element_left = nn.ParameterDict(
            {
                str(l): nn.Parameter(
                    torch.empty(self.num_elements, self.channels, self.rank)
                )
                for l in range(self.lmax + 1)
            }
        )
        self.shared_right = nn.ParameterDict(
            {
                str(l): nn.Parameter(torch.empty(self.rank, self.channels))
                for l in range(self.lmax + 1)
            }
        )
        if bias:
            self.bias = nn.ParameterDict(
                {
                    str(l): nn.Parameter(torch.zeros(self.num_elements, self.channels))
                    for l in range(self.lmax + 1)
                }
            )
        else:
            self.bias = None
        self.reset_identity_parameters()

    def reset_identity_parameters(self) -> None:
        with torch.no_grad():
            for l in range(self.lmax + 1):
                key = str(l)
                self.element_gates[key].fill_(1.0)
                self.element_left[key].zero_()
                nn.init.normal_(
                    self.shared_right[key],
                    mean=0.0,
                    std=1.0 / math.sqrt(float(self.channels)),
                )
                if self.bias is not None:
                    self.bias[key].zero_()

    def reset_mace_style_parameters(self) -> None:
        with torch.no_grad():
            for l in range(self.lmax + 1):
                key = str(l)
                nn.init.normal_(
                    self.element_gates[key],
                    mean=0.0,
                    std=1.0 / math.sqrt(float(self.channels)),
                )
                nn.init.normal_(
                    self.element_left[key],
                    mean=0.0,
                    std=1.0 / math.sqrt(float(self.rank)),
                )
                nn.init.normal_(
                    self.shared_right[key],
                    mean=0.0,
                    std=1.0 / math.sqrt(float(self.channels)),
                )
                if self.bias is not None:
                    self.bias[key].zero_()

    def set_from_full_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Best Frobenius low-rank fit of full per-element channel matrices."""
        with torch.no_grad():
            for l in range(self.lmax + 1):
                key = str(l)
                full = weights[key].to(
                    dtype=self.element_gates[key].dtype,
                    device=self.element_gates[key].device,
                )
                if tuple(full.shape) != (
                    self.num_elements,
                    self.channels,
                    self.channels,
                ):
                    raise ValueError(
                        f"full weight {key} has shape {tuple(full.shape)}, expected "
                        f"{(self.num_elements, self.channels, self.channels)}"
                    )
                gates = torch.diagonal(full, dim1=-2, dim2=-1)
                residual = (
                    full - torch.diag_embed(gates)
                ).reshape(
                    self.num_elements * self.channels,
                    self.channels,
                )
                u, s, vh = torch.linalg.svd(residual, full_matrices=False)
                used_rank = min(self.rank, int(s.numel()))
                left = torch.zeros_like(self.element_left[key])
                right = torch.zeros_like(self.shared_right[key])
                if used_rank > 0:
                    left[..., :used_rank].copy_(
                        (u[:, :used_rank] * s[:used_rank]).reshape(
                            self.num_elements,
                            self.channels,
                            used_rank,
                        )
                    )
                    right[:used_rank].copy_(vh[:used_rank])
                self.element_gates[key].copy_(gates)
                self.element_left[key].copy_(left)
                self.shared_right[key].copy_(right)
                if self.bias is not None:
                    self.bias[key].zero_()

    def _forward_blocks(
        self,
        x: torch.Tensor,
        *,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: Dict[int, torch.Tensor] = {}
        idx = (
            None
            if node_type_idx is None
            else node_type_idx.to(device=x.device, dtype=torch.long)
        )
        attrs = None if node_attrs is None else node_attrs.to(dtype=x.dtype, device=x.device)
        for l in range(self.lmax + 1):
            key = str(l)
            left_all = self.element_left[key].to(dtype=x.dtype, device=x.device)
            gates_all = self.element_gates[key].to(dtype=x.dtype, device=x.device)
            if idx is None:
                if attrs is None:
                    raise ValueError("node_attrs or node_type_idx is required")
                left = torch.einsum("ne,eor->nor", attrs, left_all)
                gates = torch.einsum("ne,eo->no", attrs, gates_all)
            else:
                left = left_all.index_select(0, idx)
                gates = gates_all.index_select(0, idx)
            block = blocks[l]
            bottleneck = torch.einsum(
                "ri,nid->nrd",
                self.shared_right[key].to(dtype=x.dtype, device=x.device),
                block,
            )
            out_block = gates.unsqueeze(-1) * block + torch.einsum(
                "nor,nrd->nod", left, bottleneck
            )
            if self.bias is not None:
                bias_all = self.bias[key].to(dtype=x.dtype, device=x.device)
                mixed_bias = (
                    torch.einsum("ne,eo->no", attrs, bias_all)
                    if idx is None
                    else bias_all.index_select(0, idx)
                )
                out_block = out_block + mixed_bias.unsqueeze(-1)
            out_blocks[l] = out_block
        return _merge_irreps(out_blocks, self.channels, self.lmax)

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        return self._forward_blocks(x, node_attrs=node_attrs, node_type_idx=None)

    def forward_type_idx(
        self,
        x: torch.Tensor,
        node_type_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_blocks(x, node_attrs=None, node_type_idx=node_type_idx)


class PerLScaleSO3(nn.Module):
    def __init__(self, channels: int, lmax: int, init_scales: list[float] | tuple[float, ...]):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        if len(init_scales) != self.lmax + 1:
            raise ValueError(f"Expected {self.lmax + 1} init scales, got {len(init_scales)}")
        scales = torch.as_tensor(init_scales, dtype=torch.get_default_dtype()).clamp_min(1e-6)
        self.log_scale = nn.Parameter(scales.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: Dict[int, torch.Tensor] = {}
        scales = self.log_scale.to(dtype=x.dtype, device=x.device).exp()
        for l in range(self.lmax + 1):
            out_blocks[l] = blocks[l] * scales[l]
        return _merge_irreps(out_blocks, self.channels, self.lmax)


class _Normalize2MomSiLU(nn.Module):
    _scale = 1.6791767923989418

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self,"_fuse_activation",False):
            from .fused_activation import normalized_silu
            return normalized_silu(x,self._scale)
        return F.silu(x) * self._scale


def _init_mace_style_linear_(linear: nn.Linear) -> None:
    with torch.no_grad():
        nn.init.normal_(linear.weight, mean=0.0, std=1.0 / math.sqrt(float(linear.in_features)))
        if linear.bias is not None:
            linear.bias.zero_()


def _node_component_message_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Share one channel matrix over contiguous node/angular rows. In particular,
    # avoid materializing [channel, node, angular] for the einsum's large GEMM.
    angular = x.movedim(-2, -1).contiguous()
    flat = angular.view(-1, angular.shape[-1])
    output = F.linear(flat, weight)
    return output.view(*angular.shape[:-1], weight.shape[0]).movedim(-1, -2)


class PathPreservingLinearSO3(nn.Module):
    def __init__(self, in_channels_by_l: Dict[int, int], out_channels: int, lmax: int):
        super().__init__()
        self.in_channels_by_l = {int(k): int(v) for k, v in in_channels_by_l.items()}
        self.out_channels = int(out_channels)
        self.lmax = int(lmax)
        self.weights = nn.ParameterDict()
        for l in range(self.lmax + 1):
            in_channels = self.in_channels_by_l.get(l, 0)
            weight = nn.Parameter(torch.empty(self.out_channels, in_channels))
            if in_channels > 0:
                nn.init.normal_(weight, mean=0.0, std=1.0 / math.sqrt(float(in_channels)))
            else:
                nn.init.zeros_(weight)
            self.weights[str(l)] = weight

    def forward(self, blocks: Dict[int, torch.Tensor]) -> torch.Tensor:
        out_blocks: Dict[int, torch.Tensor] = {}
        sample = next(iter(blocks.values()))
        for l in range(self.lmax + 1):
            x_l = blocks[l]
            weight = self.weights[str(l)].to(dtype=x_l.dtype, device=x_l.device)
            if x_l.shape[-2] == 0:
                out_blocks[l] = torch.zeros(
                    *x_l.shape[:-2], self.out_channels, 2 * l + 1, dtype=x_l.dtype, device=x_l.device
                )
            else:
                if (getattr(self, '_training_message_linear_layout', 'einsum') == 'node_component'
                        and x_l.is_cuda and x_l.dtype == torch.float32
                        and not torch.is_autocast_enabled(x_l.device.type)):
                    out_blocks[l] = _node_component_message_linear(x_l, weight)
                elif (getattr(self,'_inference_degree_layout',False) and not weight.requires_grad
                        and x_l.is_cuda and x_l.dtype==torch.float32 and x_l.stride(-2)==1
                        and not torch.is_autocast_enabled(x_l.device.type)):
                    angular=x_l.transpose(-2,-1)
                    out_blocks[l]=F.linear(angular,weight).transpose(-2,-1)
                else:
                    out_blocks[l] = torch.einsum("oc,...cm->...om", weight, x_l)
        return _merge_irreps(out_blocks, self.out_channels, self.lmax)


class ICTDSymmetricContractionSO3(nn.Module):
    """
    MACE-style symmetric contraction implemented with ICTC-SO3 operators.

    This keeps the higher-order ICTC-SO3 paths explicit as a basis list and then
    combines those basis terms with compact element-conditioned coefficients.
    That is closer in spirit to MACE's explicit product basis than the previous
    shared-contraction-plus-output-gating implementation.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        self.num_elements = int(num_elements)
        if contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(f"contraction_combine must be 'softmax', 'free', or 'path-free', got {contraction_combine!r}")
        self.contraction_combine = str(contraction_combine)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        _init_so3_linear_identity_(self.reduce)
        self.order_mix = nn.ModuleList(
            [EquivariantChannelLinearSO3(self.hidden_channels, self.lmax, bias=False) for _ in range(self.correlation)]
        )
        _init_so3_linear_identity_(self.order_mix[0])
        self.tp_layers = nn.ModuleList(
            [
                HarmonicPathWeightedTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        for tp in self.tp_layers:
            _init_path_tp_weight_to_one_(tp)
        self.tp_path_weight = nn.ParameterList()
        if self.contraction_combine == "path-free":
            for tp in self.tp_layers:
                weight = nn.Parameter(torch.empty(self.num_elements, tp.num_paths, self.hidden_channels))
                _init_contraction_path_weight_(weight)
                self.tp_path_weight.append(weight)
        self.out_linear = EquivariantChannelLinearSO3(
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        _init_so3_linear_identity_(self.out_linear)
        if self.contraction_combine == "softmax":
            self.basis_logits = nn.ParameterDict(
                {
                    str(l): nn.Parameter(
                        torch.zeros(self.num_elements, self.correlation, self.hidden_channels)
                    )
                    for l in range(self.lmax + 1)
                }
            )
            self.basis_weight = None
            for logits in self.basis_logits.values():
                _init_contraction_basis_logits_(logits)
        else:
            self.basis_logits = None
            self.basis_weight = nn.ParameterDict(
                {
                    str(l): nn.Parameter(
                        torch.empty(self.num_elements, self.correlation, self.hidden_channels)
                    )
                    for l in range(self.lmax + 1)
                }
            )
            for weight in self.basis_weight.values():
                _init_contraction_basis_weight_(weight)

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        element_index = _node_type_indices(node_attrs)

        basis_terms = [self.order_mix[0](base)]
        if self.correlation > 1:
            base_blocks = _split_irreps(base, self.hidden_channels, self.lmax)
            current_blocks = base_blocks
            for order_idx, tp in enumerate(self.tp_layers, start=1):
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.tp_path_weight[order_idx - 1][element_index].to(dtype=base.dtype)
                current_blocks = tp(current_blocks, base_blocks, path_channel_weights=path_weight)
                current_flat = _merge_irreps(current_blocks, self.hidden_channels, self.lmax)
                basis_terms.append(self.order_mix[order_idx](current_flat))

        basis_blocks = [_split_irreps(term, self.hidden_channels, self.lmax) for term in basis_terms]
        combined_blocks: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            if self.contraction_combine == "softmax":
                coeff = torch.softmax(self.basis_logits[str(l)][element_index].to(dtype=base.dtype), dim=1)
            else:
                coeff = self.basis_weight[str(l)][element_index].to(dtype=base.dtype)
            stack = torch.stack([term_blocks[l] for term_blocks in basis_blocks], dim=1)
            combined_blocks[l] = torch.sum(stack * coeff.unsqueeze(-1), dim=1)
        combined = _merge_irreps(combined_blocks, self.hidden_channels, self.lmax)
        return self.out_linear(combined)

    def forward_output_l(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        output_l: int,
    ) -> torch.Tensor:
        """Evaluate one requested output degree of the same contraction."""
        target_l = int(output_l)
        if not 0 <= target_l <= self.lmax:
            raise ValueError(
                f"output_l must be in [0, {self.lmax}], got {target_l}"
            )
        base = self.reduce(x)
        element_index = _node_type_indices(node_attrs)
        base_blocks = _split_irreps(base, self.hidden_channels, self.lmax)
        order0 = apply_channel_adapter_per_l(
            base_blocks[target_l],
            self.order_mix[0].adapters[str(target_l)],
        )
        basis_blocks = [order0]
        if self.correlation > 1:
            current_blocks = base_blocks
            last_order = len(self.tp_layers)
            for order_idx, tp in enumerate(self.tp_layers, start=1):
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.tp_path_weight[order_idx - 1][
                        element_index
                    ].to(dtype=base.dtype)
                if order_idx == last_order:
                    current_target = tp.forward_output_l(
                        current_blocks,
                        base_blocks,
                        target_l,
                        path_channel_weights=path_weight,
                    )
                else:
                    current_blocks = tp(
                        current_blocks,
                        base_blocks,
                        path_channel_weights=path_weight,
                    )
                    current_target = current_blocks[target_l]
                basis_blocks.append(
                    apply_channel_adapter_per_l(
                        current_target,
                        self.order_mix[order_idx].adapters[str(target_l)],
                    )
                )

        if self.contraction_combine == "softmax":
            coeff = torch.softmax(
                self.basis_logits[str(target_l)][element_index].to(
                    dtype=base.dtype
                ),
                dim=1,
            )
        else:
            coeff = self.basis_weight[str(target_l)][element_index].to(
                dtype=base.dtype
            )
        stack = torch.stack(basis_blocks, dim=1)
        combined = torch.sum(stack * coeff.unsqueeze(-1), dim=1)
        return apply_channel_adapter_per_l(
            combined,
            self.out_linear.adapters[str(target_l)],
        )


class ICTDProductBasisBlock(nn.Module):
    """
    MACE-style product block:
      h_{t+1} = linear( symmetric_contraction_ictd(message, node_attrs) ) + sc
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.symmetric_contractions = ICTDSymmetricContractionSO3(
            num_elements=num_elements,
            in_channels=channels,
            hidden_channels=channels,
            lmax=lmax,
            correlation=correlation,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
            contraction_combine=contraction_combine,
        )
        self.linear = EquivariantChannelLinearSO3(channels, lmax, bias=False)
        _init_so3_linear_identity_(self.linear)
        self.output_norm = nn.Identity()

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        contracted = self.symmetric_contractions(node_feats, node_attrs)
        out = self.linear(contracted)
        if sc is not None:
            out = out + sc
        return self.output_norm(out)


class ICTDScalarSymmetricContractionSO3(nn.Module):
    """
    Scalar-target MACE-style contraction.

    This mirrors MACE's final product target `64x0e`: the last TP step is
    instruction-pruned to l_out=0, and all order mixing/output projection happens
    only in scalar channel space.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        self.num_elements = int(num_elements)
        if contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(f"contraction_combine must be 'softmax', 'free', or 'path-free', got {contraction_combine!r}")
        self.contraction_combine = str(contraction_combine)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        _init_so3_linear_identity_(self.reduce)
        self.scalar_order_mix = nn.ModuleList(
            [nn.Linear(self.hidden_channels, self.hidden_channels, bias=False) for _ in range(self.correlation)]
        )
        _init_linear_identity_(self.scalar_order_mix[0])
        self.full_tp_layers = nn.ModuleList(
            [
                HarmonicPathWeightedTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 2, 0))
            ]
        )
        scalar_paths = _tp_allowed_paths_to_output_l(self.lmax, self.lmax, 0)
        self.final_scalar_tp = (
            HarmonicPathWeightedTensorProduct(
                channels=self.hidden_channels,
                lmax=self.lmax,
                allowed_paths=scalar_paths,
                path_policy=ictd_tp_path_policy,
                max_rank_other=ictd_tp_max_rank_other,
                internal_compute_dtype=internal_compute_dtype,
            )
            if self.correlation > 1
            else None
        )
        for tp in self.full_tp_layers:
            _init_path_tp_weight_to_one_(tp)
        _init_path_tp_weight_to_one_(self.final_scalar_tp)
        self.full_tp_path_weight = nn.ParameterList()
        self.final_scalar_path_weight = None
        if self.contraction_combine == "path-free":
            for tp in self.full_tp_layers:
                weight = nn.Parameter(torch.empty(self.num_elements, tp.num_paths, self.hidden_channels))
                _init_contraction_path_weight_(weight)
                self.full_tp_path_weight.append(weight)
            if self.final_scalar_tp is not None:
                self.final_scalar_path_weight = nn.Parameter(
                    torch.empty(self.num_elements, self.final_scalar_tp.num_paths, self.hidden_channels)
                )
                _init_contraction_path_weight_(self.final_scalar_path_weight)
        self.out_linear = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        _init_linear_identity_(self.out_linear)
        if self.contraction_combine == "softmax":
            self.basis_logits = nn.Parameter(
                torch.zeros(self.num_elements, self.correlation, self.hidden_channels)
            )
            self.basis_weight = None
            _init_contraction_basis_logits_(self.basis_logits)
        else:
            self.basis_logits = None
            self.basis_weight = nn.Parameter(
                torch.empty(self.num_elements, self.correlation, self.hidden_channels)
            )
            _init_contraction_basis_weight_(self.basis_weight)

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        base_blocks = _split_irreps(base, self.hidden_channels, self.lmax)
        element_index = _node_type_indices(node_attrs)

        basis_terms = [self.scalar_order_mix[0](base_blocks[0].squeeze(-1))]
        current_blocks = base_blocks
        for order_idx in range(1, self.correlation):
            if order_idx == self.correlation - 1:
                if self.final_scalar_tp is None:
                    raise RuntimeError("final_scalar_tp unexpectedly missing")
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.final_scalar_path_weight[element_index].to(dtype=base.dtype)
                current_blocks = self.final_scalar_tp(current_blocks, base_blocks, path_channel_weights=path_weight)
            else:
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.full_tp_path_weight[order_idx - 1][element_index].to(dtype=base.dtype)
                current_blocks = self.full_tp_layers[order_idx - 1](
                    current_blocks,
                    base_blocks,
                    path_channel_weights=path_weight,
                )
            scalar = current_blocks[0].squeeze(-1)
            basis_terms.append(self.scalar_order_mix[order_idx](scalar))

        if self.contraction_combine == "softmax":
            coeff = torch.softmax(self.basis_logits[element_index].to(dtype=base.dtype), dim=1)
        else:
            coeff = self.basis_weight[element_index].to(dtype=base.dtype)
        stack = torch.stack(basis_terms, dim=1)
        combined = torch.sum(stack * coeff, dim=1)
        return self.out_linear(combined)


class ICTDScalarProductBasisBlock(nn.Module):
    """
    MACE-style final product block for keep_last_layer_irreps=False.

    Native MACE changes the last product target irreps to scalar-only. This
    block uses a scalar-target ICTC contraction instead of building the full
    output irreps and slicing l=0 afterward.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.symmetric_contractions = ICTDScalarSymmetricContractionSO3(
            num_elements=num_elements,
            in_channels=channels,
            hidden_channels=channels,
            lmax=lmax,
            correlation=correlation,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
            contraction_combine=contraction_combine,
        )
        self.linear = nn.Linear(self.channels, self.channels, bias=False)
        _init_linear_identity_(self.linear)
        self.output_norm = nn.Identity()

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        contracted = self.symmetric_contractions(node_feats, node_attrs)
        out = self.linear(contracted)
        if sc is not None:
            if sc.shape[-1] == self.channels:
                sc_scalar = sc
            else:
                sc_scalar = _split_irreps(sc, self.channels, self.lmax)[0].squeeze(-1)
            out = out + sc_scalar
        return self.output_norm(out)


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


class NativeMACEProductBasisBlockSO3(nn.Module):
    """
    Hybrid product block: ICTC-SO3 interaction features are interpreted in
    MACE/e3nn mul-ir layout, then contracted by the native MACE symmetric
    contraction implementation.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=int(correlation),
            num_elements=int(num_elements),
            use_reduced_cg=self.use_reduced_cg,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, max(self.lmax, self.target_lmax))

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        x = self.basis_bridge.ictd_flat_to_e3nn_features(node_feats, self.lmax)
        out = self.linear(self.symmetric_contractions(x, node_attrs))
        if sc is not None:
            if sc.shape[-1] != self.channels:
                sc = self.basis_bridge.ictd_flat_to_e3nn_flat(sc, self.target_lmax)
            out = _add_product_self_connection(
                out,
                sc,
                channels=self.channels,
                target_lmax=self.target_lmax,
                input_lmax=self.lmax,
            )
        if self.target_lmax > 0:
            out = self.basis_bridge.e3nn_flat_to_ictd_flat(out, self.target_lmax)
        return out


def _cueq_o3_e3nn_group():
    try:
        from mace.tools.cg import O3_e3nn
    except Exception:
        try:
            from mace_ictc.models._mace_cg import O3_e3nn
        except Exception as exc:  # pragma: no cover - optional accelerator dependency
            raise RuntimeError(
                "ictd_fix_product_backend='cueq' requires the e3nn-compatible "
                "cuequivariance O3 group from mace.tools.cg or mace_ictc.models._mace_cg"
            ) from exc
    return O3_e3nn


def _cueq_o3_irreps(channels: int, lmax: int):
    """Build a cuequivariance O3 irreps string matching the local MACE/e3nn convention."""
    try:
        import cuequivariance as cue
    except Exception as exc:  # pragma: no cover - optional accelerator dependency
        raise RuntimeError(
            "ictd_fix_product_backend='cueq' requires cuequivariance and "
            "cuequivariance_torch to be installed"
        ) from exc
    terms = []
    for l in range(int(lmax) + 1):
        parity = "e" if l % 2 == 0 else "o"
        terms.append(f"{int(channels)}x{l}{parity}")
    return cue.Irreps(_cueq_o3_e3nn_group(), " + ".join(terms))


class CueqMaceSymmetricContractionSO3(nn.Module):
    """MACE symmetric contraction accelerated by cuEquivariance.

    The inner ``symmetric_contractions`` module keeps the exact MACE parameter
    layout used by the converter and by existing checkpoints. The cuEquivariance
    modules hold frozen mirror weights for eval/inference. CUDA training uses
    cuEquivariance's contraction graph with weights computed differentiably from
    the authoritative MACE parameters, so gradients still land on checkpoint-
    compatible tensors.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.correlation = int(correlation)
        self.use_reduced_cg = bool(use_reduced_cg)
        if self.correlation not in {1, 2, 3, 4}:
            raise NotImplementedError(
                "cuEquivariance product backend currently supports correlation=1,2,3,4"
            )

        try:
            import cuequivariance as cue
            import cuequivariance_torch as cuet
        except Exception as exc:  # pragma: no cover - optional accelerator dependency
            raise RuntimeError(
                "ictd_fix_product_backend='cueq' requires cuequivariance and "
                "cuequivariance_torch to be installed"
            ) from exc

        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=self.correlation,
            num_elements=int(num_elements),
            use_reduced_cg=self.use_reduced_cg,
        )
        self._cueq_weight_specs = [
            self._make_cueq_weight_spec(ref)
            for ref in self.symmetric_contractions.contractions
        ]

        irreps_in = _cueq_o3_irreps(self.channels, self.lmax)
        method = "uniform_1d" if torch.cuda.is_available() else "naive"
        self.cueq_contractions = nn.ModuleList()
        self._cueq_weight_projection_names: list[str | None] = []
        weight_projection_fn = None
        if self.use_reduced_cg:
            try:
                from mace.tools.cg_cueq_tools import symmetric_contraction_proj as weight_projection_fn
            except ImportError as exc:  # pragma: no cover - optional conversion dependency
                raise RuntimeError(
                    "ictd_fix_product_backend='cueq' with reduced CG requires "
                    "mace.tools.cg_cueq_tools.symmetric_contraction_proj from a compatible "
                    "upstream mace-torch installation. Install or upgrade the 'conversion' "
                    "extra to provide the MACE-to-cuEq weight projection."
                ) from exc
        for out_idx, (_mul, ir) in enumerate(self.target_irreps):
            out_l = int(ir.l)
            parity = "e" if out_l % 2 == 0 else "o"
            irreps_out = cue.Irreps(_cueq_o3_e3nn_group(), f"{self.channels}x{out_l}{parity}")
            projection_name = None
            if weight_projection_fn is not None:
                _, projection = weight_projection_fn(
                    irreps_in,
                    irreps_out,
                    tuple(range(1, self.correlation + 1)),
                )
                projection_name = f"_cueq_weight_projection_{out_idx}"
                self.register_buffer(
                    projection_name,
                    torch.tensor(projection, dtype=torch.get_default_dtype()),
                    persistent=False,
                )
            self._cueq_weight_projection_names.append(projection_name)
            self.cueq_contractions.append(
                cuet.SymmetricContraction(
                    irreps_in,
                    irreps_out,
                    contraction_degree=self.correlation,
                    num_elements=int(num_elements),
                    layout_in=cue.ir_mul,
                    layout_out=cue.mul_ir,
                    dtype=torch.get_default_dtype(),
                    math_dtype=torch.get_default_dtype(),
                    original_mace=not self.use_reduced_cg,
                    method=method,
                )
            )
            self.cueq_contractions[-1].weight.requires_grad_(False)
        self.refresh_cueq_weights()
        self.register_load_state_dict_post_hook(lambda module, _incompatible: module.refresh_cueq_weights())

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.refresh_cueq_weights()
        return self

    def _make_cueq_weight_spec(self, ref: nn.Module) -> list[int | str]:
        spec: list[int | str] = ["max"]
        spec.extend(range(len(ref.weights)))
        if self.use_reduced_cg:
            return spec
        active: list[int | str] = []
        for item in spec:
            flag_name = "weights_max_zeroed" if item == "max" else f"weights_{int(item)}_zeroed"
            zeroed = bool(getattr(ref, flag_name).detach().cpu().item()) if hasattr(ref, flag_name) else False
            if not zeroed:
                active.append(item)
        return active

    def _mace_weights_for_cueq(self, ref: nn.Module) -> torch.Tensor:
        """Pack local MACE symmetric-contraction weights in cuEq path order."""
        try:
            spec_idx = list(self.symmetric_contractions.contractions).index(ref)
            spec = self._cueq_weight_specs[spec_idx]
        except ValueError:
            spec = self._make_cueq_weight_spec(ref)
        parts = [
            ref.weights_max if item == "max" else ref.weights[int(item)]
            for item in spec
        ]
        if parts:
            return torch.cat(parts, dim=1)
        return ref.weights_max.new_zeros((ref.weights_max.shape[0], 0, ref.weights_max.shape[-1]))

    def refresh_cueq_weights(self) -> None:
        """Mirror MACE-contraction weights into cuEquivariance path-weight tensors."""
        with torch.no_grad():
            for ref, fast, projection_name in zip(
                self.symmetric_contractions.contractions,
                self.cueq_contractions,
                self._cueq_weight_projection_names,
            ):
                weights = self._mace_weights_for_cueq(ref)
                weights = weights.to(dtype=fast.weight.dtype, device=fast.weight.device)
                if projection_name is not None:
                    projection = getattr(self, projection_name).to(
                        dtype=fast.weight.dtype,
                        device=fast.weight.device,
                    )
                    weights = torch.einsum("zau,ab->zbu", weights, projection)
                if weights.shape == fast.weight.shape:
                    fast.weight.copy_(weights)
                    continue
                raise ValueError(
                    "Cannot mirror MACE contraction weights into cueq contraction: "
                    f"source={tuple(weights.shape)} target={tuple(fast.weight.shape)} "
                    f"projection={None if getattr(fast, 'projection', None) is None else tuple(fast.projection.shape)}"
                )

    def _cueq_forward_with_weight(
        self,
        fast: nn.Module,
        flat: torch.Tensor,
        idx: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        projection = getattr(fast, "projection", None)
        if projection is not None:
            projection = projection.to(dtype=weight.dtype, device=weight.device)
            weight = torch.einsum("zau,ab->zbu", weight, projection)
        output = fast.f([weight.flatten(1), fast.transpose_in(flat)], input_indices={0: idx})
        return fast.transpose_out(output[0])

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            if node_attrs is None:
                raise ValueError("node_attrs is required for the reference MACE contraction path")
            return self.symmetric_contractions(x, node_attrs)
        if node_type_idx is None:
            if node_attrs is None:
                raise ValueError("node_type_idx or node_attrs is required for cueq contraction")
            idx = _node_type_indices(node_attrs).to(device=x.device, dtype=torch.int32)
        else:
            idx = node_type_idx.to(device=x.device, dtype=torch.int32)
        flat = x.transpose(1, 2).reshape(x.shape[0], -1)
        if self.training:
            outs = []
            for ref, fast, projection_name in zip(
                self.symmetric_contractions.contractions,
                self.cueq_contractions,
                self._cueq_weight_projection_names,
            ):
                weights = self._mace_weights_for_cueq(ref).to(dtype=flat.dtype, device=flat.device)
                if projection_name is not None:
                    projection = getattr(self, projection_name).to(dtype=weights.dtype, device=weights.device)
                    weights = torch.einsum("zau,ab->zbu", weights, projection)
                outs.append(self._cueq_forward_with_weight(fast, flat, idx, weights))
        else:
            outs = [contraction(flat, idx) for contraction in self.cueq_contractions]
        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]


class CueqMACEProductBasisBlockSO3(nn.Module):
    """Native-MACE product block using cuEquivariance for the contraction."""

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self._e3nn_basis = False
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, max(self.lmax, self.target_lmax))
        self.symmetric_contractions = CueqMaceSymmetricContractionSO3(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=target_lmax,
            correlation=correlation,
            use_reduced_cg=self.use_reduced_cg,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)

    def refresh_cueq_weights(self) -> None:
        self.symmetric_contractions.refresh_cueq_weights()

    def enable_e3nn_basis(self, q_blocks: "List[torch.Tensor] | None" = None) -> None:
        """Consume and return e3nn-basis features directly.

        cuEquivariance's MACE contraction already uses the e3nn/O3 convention, so
        when the rest of the model has folded its interaction CGs into that same
        basis we can skip the per-l ICTC<->e3nn bridge around the product block.
        """
        del q_blocks
        self._e3nn_basis = True

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: torch.Tensor | None,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._e3nn_basis:
            x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        else:
            x = self.basis_bridge.ictd_flat_to_e3nn_features(node_feats, self.lmax)
        out = self.linear(self.symmetric_contractions(x, node_attrs, node_type_idx=node_type_idx))
        if sc is not None:
            if (not self._e3nn_basis) and sc.shape[-1] != self.channels:
                sc = self.basis_bridge.ictd_flat_to_e3nn_flat(sc, self.target_lmax)
            out = _add_product_self_connection(
                out,
                sc,
                channels=self.channels,
                target_lmax=self.target_lmax,
                input_lmax=self.lmax,
            )
        if self.target_lmax > 0 and not self._e3nn_basis:
            out = self.basis_bridge.e3nn_flat_to_ictd_flat(out, self.target_lmax)
        return out


class ICTDBridgeUSymmetricContractionSO3(nn.Module):
    """
    Bridge-U symmetric contraction expressed directly in the ICTC basis.

    This is algebraically equivalent to:
      ICTC features -> e3nn basis -> MACE SymmetricContraction -> ICTC basis
    but the per-l basis change is folded into the stored U tensors once at
    initialization. The forward path therefore consumes and returns ICTC-basis
    flat SO3 features. This backend is the stable high-l bridge used when
    pure ICTC U generation is not numerically reliable.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, max(self.lmax, self.target_lmax))
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=int(correlation),
            num_elements=int(num_elements),
            use_reduced_cg=self.use_reduced_cg,
        )
        self._fold_basis_change_into_u_tensors()

    def _input_q(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        blocks = [self.basis_bridge._q(l, dtype=dtype, device=device) for l in range(self.lmax + 1)]
        return torch.block_diag(*blocks)

    def _transform_u_tensor(self, u_tensor: torch.Tensor, output_l: int) -> torch.Tensor:
        dtype = u_tensor.dtype
        device = u_tensor.device
        q_in = self._input_q(dtype=dtype, device=device)
        nu = int(u_tensor.dim() - 1) if int(output_l) == 0 else int(u_tensor.dim() - 2)
        if int(output_l) == 0:
            if nu == 1:
                return torch.einsum("ai,ip->ap", q_in, u_tensor)
            if nu == 2:
                return torch.einsum("ai,bj,ijp->abp", q_in, q_in, u_tensor)
            if nu == 3:
                return torch.einsum("ai,bj,ck,ijkp->abcp", q_in, q_in, q_in, u_tensor)
        q_out = self.basis_bridge._q(output_l, dtype=dtype, device=device)
        if nu == 1:
            return torch.einsum("ro,ai,oip->rap", q_out, q_in, u_tensor)
        if nu == 2:
            return torch.einsum("ro,ai,bj,oijp->rabp", q_out, q_in, q_in, u_tensor)
        if nu == 3:
            return torch.einsum("ro,ai,bj,ck,oijkp->rabcp", q_out, q_in, q_in, q_in, u_tensor)
        # Preserve the established small-order expressions. Higher orders use
        # the same fixed basis map on each tensor leg, without a cubic cap or
        # an einsum alphabet limit. This only lowers immutable U coefficients.
        if int(output_l)==0:
            result=u_tensor
            first_axis=0
        else:
            result=torch.tensordot(q_out,u_tensor,dims=([1],[0]))
            first_axis=1
        for axis in range(first_axis,first_axis+nu):
            result=torch.tensordot(q_in,result,dims=([1],[axis])).movedim(0,axis)
        return result

    def _fold_basis_change_into_u_tensors(self) -> None:
        with torch.no_grad():
            for (mul, ir), contraction in zip(self.target_irreps, self.symmetric_contractions.contractions):
                del mul
                output_l = int(ir.l)
                for nu in range(1, int(contraction.correlation) + 1):
                    name = f"U_matrix_{nu}"
                    old = getattr(contraction, name)
                    new = self._transform_u_tensor(old, output_l)
                    old.copy_(new.to(dtype=old.dtype, device=old.device))
                if getattr(contraction, "_use_scalar_corr3_fast", False) and hasattr(
                    contraction, "refresh_scalar_corr3_fast_buffers"
                ):
                    contraction.refresh_scalar_corr3_fast_buffers()

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        deployment=getattr(self,"_deployment_contraction",None)
        if deployment is not None:
            if node_type_idx is None:node_type_idx=node_attrs.argmax(-1)
            return deployment(x,node_type_idx)
        if node_type_idx is None:
            from .reference_u_residency import prepare_reference_u
            prepare_reference_u(self,x)
            return self.symmetric_contractions(x, node_attrs)
        if getattr(self, "_factorized_contraction", False):
            from .factorized_contraction import factorized_contractions
            factorized = factorized_contractions(self, x, node_type_idx)
            if factorized is not None:
                return factorized
        if getattr(self, "_compact_contraction", False):
            from .compact_contraction import compact_contractions
            compact = compact_contractions(self.symmetric_contractions.contractions, x, node_type_idx)
            if compact is not None:
                return compact
        from .reference_u_residency import prepare_reference_u
        prepare_reference_u(self,x)
        return torch.cat(
            [
                contraction(x, node_attrs, node_type_idx=node_type_idx)
                for contraction in self.symmetric_contractions.contractions
            ],
            dim=-1,
        )


class ICTDBridgeUProductBasisBlockSO3(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = ICTDBridgeUSymmetricContractionSO3(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=target_lmax,
            correlation=correlation,
            use_reduced_cg=self.use_reduced_cg,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: torch.Tensor | None,
        node_attrs: torch.Tensor,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.linear(
            self.symmetric_contractions(
                node_feats,
                node_attrs,
                node_type_idx=node_type_idx,
            )
        )
        if sc is not None:
            out = _add_product_self_connection(
                out,
                sc,
                channels=self.channels,
                target_lmax=self.target_lmax,
                input_lmax=self.lmax,
            )
        return out


# Backward-compatible aliases for checkpoints/scripts that still refer to the old name.
ICTDMACEUSymmetricContractionSO3 = ICTDBridgeUSymmetricContractionSO3
ICTDMACEUProductBasisBlockSO3 = ICTDBridgeUProductBasisBlockSO3


class _ICTDPureUContraction(nn.Module):
    """MACE contraction recursion over caller-provided ICTC U tensors."""

    def __init__(
        self,
        *,
        u_tensors: Dict[int, torch.Tensor],
        output_l: int,
        num_elements: int,
        num_features: int,
    ):
        super().__init__()
        self.output_l = int(output_l)
        self.correlation = int(max(u_tensors))
        self.num_elements = int(num_elements)
        self.num_features = int(num_features)
        for nu in range(1, self.correlation + 1):
            self.register_buffer(f"U_matrix_{nu}", u_tensors[nu].contiguous())

        self.contractions_weighting = nn.ModuleList()
        self.contractions_features = nn.ModuleList()
        self.weights = nn.ParameterList([])

        for i in range(self.correlation, 0, -1):
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * self.output_l + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == self.correlation:
                parse_subscript_main = (
                    [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum("".join(parse_subscript_main), x, y, w, z)
                )
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn([num_equivariance] + [num_ell] * i + [num_params]).squeeze(0),
                        torch.randn((self.num_elements, num_params, self.num_features)),
                        torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_elements)),
                    ),
                )
                self.weights_max = nn.Parameter(
                    torch.randn((self.num_elements, num_params, self.num_features)) / max(num_params, 1)
                )
            else:
                parse_subscript_weighting = (
                    [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1))]
                    + ["k,ekc,be->bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i - 1 + min(self.output_l, 1))]
                    + ["i,bci->bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i - 1 + min(self.output_l, 1))]
                )

                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum("".join(parse_subscript_weighting), x, y, z)
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )
                self.contractions_weighting.append(
                    opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_weighting,
                        example_inputs=(
                            torch.randn([num_equivariance] + [num_ell] * i + [num_params]).squeeze(0),
                            torch.randn((self.num_elements, num_params, self.num_features)),
                            torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_elements)),
                        ),
                    )
                )
                self.contractions_features.append(
                    opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_features,
                        example_inputs=(
                            torch.randn(
                                [_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_equivariance]
                                + [num_ell] * i
                            ).squeeze(2),
                            torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_ell)),
                        ),
                    )
                )
                self.weights.append(
                    nn.Parameter(torch.randn((self.num_elements, num_params, self.num_features)) / max(num_params, 1))
                )

    def U_tensors(self, nu: int) -> torch.Tensor:
        return dict(self.named_buffers())[f"U_matrix_{int(nu)}"]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.graph_opt_main(self.U_tensors(self.correlation), self.weights_max, x, y)
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(self.U_tensors(self.correlation - i - 1), weight, y)
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)
        return out.view(out.shape[0], -1)


class ICTDPureUSymmetricContractionSO3(nn.Module):
    """
    MACE-style symmetric contraction with U tensors generated from ICTC CG only.

    This keeps the optimized MACE contraction/einsum wrapper and trainable
    per-element weights, but replaces every `U_matrix_real` buffer with the
    corresponding ICTC-basis U generated by `ictd_u_matrix_so3`. It is the
    pure-ICTC contraction ablation against `ictd-bridge-u`.

    NATIVE-TRAINING ONLY -- does NOT correspond to an e3nn/MACE checkpoint. The
    ICTC U is obtained by solving CG null-spaces via SVD, whose basis is not unique;
    the resulting path gauge spans the correct intertwiner space but does not align
    with e3nn's Wigner-3j paths. A trained e3nn checkpoint's contraction weights
    therefore have no image here, so this backend cannot reproduce a pretrained
    spherical model -- use `ictd-bridge-u` (which reuses e3nn's own U) for exact
    checkpoint utilization. (Empirically edge_lmax==lmax or random weights look
    close, ~1e-5 in fp64, but real pretrained weights diverge by tens of percent on
    a converted MACE-OFF23: that small residual is a benign-regime artifact, not a
    fidelity floor.)
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.contractions = nn.ModuleList()
        dtype = torch.get_default_dtype()
        for mul, ir in self.target_irreps:
            del mul
            output_l = int(ir.l)
            u_tensors = {
                nu: ictd_u_matrix_so3(
                    lmax=self.lmax,
                    output_l=output_l,
                    correlation=nu,
                    irrep_normalization="component",
                    dtype=dtype,
                )
                for nu in range(1, int(correlation) + 1)
            }
            self.contractions.append(
                _ICTDPureUContraction(
                    u_tensors=u_tensors,
                    output_l=output_l,
                    num_elements=int(num_elements),
                    num_features=self.channels,
                )
            )

        # angular_basis='e3nn': wrap this (order-nu) contraction so it RUNS in the ICTC basis
        # (U tensors stay ICTC) yet consumes/returns e3nn-basis features. Folding Q into the
        # order-3 U is mathematically identical but introduces input-dependent float64 cancellation
        # in the rotated basis (~1e-6 on the l>=1 features); the wrap keeps the contraction output
        # equal to (ictd output) @ Q to MACHINE PRECISION. The interaction TP (order 2) folds
        # exactly, so only the contraction is wrapped.
        self._e3nn_basis = False
        self._e3nn_bridge: SO3ToE3NNBasisBridge | None = None

    def enable_e3nn_basis(self, q_blocks: "List[torch.Tensor] | None" = None) -> None:
        """Make this contraction consume + return e3nn-basis features: rotate the input
        e3nn->ICTC, run the numerically-stable ICTC contraction, rotate the output ICTC->e3nn."""
        del q_blocks  # the bridge rebuilds the identical Q (same deterministic Procrustes fit)
        self._e3nn_basis = True
        if self._e3nn_bridge is None:
            self._e3nn_bridge = SO3ToE3NNBasisBridge(self.channels, self.lmax)

    def forward(self, node_feats: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        if self._e3nn_basis:
            node_feats = self._e3nn_bridge.e3nn_flat_to_ictd_flat(node_feats, self.lmax)
        x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        out = torch.cat([contraction(x, node_attrs) for contraction in self.contractions], dim=-1)
        if self._e3nn_basis and self.target_lmax > 0:
            out = self._e3nn_bridge.ictd_flat_to_e3nn_flat(out, self.target_lmax)
        return out


class ICTDPureUProductBasisBlockSO3(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = ICTDPureUSymmetricContractionSO3(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=target_lmax,
            correlation=correlation,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.symmetric_contractions(node_feats, node_attrs))
        if sc is not None:
            out = _add_product_self_connection(
                out,
                sc,
                channels=self.channels,
                target_lmax=self.target_lmax,
                input_lmax=self.lmax,
            )
        return out


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


class CueqConvTensorProductSO3(nn.Module):
    """cuEquivariance backend for the interaction convolution tensor product.

    The reference ``EdgeWeightedPathPreservingTensorProduct`` evaluates the conv-TP
    with dense per-``(l1, l2)`` projector matmuls in the ICTC Cartesian basis.
    This module evaluates the same algebra with cuEquivariance's fused CUDA
    segmented-polynomial kernel (``cuet.ChannelWiseTensorProduct``), folding the
    per-edge/per-channel gates AND the learnable per-path channel weights into the
    kernel's external weight operand.

    The ICTC and e3nn/MACE bases differ only by fixed per-l orthogonal matrices
    ``Q_l`` (see ``mace_ictc.mace_basis``), and every linear map around the TP acts
    on the channel axis only, so the basis change is an exact boundary bridge:

        out_ictc_p  =  Q_l3^T @ cueq_p(Q_l1 @ x1, Q_l2 @ x2;  u_p)
        u_p         =  gates_p * w_p / s_p

    where ``1/s_p`` is the reference TP's explicit element/path-normalization
    scalar.  The frozen ICTC basis fixes the same positive component-normalized
    natural-parity CG convention as the Q-bridged e3nn representation, so no
    high-L CG table needs to be rebuilt just to recover this scalar.

    Layout is derived from cuEquivariance's public segmented-polynomial descriptor:
    both weight and output segments are explicitly mapped back to reference
    ``(l1,l2,l3)`` paths.  No undocumented cuEquivariance sorting convention is
    assumed.

    The module holds NO trainable parameters: the authoritative per-path channel
    weights stay in ``tp.weight`` (checkpoint-compatible) and are passed to
    ``forward`` so gradients land on the reference tensors.  All buffers (Q
    bridges, inverse path scales, path permutation) are non-persistent, so the
    state_dict layout is unchanged relative to the pytorch backend.  The compute
    dtype is fixed at construction (from ``tp.internal_compute_dtype``), matching
    the existing cueq product backend; cueq's fused ops are CUDA-only, so the
    caller keeps the reference pytorch path as the CPU fallback.

    Only the message-passing TP of ``ICTDResidualInteractionBlock`` is mirrored
    here: path list ``tp.paths`` (already pruned to ``l1 <= input_lmax`` and
    ``l3 <= target_lmax``), component irrep normalization and element path
    normalization (the only normalization cueq supports, and exactly the
    convention of the e3nn reference TP used for calibration).
    """

    def __init__(
        self,
        *,
        tp: EdgeWeightedPathPreservingTensorProduct,
        input_lmax: int,
        target_lmax: int,
    ):
        super().__init__()
        try:
            import cuequivariance as cue
            import cuequivariance_torch as cuet
        except Exception as exc:  # pragma: no cover - optional accelerator dependency
            raise RuntimeError(
                "ictd_tp_backend='cueq' requires cuequivariance and "
                "cuequivariance_torch to be installed"
            ) from exc

        from mace_ictc.mace_basis import orthogonal_Q_blocks

        self.channels = int(tp.channels)
        self.lmax = int(tp.lmax)  # edge (in2) lmax
        self.input_lmax = int(input_lmax)
        self.target_lmax = int(target_lmax)
        self.num_paths = int(tp.num_paths)
        self.compute_dtype = tp.internal_compute_dtype
        self._e3nn_basis = False

        paths = [tuple(int(v) for v in p) for p in tp.paths]
        self.paths = paths
        self._in1_ls = sorted({l1 for l1, _l2, _l3 in paths})
        self._in2_ls = sorted({l2 for _l1, l2, _l3 in paths})
        self._out_ls = sorted({l3 for _l1, _l2, l3 in paths})
        self.path_counts_by_l = {
            l: int(tp.path_counts_by_l.get(l, 0)) for l in range(self.target_lmax + 1)
        }
        # --- exact per-path reference normalization ---
        # The frozen ICTC basis and Q bridge fix the positive component-normalized
        # natural-parity CG convention.  Both normalized CG tensors therefore have
        # identical Frobenius norm sqrt(2*l3+1); the only remaining factor in the
        # reference TP is its explicit element/path normalization.  Using that
        # stored scalar directly avoids rebuilding O(L^3) CG tables merely to
        # calibrate a scalar, which was especially costly at L6/L8.
        q_blocks_f64 = orthogonal_Q_blocks(self.lmax, dtype=torch.float64, device="cpu")
        if str(tp._normalization) != "component":
            raise RuntimeError(
                "ictd_tp_backend='cueq' currently requires component CG normalization"
            )
        inv_scales = [float(value) for value in tp._path_normalization_scales]
        self.register_buffer(
            "cueq_inv_path_scales",
            torch.tensor(inv_scales, dtype=self.compute_dtype).reshape(self.num_paths, 1),
            persistent=False,
        )

        # --- fixed orthogonal basis bridges (applied per l on the m axis) ---
        for l in self._in1_ls:
            self.register_buffer(
                f"cueq_q_in_{l}",
                q_blocks_f64[l].to(dtype=self.compute_dtype),
                persistent=False,
            )
        for l in self._in2_ls:
            self.register_buffer(
                f"cueq_q_edge_{l}",
                q_blocks_f64[l].to(dtype=self.compute_dtype),
                persistent=False,
            )
        for l in self._out_ls:
            self.register_buffer(
                f"cueq_q_out_t_{l}",
                q_blocks_f64[l].transpose(0, 1).contiguous().to(dtype=self.compute_dtype),
                persistent=False,
            )

        # --- cueq channelwise TP: external per-edge weights, e3nn mul_ir layout ---
        group = _cueq_o3_e3nn_group()

        def _parity(l: int) -> str:
            return "e" if int(l) % 2 == 0 else "o"

        irreps_in1 = cue.Irreps(
            group, " + ".join(f"{self.channels}x{l}{_parity(l)}" for l in self._in1_ls)
        )
        irreps_in2 = cue.Irreps(
            group, " + ".join(f"1x{l}{_parity(l)}" for l in self._in2_ls)
        )
        filter_out = [
            group(l=l, p=1 if int(l) % 2 == 0 else -1) for l in self._out_ls
        ]
        # Read the exact path-to-segment table from the same public descriptor
        # factory used by ChannelWiseTensorProduct.  Do not infer it from an
        # undocumented sorting convention: a same-l3 reordering would preserve
        # all shapes/counts while silently coupling every radial weight to the
        # wrong CG path.
        cueq_descriptor = cue.descriptors.channelwise_tensor_product(
            irreps_in1,
            irreps_in2,
            filter_out,
        )
        segmented_tp = cueq_descriptor.polynomial.operations[0][1]
        self.cueq_tp = cuet.ChannelWiseTensorProduct(
            irreps_in1,
            irreps_in2,
            filter_out,
            layout=cue.mul_ir,
            shared_weights=False,
            internal_weights=False,
            dtype=self.compute_dtype,
            math_dtype=self.compute_dtype,
        )
        # cuEquivariance's fallback/naive implementations register generated CG
        # constants as ordinary persistent buffers.  They are fully derivable from
        # the descriptor and must not leak into our backend-independent checkpoint
        # schema.  Mark all such implementation buffers non-persistent while
        # retaining normal device/dtype movement.
        for module in self.cueq_tp.modules():
            module._non_persistent_buffers_set.update(module._buffers)
        if str(self.cueq_tp.method) != "uniform_1d":
            raise RuntimeError(
                "ictd_tp_backend='cueq' requires a fused uniform_1d descriptor; "
                "the requested channel/path layout fell back to the naive backend"
            )
        if int(self.cueq_tp.weight_numel) != self.num_paths * self.channels:
            raise RuntimeError(
                "ictd_tp_backend='cueq': unexpected cueq weight layout "
                f"({int(self.cueq_tp.weight_numel)} != {self.num_paths * self.channels}); "
                "the conv-TP mirror assumes one weight per (path, channel)"
            )

        reference_path_by_triple = {path: idx for idx, path in enumerate(paths)}
        if len(reference_path_by_triple) != self.num_paths:
            raise RuntimeError("ictd_tp_backend='cueq': duplicate reference conv-TP paths")
        out_irreps = list(self.cueq_tp.irreps_out)
        weight_segment_to_reference = [-1] * self.num_paths
        output_segment_to_reference = [-1] * len(out_irreps)
        for row in segmented_tp.indices.tolist():
            weight_segment, in1_segment, in2_segment, out_segment = (
                int(value) for value in row
            )
            triple = (
                int(self._in1_ls[in1_segment]),
                int(self._in2_ls[in2_segment]),
                int(out_irreps[out_segment].ir.l),
            )
            reference_path = reference_path_by_triple.get(triple)
            if reference_path is None:
                raise RuntimeError(
                    "ictd_tp_backend='cueq': descriptor contains a path absent "
                    f"from the reference TP: {triple}"
                )
            if weight_segment_to_reference[weight_segment] not in (-1, reference_path):
                raise RuntimeError(
                    "ictd_tp_backend='cueq': one weight segment maps to multiple paths"
                )
            if output_segment_to_reference[out_segment] not in (-1, reference_path):
                raise RuntimeError(
                    "ictd_tp_backend='cueq': one output segment maps to multiple paths"
                )
            weight_segment_to_reference[weight_segment] = reference_path
            output_segment_to_reference[out_segment] = reference_path
        if sorted(weight_segment_to_reference) != list(range(self.num_paths)):
            raise RuntimeError(
                "ictd_tp_backend='cueq': descriptor weight segments do not form "
                "a bijection with reference paths"
            )
        if sorted(output_segment_to_reference) != list(range(self.num_paths)):
            raise RuntimeError(
                "ictd_tp_backend='cueq': descriptor output segments do not form "
                "a bijection with reference paths"
            )
        self.register_buffer(
            "cueq_path_perm",
            torch.tensor(weight_segment_to_reference, dtype=torch.long),
            persistent=False,
        )

        # Output segments are grouped by l3 but may use any within-degree order.
        # Record an explicit descriptor-derived permutation back to the reference
        # path_offset order for each degree.
        self._out_slices: Dict[int, tuple[int, int]] = {}
        offset = 0
        prev_l3: int | None = None
        seg_count_by_l: Dict[int, int] = {}
        output_paths_by_l: Dict[int, List[int]] = {}
        for out_segment, (mul, ir) in enumerate(self.cueq_tp.irreps_out):
            l3 = int(ir.l)
            if int(mul) != self.channels:
                raise RuntimeError(
                    "ictd_tp_backend='cueq': unexpected cueq output segment multiplicity "
                    f"({int(mul)} != {self.channels})"
                )
            if prev_l3 is not None and l3 < prev_l3:
                raise RuntimeError(
                    "ictd_tp_backend='cueq': cueq output segments are not sorted by l3; "
                    "the path-preserving layout mirror would be wrong"
                )
            prev_l3 = l3
            output_paths_by_l.setdefault(l3, []).append(
                int(output_segment_to_reference[out_segment])
            )
            start, end = offset, offset + int(mul) * int(ir.dim)
            seg_count_by_l[l3] = seg_count_by_l.get(l3, 0) + 1
            if l3 in self._out_slices:
                self._out_slices[l3] = (self._out_slices[l3][0], end)
            else:
                self._out_slices[l3] = (start, end)
            offset = end
        for l, count in seg_count_by_l.items():
            if count != self.path_counts_by_l.get(l, 0):
                raise RuntimeError(
                    f"ictd_tp_backend='cueq': cueq produced {count} paths for l={l}, "
                    f"reference tp has {self.path_counts_by_l.get(l, 0)}; path tables diverged"
                )
            current_paths = output_paths_by_l[l]
            desired_paths = sorted(current_paths, key=lambda path: int(tp.path_offset[path]))
            reorder = [current_paths.index(path) for path in desired_paths]
            self.register_buffer(
                f"cueq_out_reorder_{l}",
                torch.tensor(reorder, dtype=torch.long),
                persistent=False,
            )

    def fold_cg_to_e3nn(self, q_blocks: "List[torch.Tensor] | None" = None) -> None:
        """``angular_basis='e3nn'`` fold hook: features and edge harmonics already
        arrive in the e3nn convention, so the Q bridges become identity and are
        skipped. The per-path scale ``s_p`` is basis-independent and stays."""
        del q_blocks
        self._e3nn_basis = True

    def enable_e3nn_basis(self, q_blocks: "List[torch.Tensor] | None" = None) -> None:
        """Same flag for the checkpoint-reload path (``activate_e3nn_basis_from_folded_state_dict``)."""
        del q_blocks
        self._e3nn_basis = True

    def forward(
        self,
        x1: Dict[int, torch.Tensor],
        edge_src: torch.Tensor,
        edge_attrs: Dict[int, torch.Tensor],
        gates: torch.Tensor,
        path_weight: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """cueq conv-TP. ``x1`` is the NODE-level ICTC feature dict (gather by
        ``edge_src`` happens inside the kernel); ``edge_attrs`` the per-edge
        harmonic dict ``(E, 1, 2l+1)``; ``gates`` the radial-MLP output
        ``(E, num_paths*channels)``; ``path_weight`` the authoritative
        ``tp.weight`` ``(num_paths, channels)``. Returns the path-preserving
        edge blocks ``{l: (E, channels*paths_l, 2l+1)}`` in the ICTC convention
        (or e3nn when the model runs folded), exactly like the reference tp."""
        sample = next(iter(x1.values()))
        dtype = sample.dtype
        compute_dtype = self.compute_dtype
        num_edges = int(edge_src.shape[0])

        g = gates
        if g.shape[-1] == self.num_paths * self.channels:
            g = g.view(num_edges, self.num_paths, self.channels)
        elif g.shape[-2:] != (self.num_paths, self.channels):
            raise ValueError(
                f"Expected gates shape (E, {self.num_paths * self.channels}) or "
                f"(E, {self.num_paths}, {self.channels}), got {tuple(g.shape)}"
            )
        if g.dtype != compute_dtype:
            g = g.to(compute_dtype)
        w = path_weight.to(compute_dtype) * self.cueq_inv_path_scales.to(compute_dtype)
        u = (g * w).index_select(1, self.cueq_path_perm).reshape(
            num_edges, self.num_paths * self.channels
        )

        parts1 = []
        for l in self._in1_ls:
            blk = x1[l]
            if blk.dtype != compute_dtype:
                blk = blk.to(compute_dtype)
            if not self._e3nn_basis:
                blk = blk @ getattr(self, f"cueq_q_in_{l}").to(compute_dtype)
            parts1.append(blk.reshape(blk.shape[0], self.channels * (2 * l + 1)))
        x1f = torch.cat(parts1, dim=-1)

        parts2 = []
        for l in self._in2_ls:
            blk = edge_attrs[l].squeeze(-2)
            if blk.dtype != compute_dtype:
                blk = blk.to(compute_dtype)
            if not self._e3nn_basis:
                blk = blk @ getattr(self, f"cueq_q_edge_{l}").to(compute_dtype)
            parts2.append(blk)
        x2f = torch.cat(parts2, dim=-1)

        out = self.cueq_tp(
            x1f,
            x2f,
            u,
            indices_1=edge_src.to(torch.int32).contiguous(),
        )

        blocks: Dict[int, torch.Tensor] = {}
        for l in range(self.target_lmax + 1):
            count = self.path_counts_by_l.get(l, 0)
            m = 2 * l + 1
            if count == 0:
                blocks[l] = sample.new_zeros(num_edges, 0, m)
                continue
            start, end = self._out_slices[l]
            blk = out[:, start:end].reshape(num_edges, count, self.channels, m)
            blk = blk.index_select(
                1, getattr(self, f"cueq_out_reorder_{l}")
            ).reshape(num_edges, count * self.channels, m)
            if not self._e3nn_basis:
                blk = blk @ getattr(self, f"cueq_q_out_t_{l}").to(compute_dtype)
            blocks[l] = blk.to(dtype) if blk.dtype != dtype else blk
        return blocks


class ICTDResidualInteractionBlock(nn.Module):
    """
    ICTC-SO3 interaction block with MACE-like interface.

    Returns:
      - message: scatter-aggregated neighbor message
      - sc:      element-conditioned self-connection
    """

    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        input_lmax: int | None = None,
        target_lmax: int | None = None,
        sc_lmax: int | None = None,
        number_of_basis: int,
        num_elements: int,
        function_type: str = "gaussian",
        ictd_save_tp_mode: str = "fully-connected",
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        equivariant_post_linear: bool = False,
        use_self_connection: bool = True,
        self_connection_rank: int = 0,
        self_connection_rng_offset: int = 0,
        avg_num_neighbors: float | None = None,
        message_scale_init: list[float] | tuple[float, ...] | None = None,
        sc_scale_init: list[float] | tuple[float, ...] | None = None,
        conv_tp_scale_init: str = "none",
        freeze_conv_tp_weight: bool = False,
        interaction_init: str = "identity",
        use_rms_norm: bool = False,
        interaction_attn_heads: int = 0,
        radial_mlp_hidden: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.use_rms_norm = bool(use_rms_norm)
        self.input_lmax = self.lmax if input_lmax is None else int(input_lmax)
        self.target_lmax = self.lmax if target_lmax is None else int(target_lmax)
        self.sc_lmax = self.input_lmax if sc_lmax is None else int(sc_lmax)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)
        self.equivariant_post_linear = bool(equivariant_post_linear)
        self.use_self_connection = bool(use_self_connection)
        self.self_connection_rank = int(self_connection_rank)
        if self.self_connection_rank < 0:
            raise ValueError(
                f"self_connection_rank must be non-negative, got {self_connection_rank}"
            )
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        self.conv_tp_scale_init = str(conv_tp_scale_init)
        if self.conv_tp_scale_init not in {"none", "e3nn"}:
            raise ValueError(f"conv_tp_scale_init must be 'none' or 'e3nn', got {conv_tp_scale_init!r}")
        self.freeze_conv_tp_weight = bool(freeze_conv_tp_weight)
        self.interaction_init = str(interaction_init)
        if self.interaction_init not in {"identity", "mace-random"}:
            raise ValueError(f"interaction_init must be 'identity' or 'mace-random', got {interaction_init!r}")
        allowed_paths = _tp_allowed_paths_from_target_lmax(
            lmax_in1=self.input_lmax,
            lmax_in2=self.lmax,
            lmax_target=self.target_lmax,
        )
        self.linear_up = EquivariantChannelLinearSO3(self.channels, self.input_lmax, bias=False)
        self.tp = EdgeWeightedPathPreservingTensorProduct(
            channels=self.channels,
            lmax=self.lmax,
            allowed_paths=allowed_paths,
            path_policy=ictd_tp_path_policy,
            max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
        )
        _init_path_tp_weight_to_one_(self.tp)
        if self.conv_tp_scale_init == "e3nn":
            scales = _mace_like_conv_tp_path_scales(
                self.tp,
                channels=self.channels,
                input_lmax=self.input_lmax,
                edge_lmax=self.lmax,
            )
            with torch.no_grad():
                for path_idx, scale in enumerate(scales):
                    self.tp.weight[path_idx].fill_(float(scale))
        if self.freeze_conv_tp_weight:
            self.tp.weight.requires_grad_(False)
        # conv-TP compute backend. "pytorch" (default) is the reference projector
        # path; "auto"/"cuda_ext" are accepted for interface consistency with the
        # other ICTC TPs but this block's message-passing TP has no cuda_ext kernel,
        # so they keep the reference behavior; "cueq" mirrors the TP with a fused
        # cuEquivariance kernel (CUDA only, CPU forward falls back to pytorch).
        self.ictd_tp_backend = str(ictd_tp_backend).strip().lower()
        if self.ictd_tp_backend not in {"pytorch", "auto", "cuda_ext", "cueq"}:
            raise ValueError(
                f"ictd_tp_backend must be 'pytorch', 'auto', 'cuda_ext', or 'cueq', "
                f"got {ictd_tp_backend!r}"
            )
        self.tp_cueq = (
            CueqConvTensorProductSO3(
                tp=self.tp,
                input_lmax=self.input_lmax,
                target_lmax=self.target_lmax,
            )
            if self.ictd_tp_backend == "cueq"
            else None
        )
        self.radial_mlp_hidden = _radial_mlp_widths(radial_mlp_hidden)
        radial_sizes = (self.number_of_basis, *self.radial_mlp_hidden, self.tp.num_paths * self.channels)
        radial_layers = []
        for index, (size_in, size_out) in enumerate(zip(radial_sizes, radial_sizes[1:])):
            radial_layers.append(nn.Linear(size_in, size_out, bias=False))
            if index < len(radial_sizes) - 2:
                radial_layers.append(_Normalize2MomSiLU())
        self.fc = nn.Sequential(*radial_layers)
        for module in self.fc:
            if isinstance(module, nn.Linear):
                _init_mace_style_linear_(module)
        self.message_linear = PathPreservingLinearSO3(
            {
                l: self.channels * int(self.tp.path_counts_by_l.get(l, 0))
                for l in range(self.target_lmax + 1)
            },
            out_channels=self.channels,
            lmax=self.target_lmax,
        )
        if self.interaction_init == "mace-random":
            _init_so3_linear_mace_style_(self.linear_up)
        else:
            _init_so3_linear_identity_(self.linear_up)
        self.message_selector = (
            ElementConditionedLinearSO3(
                num_elements=num_elements,
                channels=self.channels,
                lmax=self.target_lmax,
                bias=False,
            )
            if not self.use_self_connection
            else None
        )
        if self.interaction_init == "mace-random":
            _init_element_conditioned_mace_style_(self.message_selector)
        else:
            _init_element_conditioned_identity_(self.message_selector)
        if not self.use_self_connection:
            self.self_connection = None
        else:
            # The full and low-rank parameterizations allocate different numbers
            # of random tensors.  Keep those draws local so a matched ablation has
            # byte-identical initialization for every parameter outside SC.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(
                    torch.initial_seed()
                    + 104729 * (int(self_connection_rng_offset) + 1)
                )
                if self.self_connection_rank > 0:
                    self.self_connection = LowRankElementConditionedLinearSO3(
                        num_elements=num_elements,
                        channels=self.channels,
                        lmax=self.sc_lmax,
                        rank=self.self_connection_rank,
                        bias=False,
                    )
                else:
                    self.self_connection = ElementConditionedLinearSO3(
                        num_elements=num_elements,
                        channels=self.channels,
                        lmax=self.sc_lmax,
                        bias=False,
                    )
                if self.interaction_init == "mace-random":
                    _init_element_conditioned_mace_style_(self.self_connection)
                else:
                    _init_element_conditioned_identity_(self.self_connection)
        self.message_norm = (
            SO3BlockRMSNorm(self.channels, self.target_lmax) if self.use_rms_norm else nn.Identity()
        )
        self.sc_norm = (
            SO3BlockRMSNorm(self.channels, self.sc_lmax) if (self.use_rms_norm and self.self_connection is not None) else nn.Identity()
        )
        self.message_output_scale = (
            PerLScaleSO3(self.channels, self.target_lmax, message_scale_init)
            if message_scale_init is not None
            else nn.Identity()
        )
        self.sc_output_scale = (
            PerLScaleSO3(self.channels, self.sc_lmax, sc_scale_init)
            if sc_scale_init is not None
            else nn.Identity()
        )
        # --- Optional equivariant neighbor-attention scatter (DPA-4/SeZM-style) ---
        # heads=0 -> plain envelope scatter-sum (byte-identical to before). heads>0 ->
        # an invariant attention weight per (edge, head): logit = (q[dst].k[src])/sqrt(d)
        # + radial_bias, computed from the l=0 node scalars (q=dst, k=src); the weights
        # are env^2-gated zeta-softmax over each dst's incoming edges (smooth at rcut, so
        # forces stay continuous). The scalar alpha is shared across the 2l+1 m-components
        # of every l (=> equivariance preserved) and across the head's channels.
        self.interaction_attn_heads = int(interaction_attn_heads)
        self._fused_selector_message_enabled = False
        self._fused_selector_message_has_bias = False
        if self.interaction_attn_heads > 0:
            if self.channels % self.interaction_attn_heads != 0:
                raise ValueError(
                    f"channels ({self.channels}) must be divisible by interaction_attn_heads ({self.interaction_attn_heads})"
                )
            self.attn_head_dim = self.channels // self.interaction_attn_heads
            self.attn_qk_norm = nn.LayerNorm(self.channels)
            self.attn_q_proj = nn.Linear(self.channels, self.channels, bias=False)
            self.attn_k_proj = nn.Linear(self.channels, self.channels, bias=False)
            self.attn_radial_bias = nn.Linear(self.number_of_basis, self.interaction_attn_heads, bias=False)
            self.attn_z_bias_raw = nn.Parameter(torch.zeros(self.interaction_attn_heads))
            # DPA-4-style gentle start: a learnable per-(head, head_channel) weight on the
            # q*k product (replaces the fixed 1/sqrt(d) scale), init tiny so the content
            # logit ~= 0 at init; together with zero-init radial_bias the whole logit starts
            # at 0 => alpha = pure env^2-weighted average (no content "neighbor picking" yet),
            # and the content/distance attention ramps in gradually as these weights grow.
            self.attn_logit_w = nn.Parameter(
                torch.empty(self.interaction_attn_heads, self.attn_head_dim)
            )
            nn.init.normal_(self.attn_logit_w, mean=0.0, std=0.01)
            nn.init.zeros_(self.attn_radial_bias.weight)
        else:
            self.attn_head_dim = 0
            self.attn_qk_norm = None
            self.attn_q_proj = None
            self.attn_k_proj = None
            self.attn_radial_bias = None
            self.attn_z_bias_raw = None
            self.attn_logit_w = None

    def enable_eval_fused_selector_message(self) -> bool:
        """Precompose message_linear and element selector for eval/AOTI inference.

        The first MACE residual block applies, per l:
            scatter -> message_linear -> /avg_num_neighbors -> element selector -> output scale.
        When the selector is present and all these maps are linear, precompose the
        fixed weights into one element-conditioned linear map. This changes only
        fp32 accumulation order and is disabled for training by default.
        """
        if self.interaction_attn_heads > 0 or self.message_selector is None:
            self._fused_selector_message_enabled = False
            return False
        if self.avg_num_neighbors is None:
            self._fused_selector_message_enabled = False
            return False
        if not isinstance(self.message_norm, nn.Identity):
            self._fused_selector_message_enabled = False
            return False

        if isinstance(self.message_output_scale, nn.Identity):
            scales = [1.0 for _ in range(self.target_lmax + 1)]
        elif isinstance(self.message_output_scale, PerLScaleSO3):
            scales = [
                float(v)
                for v in self.message_output_scale.log_scale.detach().exp().cpu().tolist()
            ]
        else:
            self._fused_selector_message_enabled = False
            return False

        selector_bias = getattr(self.message_selector, "bias", None)
        self._fused_selector_message_has_bias = selector_bias is not None
        avg = float(self.avg_num_neighbors)
        with torch.no_grad():
            for l in range(self.target_lmax + 1):
                msg_w = self.message_linear.weights[str(l)].detach()
                sel_w = self.message_selector.weights[str(l)].detach()
                fused = torch.einsum("eoj,jc->eoc", sel_w, msg_w) * (float(scales[l]) / avg)
                name = f"_fused_selector_message_weight_{l}"
                if name in self._buffers:
                    self._buffers[name] = fused.contiguous()
                else:
                    self.register_buffer(name, fused.contiguous(), persistent=False)
                if selector_bias is not None:
                    bias = selector_bias[str(l)].detach() * float(scales[l])
                    bias_name = f"_fused_selector_message_bias_{l}"
                    if bias_name in self._buffers:
                        self._buffers[bias_name] = bias.contiguous()
                    else:
                        self.register_buffer(bias_name, bias.contiguous(), persistent=False)
        self._fused_selector_message_enabled = True
        return True

    def _attention_alpha(
        self,
        node_feats_l0: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_env: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Invariant env^2-gated zeta-softmax attention weight, shape (E, H).

        node_feats_l0: (N, channels) l=0 scalar block. edge_env: (E,) cutoff envelope.
        Destination-wise (over each dst's incoming edges):
            alpha_ij = env_ij^2 exp(logit_ij) / (zeta_h + sum_k env_ik^2 exp(logit_ik))
        with logit = (q[dst].k[src])/sqrt(d) + radial_bias and zeta = softplus(z_bias_raw).
        env^2 -> 0 at rcut + the unnormalized (zeta) denominator keep forces smooth."""
        H = self.interaction_attn_heads
        d = self.attn_head_dim
        qk = self.attn_qk_norm(node_feats_l0)
        q = self.attn_q_proj(qk).reshape(-1, H, d)
        k = self.attn_k_proj(qk).reshape(-1, H, d)
        # learnable per-(head, channel) weight on q*k (replaces fixed 1/sqrt(d)); init ~0
        # so the content logit starts at 0 and ramps in (DPA-4-style gentle start).
        logit = (q[edge_dst] * k[edge_src] * self.attn_logit_w).sum(-1)  # (E, H)
        logit = logit + self.attn_radial_bias(edge_feats)  # (E, H); radial_bias zero-init
        env2 = edge_env.reshape(-1, 1).to(dtype=logit.dtype).clamp_min(0.0).square()  # (E, 1)
        # group max over dst's edges, floored at 0 (the zeta term sits at logit 0) for
        # overflow-free exp shifts: exp(logit-gmax)<=exp(0)=1 and exp(-gmax)<=1.
        # per-dst max over incoming logits (softmax stability shift). Use torch-native
        # scatter_reduce(amax) which returns ONLY values -- torch_scatter's scatter_max returns a
        # non-differentiable argmax that breaks compiled-autograd (non_differentiable assert) AND
        # CUDA-graph capture. Numerically identical (same max). clamp_min(0): the zeta term sits at 0.
        gmax = logit.new_zeros(num_nodes, H).scatter_reduce_(
            0, edge_dst.unsqueeze(-1).expand(-1, H), logit, reduce="amax", include_self=False
        ).clamp_min(0.0)  # (N, H)
        ex = env2 * torch.exp(logit - gmax[edge_dst])  # (E, H)
        denom = scatter(ex, edge_dst, dim=0, dim_size=num_nodes, reduce="sum")  # (N, H)
        zeta = F.softplus(self.attn_z_bias_raw).reshape(1, H).to(dtype=logit.dtype)
        denom = denom + zeta * torch.exp(-gmax)  # (N, H)
        return ex / (denom[edge_dst] + 1e-20)  # (E, H)

    def forward(
        self,
        *,
        node_attrs: torch.Tensor | None,
        node_feats: torch.Tensor,
        edge_attrs: Dict[int, torch.Tensor],
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        edge_env: torch.Tensor | None = None,
        node_type_idx: torch.Tensor | None = None,
        edge_block_callback: callable | None = None,
        sync_after_scatter: callable | None = None,
        receiver_rows: torch.Tensor | None = None,
        compact_message_output: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
    ):
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        num_nodes = node_feats.size(0)

        if receiver_rows is not None and (self.interaction_attn_heads > 0
                or edge_block_callback is not None or sync_after_scatter is not None):
            raise ValueError("Receiver-row projection requires attention-free local inference")

        node_feats_for_sc = node_feats
        node_feats = self.linear_up(node_feats)
        x1 = _split_irreps(node_feats, self.channels, self.input_lmax)
        fused_message_blocks = None
        selected_message_output = False
        deployment_message=getattr(self,"_deployment_message",None)
        if deployment_message is not None:
            fused_message_blocks=deployment_message(x1,edge_attrs,self.fc(edge_feats),edge_src,edge_dst,edge_mask)
        if (getattr(self,"_streamed_gates",False) and getattr(self,"_edge_fusion","reference")=="triton_remat"
                and self.tp_cueq is None and self.interaction_attn_heads==0 and edge_block_callback is None and fused_message_blocks is None):
            from .streamed_gate_message import streamed_gate_message
            if receiver_rows is not None and compact_message_output:
                receiver_map = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_dst.device)
                receiver_map[receiver_rows] = torch.arange(receiver_rows.numel(), device=edge_dst.device)
                fused_message_blocks=streamed_gate_message(self.tp,x1,edge_attrs,edge_feats,self.fc,
                    edge_src,receiver_map[edge_dst],edge_mask,output_nodes=receiver_rows.numel())
                selected_message_output = fused_message_blocks is not None
            if fused_message_blocks is None:
                fused_message_blocks=streamed_gate_message(self.tp,x1,edge_attrs,edge_feats,self.fc,edge_src,edge_dst,edge_mask)
        if fused_message_blocks is None:
            from .radial_checkpoint import radial_apply
            gates = radial_apply(self.fc,edge_feats,checkpoint=getattr(self.tp,'_radial_checkpoint',False),
                gemm_policy=getattr(self.tp,'_training_gate_gemm_policy','torch'),
                chunk_rows=getattr(self.tp,'_radial_chunk_rows',0))
        else:
            gates = None
        if (getattr(self, "_edge_fusion", "reference") in ("triton", "cuda", "cuda_projection", "triton_projection", "triton_csr", "triton_remat")
                and self.tp_cueq is None and self.interaction_attn_heads == 0
                and edge_block_callback is None and fused_message_blocks is None):
            from .fused_message import aggregate_message
            fused_message_blocks = aggregate_message(self.tp,x1,edge_attrs,gates,edge_src,edge_dst,edge_mask,backend=self._edge_fusion)
        if fused_message_blocks is not None:
            edge_blocks = {}
        elif self.tp_cueq is not None and node_feats.is_cuda:
            # cueq fused kernel gathers node feats internally and emits the same
            # path-preserving ICTC edge blocks; CPU falls back to the reference TP.
            edge_blocks = self.tp_cueq(x1, edge_src, edge_attrs, gates, self.tp.weight)
        else:
            x1e = {l: x1[l][edge_src] for l in range(self.input_lmax + 1)}
            edge_blocks = self.tp(x1e, edge_attrs, gates)
        if edge_mask is not None and fused_message_blocks is None:
            mask = edge_mask.to(dtype=node_feats.dtype)
            edge_blocks = {l: block * mask.unsqueeze(-1) for l, block in edge_blocks.items()}
        selector_message_fused = False
        if self.interaction_attn_heads > 0:
            if edge_env is None:
                raise ValueError("interaction_attn_heads > 0 requires edge_env to be passed to forward()")
            H = self.interaction_attn_heads
            alpha = self._attention_alpha(
                x1[0].squeeze(-1), edge_feats, edge_src, edge_dst, edge_env, num_nodes
            )  # (E, H) invariant env^2-gated zeta-softmax weights
            a = alpha.reshape(-1, H, 1, 1)
            attended_edge_blocks: Dict[int, torch.Tensor] = {}
            for l in range(self.target_lmax + 1):
                eb = edge_blocks[l]  # (E, C_l, 2l+1); C_l = channels * num_paths_l, H | channels => H | C_l
                e_n, c_l, m = eb.shape
                eb = (eb.reshape(e_n, H, c_l // H, m) * a).reshape(e_n, c_l, m)
                attended_edge_blocks[l] = eb
            edge_blocks = attended_edge_blocks
        auxiliary_edge_blocks: Dict[int, torch.Tensor] | None = None
        if edge_block_callback is not None:
            callback_output = edge_block_callback(
                edge_blocks=edge_blocks,
                edge_attrs=edge_attrs,
                node_blocks=x1,
                edge_feats=edge_feats,
                edge_index=edge_index,
                edge_env=edge_env,
                interaction=self,
            )
            if isinstance(callback_output, tuple):
                if len(callback_output) != 2:
                    raise ValueError(
                        "edge_block_callback tuple output must contain "
                        "(neutral_blocks, auxiliary_blocks)"
                    )
                edge_blocks, auxiliary_edge_blocks = callback_output
            else:
                edge_blocks = callback_output
            expected_degrees = set(range(self.target_lmax + 1))
            if set(edge_blocks) != expected_degrees:
                raise ValueError(
                    "edge_block_callback must return every target degree; "
                    f"expected {sorted(expected_degrees)}, got {sorted(edge_blocks)}"
                )
            if auxiliary_edge_blocks is not None:
                if set(auxiliary_edge_blocks) != expected_degrees:
                    raise ValueError(
                        "auxiliary edge blocks must return every target degree; "
                        f"expected {sorted(expected_degrees)}, got "
                        f"{sorted(auxiliary_edge_blocks)}"
                    )
                stream_count = None
                for l in range(self.target_lmax + 1):
                    neutral_block = edge_blocks[l]
                    auxiliary_block = auxiliary_edge_blocks[l]
                    if auxiliary_block.ndim != neutral_block.ndim + 1:
                        raise ValueError(
                            "auxiliary edge blocks require one leading stream "
                            f"axis; l={l}, neutral={tuple(neutral_block.shape)}, "
                            f"auxiliary={tuple(auxiliary_block.shape)}"
                        )
                    if (
                        auxiliary_block.shape[0] != neutral_block.shape[0]
                        or auxiliary_block.shape[-2:]
                        != neutral_block.shape[-2:]
                    ):
                        raise ValueError(
                            "auxiliary and neutral edge block shapes disagree; "
                            f"l={l}, neutral={tuple(neutral_block.shape)}, "
                            f"auxiliary={tuple(auxiliary_block.shape)}"
                        )
                    current_stream_count = int(auxiliary_block.shape[1])
                    if stream_count is None:
                        stream_count = current_stream_count
                    elif current_stream_count != stream_count:
                        raise ValueError(
                            "all auxiliary edge degrees must use the same "
                            "number of streams"
                        )
                if stream_count is None or stream_count <= 0:
                    raise ValueError(
                        "auxiliary edge blocks must contain at least one stream"
                    )
                if self.interaction_attn_heads > 0:
                    raise ValueError(
                        "auxiliary edge streams are not yet compatible with "
                        "interaction attention"
                    )
        auxiliary_message = None
        if self.interaction_attn_heads > 0:
            message_blocks = {
                l: scatter(
                    edge_blocks[l],
                    edge_dst,
                    dim=0,
                    dim_size=num_nodes,
                    reduce="sum",
                )
                for l in range(self.target_lmax + 1)
            }
            # zeta-softmax already normalizes (sum of alpha <= 1); do NOT divide by
            # avg_num_neighbors (that would double-normalize the attention path).
            message = self.message_linear(message_blocks)
        else:
            if fused_message_blocks is not None:
                message_blocks = fused_message_blocks
            elif auxiliary_edge_blocks is None:
                message_blocks = {
                    l: scatter(edge_blocks[l], edge_dst, dim=0, dim_size=num_nodes, reduce="sum")
                    for l in range(self.target_lmax + 1)
                }
            else:
                # The neutral and auxiliary streams have identical SO(3)
                # geometry and channel maps.  Aggregate them in one scatter
                # and apply the path projection as a batched linear map.  This
                # is the same fused ordering used by CHORUS for its neutral,
                # charged-real and charged-imaginary streams.
                message_blocks = {
                    l: scatter(
                        torch.cat(
                            (
                                edge_blocks[l].unsqueeze(1),
                                auxiliary_edge_blocks[l],
                            ),
                            dim=1,
                        ),
                        edge_dst,
                        dim=0,
                        dim_size=num_nodes,
                        reduce="sum",
                    )
                    for l in range(self.target_lmax + 1)
                }
            if receiver_rows is not None:
                # Only selected receivers feed the next product/readout. Slice
                # before projection and self-connection GEMMs, keeping the TP
                # source rows and their coordinate adjoints intact.
                if not selected_message_output:
                    message_blocks = {l: block.index_select(0, receiver_rows)
                                      for l, block in message_blocks.items()}
                node_feats_for_sc = node_feats_for_sc.index_select(0, receiver_rows)
                node_attrs = None if node_attrs is None else node_attrs.index_select(0, receiver_rows)
                node_type_idx = None if node_type_idx is None else node_type_idx.index_select(0, receiver_rows)
            if (
                getattr(self, "_fused_selector_message_enabled", False)
                and sync_after_scatter is None
                and auxiliary_edge_blocks is None
            ):
                type_idx = (
                    node_type_idx.to(device=node_feats.device, dtype=torch.long)
                    if node_type_idx is not None
                    else None
                )
                if type_idx is None:
                    if node_attrs is None:
                        raise ValueError("node_attrs is required when node_type_idx is not provided")
                    attrs = node_attrs.to(dtype=node_feats.dtype)
                else:
                    attrs = None
                out_blocks: Dict[int, torch.Tensor] = {}
                for l in range(self.target_lmax + 1):
                    weight = getattr(self, f"_fused_selector_message_weight_{l}").to(
                        dtype=message_blocks[l].dtype,
                        device=message_blocks[l].device,
                    )
                    if type_idx is None:
                        if attrs is None:
                            raise ValueError("node_attrs is required when node_type_idx is not provided")
                        mixed_weight = torch.einsum("ne,eoc->noc", attrs, weight)
                        out_block = torch.einsum("noc,ncm->nom", mixed_weight, message_blocks[l])
                    else:
                        mixed_weight = weight.index_select(0, type_idx)
                        out_block = torch.einsum("noc,ncm->nom", mixed_weight, message_blocks[l])
                    if self._fused_selector_message_has_bias:
                        bias = getattr(self, f"_fused_selector_message_bias_{l}").to(
                            dtype=out_block.dtype,
                            device=out_block.device,
                        )
                        if type_idx is None:
                            if attrs is None:
                                raise ValueError("node_attrs is required when node_type_idx is not provided")
                            mixed_bias = torch.einsum("ne,eo->no", attrs, bias)
                        else:
                            mixed_bias = bias.index_select(0, type_idx)
                        out_block = out_block + mixed_bias.unsqueeze(-1)
                    out_blocks[l] = out_block
                message = _merge_irreps(out_blocks, self.channels, self.target_lmax)
                selector_message_fused = True
            else:
                if self.avg_num_neighbors is None:
                    if edge_mask is not None:
                        avg_num_neighbors = float(edge_mask.detach().sum().item()) / float(max(num_nodes, 1))
                    else:
                        avg_num_neighbors = float(edge_src.numel()) / float(max(num_nodes, 1))
                else:
                    avg_num_neighbors = self.avg_num_neighbors
                projected_message = self.message_linear(message_blocks) / max(
                    avg_num_neighbors, 1e-8
                )
                if auxiliary_edge_blocks is None:
                    message = projected_message
                else:
                    message = projected_message[:, 0]
                    auxiliary_message = projected_message[:, 1:]
        if sync_after_scatter is not None:
            if auxiliary_message is None:
                message = sync_after_scatter(message)
            else:
                synchronized = sync_after_scatter(
                    torch.cat((message.unsqueeze(1), auxiliary_message), dim=1)
                )
                message = synchronized[:, 0]
                auxiliary_message = synchronized[:, 1:]
        if not self.use_self_connection and not selector_message_fused:
            if node_type_idx is not None:
                message = self.message_selector.forward_type_idx(
                    message, node_type_idx
                )
            else:
                if node_attrs is None:
                    raise ValueError(
                        "node_attrs is required for the unfused message selector path"
                    )
                message = self.message_selector(message, node_attrs)
        if not selector_message_fused:
            message = self.message_norm(message)
            message = self.message_output_scale(message)
            if auxiliary_message is not None:
                # The charged transport deliberately does not use the neutral
                # RMS normalization, matching the existing O2U1 wrapper and
                # CHORUS persistent stream.  The learned per-l output scale is
                # shared with the neutral interaction and supports arbitrary
                # leading stream dimensions.
                auxiliary_message = self.message_output_scale(
                    auxiliary_message
                )
        sc = None
        if self.self_connection is not None:
            if self.sc_lmax == self.input_lmax:
                sc_input = node_feats_for_sc
            elif self.sc_lmax == 0:
                sc_input = _split_irreps(node_feats_for_sc, self.channels, self.input_lmax)[0].reshape(
                    node_feats_for_sc.shape[0], self.channels
                )
            else:
                raise ValueError(
                    f"Unsupported ICTC self-connection projection input_lmax={self.input_lmax}, sc_lmax={self.sc_lmax}"
                )
            if node_type_idx is not None:
                sc = self.self_connection.forward_type_idx(sc_input, node_type_idx)
            else:
                if node_attrs is None:
                    raise ValueError("node_attrs is required for the training self-connection path")
                sc = self.self_connection(sc_input, node_attrs)
            sc = self.sc_norm(sc)
            sc = self.sc_output_scale(sc)
        if auxiliary_message is not None:
            return message, sc, auxiliary_message
        return message, sc


class PureCartesianICTDFix(nn.Module):
    """
    ICTC-SO3 model organized with a MACE-style backbone:

      h_t -> interaction_t(node_attrs, h_t, edge_*) = (m_t, sc_t)
          -> product_t(m_t, sc_t, node_attrs) = h_{t+1}
          -> layer_readout_t(h_{t+1})

    Optional route:
      - baseline: sum(layerwise readouts)
      - fusion:   sum(layerwise readouts) + E_fusion(h_1, ..., h_N)
    """

    def __init__(
        self,
        max_embed_radius: float,
        main_max_radius: float,
        main_number_of_basis: int,
        hidden_dim_conv: int,
        hidden_dim_sh: int,
        hidden_dim: int,
        channel_in2: int = 32,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        atomic_numbers: list[int] | tuple[int, ...] | None = None,
        output_size: int = 8,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers: int = 1,
        num_interaction: int = 2,
        device=None,
        function_type_main: str = "gaussian",
        lmax: int = 2,
        ictd_Lmax: int = 6,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        max_rank_other: int = 1,
        k_policy: str = "k0",
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        product5_muls_by_l: dict[int, int] | None = None,
        invariant_channels: int = 32,
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
        long_range_max_multipole_l: int = 0,
        long_range_multipole_gate_init: float = 0.1,
        long_range_dispersion: bool = False,
        long_range_dispersion_mode: str | None = None,
        dispersion_cutoff: float = 8.0,
        dispersion_max_num_neighbors: int | None = None,
        dispersion_neighbor_method: str = "auto",
        dispersion_bruteforce_threshold: int = 1024,
        dispersion_allow_large_bruteforce_fallback: bool = False,
        dispersion_slq_num_probes: int = 8,
        dispersion_slq_lanczos_steps: int = 16,
        mbd_operator_backend: str = "edge_sparse",
        mbd_pme_mesh_size: int = 32,
        mbd_pme_assignment: str = "pcs",
        mbd_pme_k_norm_floor: float = 1.0e-6,
        mbd_pme_assignment_window_floor: float = 1.0e-6,
        mbd_pme_ewald_alpha_prefactor: float = 5.0,
        mbd_anisotropic_polarizability: bool = False,
        mbd_learnable_energy_scale: bool = True,
        mbd_alpha_floor: float = 1.0e-4,
        dispersion_min_cutoff: float = 0.0,
        dispersion_switch_width: float = 0.5,
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
        equivariant_post_linear: bool = False,
        ictd_save_tp_mode: str = "fully-connected",
        ictd_fix_route: str = "baseline",
        ictd_fix_contraction_combine: str = "softmax",
        ictd_fix_product_backend: str = "ictd-bridge-u",
        ictd_fix_use_reduced_cg: bool = False,
        ictd_fix_first_layer_self_connection: bool = False,
        ictd_fix_self_connection_rank: int = 0,
        ictd_fix_conv_tp_scale_init: str = "none",
        ictd_fix_freeze_conv_tp_weight: bool = False,
        ictd_fix_interaction_init: str = "identity",
        ictd_fix_edge_lmax: int | None = None,
        angular_basis: str = "ictd",
        ictd_fix_interaction_scale: str = "none",
        ictd_fix_fusion_scale_init: float = 0.1,
        ictd_fix_fusion_heads: int = 1,
        ictd_fix_fusion_head_weight_mode: str = "softmax",
        ictd_fix_fusion_input_scale_init: float = 1.0,
        ictd_fix_fusion_input_scale_trainable: bool = False,
        ictd_fix_fusion_depth_attention: bool = False,
        ictd_fix_gmix_gate_init: float = 1.0,
        ictd_fix_gmix_gate_trainable: bool = False,
        ictd_fix_gmix_block_rmsnorm: bool = False,
        ictd_fix_gmix_block_rmsnorm_gamma_init: float = 1.0,
        ictd_fix_readout_head_scale_init: float = 1.0,
        ictd_fix_readout_head_scale_trainable: bool = False,
        ictd_fix_fusion_readout_mixed_channels: bool = True,
        ictd_fix_fusion_pre_product_norm: bool = True,
        ictd_fix_interaction_rms_norm: bool = False,
        radial_sqrt_num_basis: bool = False,
        ictd_fix_interaction_attn_heads: int = 0,
        ictd_fix_gmix_energy_readout: bool = True,
        ictd_fix_gmix_readout_scale_init: float | None = None,
        ictd_fix_gmix_readout_output_init_std: float = 0.003,
        ictd_fix_gmix_output_lmax: int | None = None,
        ictd_fix_readout_hidden_channels: int = 16,
        ictd_fix_layer_readout_output_init_std: float = 0.003,
        polynomial_cutoff_p: int | None = 6,
        save_contraction_order: int = 3,
        save_multiple_mix_channels: int | None = None,
        avg_num_neighbors: float | None = None,
        energy_output_scale: float = 1.0,
        energy_output_scale_enabled: bool = False,
        energy_output_shift: float = 0.0,
        energy_output_shift_enabled: bool = False,
        radial_mlp_hidden: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        if embed_size is None:
            embed_size = [128, 128, 128]
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64]
        self.radial_mlp_hidden = _radial_mlp_widths(radial_mlp_hidden)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if int(num_interaction) < 1:
            raise ValueError(f"num_interaction must be >= 1, got {num_interaction}")
        if ictd_fix_route != "baseline":
            raise ValueError(
                f"this baseline-only MACE-ICTC build supports ictd_fix_route='baseline' only, "
                f"got {ictd_fix_route!r}"
            )
        if ictd_fix_contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(
                f"ictd_fix_contraction_combine must be 'softmax', 'free', or 'path-free', got {ictd_fix_contraction_combine!r}"
            )
        self.angular_basis = str(angular_basis)
        if self.angular_basis not in {"ictd", "e3nn"}:
            raise ValueError(f"angular_basis must be 'ictd' or 'e3nn', got {self.angular_basis!r}")
        # Lazily applied on first forward (folds the fixed angular operators into the e3nn basis).
        self._e3nn_folded = False
        self._e3nn_q_blocks: List[torch.Tensor] = []
        requested_product_backend = str(ictd_fix_product_backend)
        if requested_product_backend == "ictd-mace-u":
            requested_product_backend = "ictd-bridge-u"
        if requested_product_backend not in {"ictd", "native-mace", "ictd-bridge-u", "ictd-pure-u", "cueq"}:
            raise ValueError(
                "ictd_fix_product_backend must be 'ictd', 'native-mace', 'ictd-bridge-u', "
                f"'ictd-mace-u' alias, 'ictd-pure-u', or 'cueq', got {ictd_fix_product_backend!r}"
            )
        if ictd_fix_interaction_scale not in {"none", "mace-rms"}:
            raise ValueError(
                f"ictd_fix_interaction_scale must be 'none' or 'mace-rms', got {ictd_fix_interaction_scale!r}"
            )
        if ictd_fix_conv_tp_scale_init not in {"none", "e3nn"}:
            raise ValueError(
                "ictd_fix_conv_tp_scale_init must be 'none' or 'e3nn', "
                f"got {ictd_fix_conv_tp_scale_init!r}"
            )
        # conv-TP compute backend of the interaction blocks (forwarded as-is; the
        # interaction validates the value). "cueq" mirrors the message-passing TP
        # with a fused cuEquivariance kernel and only affects the compute path, not
        # the parameterization or the function.
        self.ictd_tp_backend = str(ictd_tp_backend).strip().lower()
        if self.ictd_tp_backend == "cueq" and requested_product_backend == "ictd":
            raise ValueError(
                "ictd_tp_backend='cueq' accelerates the interaction conv-TP only; the 'ictd' "
                "product backend runs its own TP engine that does not accept 'cueq'. Use product "
                "backend 'native-mace', 'ictd-bridge-u', 'ictd-pure-u', or 'cueq'."
            )
        if ictd_fix_interaction_init not in {"identity", "mace-random"}:
            raise ValueError(
                "ictd_fix_interaction_init must be 'identity' or 'mace-random', "
                f"got {ictd_fix_interaction_init!r}"
            )
        if ictd_fix_fusion_head_weight_mode not in {"softmax", "free"}:
            raise ValueError(
                "ictd_fix_fusion_head_weight_mode must be 'softmax' or 'free', "
                f"got {ictd_fix_fusion_head_weight_mode!r}"
            )
        if feature_spectral_mode != "none":
            raise NotImplementedError("pure-cartesian-ictd-fix currently supports only feature_spectral_mode=none (long_range is wired)")

        self.channels = int(hidden_dim_conv)
        self.lmax = int(lmax)
        self.ictd_fix_edge_lmax = self.lmax if ictd_fix_edge_lmax is None else int(ictd_fix_edge_lmax)
        if self.ictd_fix_edge_lmax < 0:
            raise ValueError(f"ictd_fix_edge_lmax must be >= 0, got {self.ictd_fix_edge_lmax}")
        if self.ictd_fix_edge_lmax < self.lmax:
            raise NotImplementedError(
                "MACE-ICTC currently requires ictd_fix_edge_lmax >= lmax; "
                f"got edge_lmax={self.ictd_fix_edge_lmax}, lmax={self.lmax}."
            )
        self.num_interaction = int(num_interaction)
        self.max_radius = float(max_embed_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)
        self.ictd_fix_route = str(ictd_fix_route)
        self.ictd_fix_contraction_combine = str(ictd_fix_contraction_combine)
        self.ictd_fix_requested_product_backend = requested_product_backend
        self.save_contraction_order = int(save_contraction_order)
        self.ictd_fix_product_backend = (
            "ictd-bridge-u"
            # native ICTC U generation is the reliability limit on the contraction INPUT degree,
            # which is edge_lmax (the message), not just the hidden lmax; fall back past degree 3.
            if requested_product_backend == "ictd-pure-u" and max(self.lmax, self.ictd_fix_edge_lmax) > 3
            else requested_product_backend
        )
        self.ictd_fix_use_reduced_cg = bool(ictd_fix_use_reduced_cg)
        self.ictd_fix_first_layer_self_connection = bool(ictd_fix_first_layer_self_connection)
        self.ictd_fix_self_connection_rank = int(ictd_fix_self_connection_rank)
        if self.ictd_fix_self_connection_rank < 0:
            raise ValueError(
                "ictd_fix_self_connection_rank must be non-negative, got "
                f"{ictd_fix_self_connection_rank}"
            )
        self.ictd_fix_conv_tp_scale_init = str(ictd_fix_conv_tp_scale_init)
        self.ictd_fix_freeze_conv_tp_weight = bool(ictd_fix_freeze_conv_tp_weight)
        self.ictd_fix_interaction_init = str(ictd_fix_interaction_init)
        self.ictd_fix_product_backend_fallback = self.ictd_fix_product_backend != self.ictd_fix_requested_product_backend
        if self.ictd_fix_edge_lmax != self.lmax and self.ictd_fix_product_backend not in {"native-mace", "ictd-bridge-u", "cueq", "ictd-pure-u"}:
            raise NotImplementedError(
                "ictd_fix_edge_lmax != lmax is currently supported only for the exact MACE product "
                "backends 'native-mace', 'ictd-bridge-u', 'cueq', and the native 'ictd-pure-u' "
                "(which itself falls back to 'ictd-bridge-u' when max(lmax, edge_lmax) > 3)."
            )
        self.ictd_fix_interaction_scale = str(ictd_fix_interaction_scale)
        self.ictd_fix_fusion_scale_init = float(ictd_fix_fusion_scale_init)
        self.ictd_fix_fusion_heads = int(ictd_fix_fusion_heads)
        self.ictd_fix_fusion_head_weight_mode = str(ictd_fix_fusion_head_weight_mode)
        self.ictd_fix_fusion_input_scale_init = float(ictd_fix_fusion_input_scale_init)
        self.ictd_fix_fusion_input_scale_trainable = bool(ictd_fix_fusion_input_scale_trainable)
        self.ictd_fix_fusion_depth_attention = bool(ictd_fix_fusion_depth_attention)
        self.ictd_fix_gmix_gate_init = float(ictd_fix_gmix_gate_init)
        self.ictd_fix_gmix_gate_trainable = bool(ictd_fix_gmix_gate_trainable)
        self.ictd_fix_gmix_block_rmsnorm = bool(ictd_fix_gmix_block_rmsnorm)
        self.ictd_fix_gmix_block_rmsnorm_gamma_init = float(ictd_fix_gmix_block_rmsnorm_gamma_init)
        self.ictd_fix_readout_head_scale_init = float(ictd_fix_readout_head_scale_init)
        self.ictd_fix_readout_head_scale_trainable = bool(ictd_fix_readout_head_scale_trainable)
        self.ictd_fix_fusion_readout_mixed_channels = bool(ictd_fix_fusion_readout_mixed_channels)
        self.ictd_fix_fusion_pre_product_norm = bool(ictd_fix_fusion_pre_product_norm)
        self.ictd_fix_interaction_rms_norm = bool(ictd_fix_interaction_rms_norm)
        # radial_sqrt_num_basis=False -> byte-literal MACE radial (default for new models);
        # from_checkpoint forces True for back-compat with FSCETP checkpoints trained with the scale.
        self.radial_sqrt_num_basis = bool(radial_sqrt_num_basis)
        self.ictd_fix_interaction_attn_heads = int(ictd_fix_interaction_attn_heads)
        self.ictd_fix_gmix_energy_readout = bool(ictd_fix_gmix_energy_readout)
        self.ictd_fix_gmix_readout_output_init_std = float(ictd_fix_gmix_readout_output_init_std)
        self.ictd_fix_layer_readout_output_init_std = float(ictd_fix_layer_readout_output_init_std)
        self.ictd_fix_readout_hidden_channels = int(ictd_fix_readout_hidden_channels)
        if self.ictd_fix_readout_hidden_channels <= 0:
            raise ValueError(
                "ictd_fix_readout_hidden_channels must be positive, "
                f"got {self.ictd_fix_readout_hidden_channels}"
            )
        self.ictd_fix_gmix_readout_scale_init = (
            float(self.ictd_fix_readout_head_scale_init)
            if ictd_fix_gmix_readout_scale_init is None
            else float(ictd_fix_gmix_readout_scale_init)
        )
        # Fusion gmix output lmax: the gmix (multiple_contraction_mix) symmetric
        # contraction can emit a HIGHER output lmax than the backbone input lmax,
        # giving product5 extra higher-l angular invariants from the (already
        # message-passed) backbone features, at near-zero backbone cost. Default
        # = lmax => byte-identical to before.
        self.ictd_fix_gmix_output_lmax = (
            self.lmax if ictd_fix_gmix_output_lmax is None else int(ictd_fix_gmix_output_lmax)
        )
        if self.ictd_fix_gmix_output_lmax < self.lmax:
            raise ValueError(
                f"ictd_fix_gmix_output_lmax ({self.ictd_fix_gmix_output_lmax}) must be >= lmax ({self.lmax})"
            )
        self.polynomial_cutoff_p = (
            None
            if polynomial_cutoff_p is None or int(polynomial_cutoff_p) <= 0
            else int(polynomial_cutoff_p)
        )
        if self.ictd_fix_fusion_heads < 1:
            raise ValueError(f"ictd_fix_fusion_heads must be >= 1, got {self.ictd_fix_fusion_heads}")
        self.max_atomvalue = int(max_atomvalue)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        self.edge_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        if atomic_numbers is None:
            atomic_numbers = tuple(range(self.max_atomvalue))
        else:
            atomic_numbers = tuple(sorted({int(z) for z in atomic_numbers}))
            if len(atomic_numbers) == 0:
                raise ValueError("atomic_numbers must not be empty")
        self.atomic_numbers = atomic_numbers
        self.num_elements = len(self.atomic_numbers)
        map_size = max(self.max_atomvalue, max(self.atomic_numbers) + 1)
        atomic_number_to_index = torch.full((map_size,), -1, dtype=torch.long)
        for idx, z in enumerate(self.atomic_numbers):
            if z < 0:
                raise ValueError(f"atomic_numbers must be non-negative, got {z}")
            atomic_number_to_index[z] = idx
        self.register_buffer("atomic_number_to_index", atomic_number_to_index, persistent=False)

        self.node_embedding = nn.Linear(self.num_elements, self.channels, bias=False)
        product_target_lmax = [
            self.lmax if layer_idx < self.num_interaction - 1 else 0
            for layer_idx in range(self.num_interaction)
        ]
        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        self.ictd_fix_effective_product_backends: list[str] = []
        for layer_idx, target_lmax in enumerate(product_target_lmax):
            effective_product_backend = self.ictd_fix_product_backend
            self.ictd_fix_effective_product_backends.append(effective_product_backend)
            input_lmax = 0 if layer_idx == 0 else self.lmax
            first_layer_sc = self.ictd_fix_first_layer_self_connection and layer_idx == 0
            sc_lmax = 0 if first_layer_sc else target_lmax
            message_scale_init = None
            sc_scale_init = None
            if self.ictd_fix_interaction_scale == "mace-rms":
                # Initialized from the ICTC/native-MACE basisbridge diagnostic on
                # aspirin lmax=3/ch64. The scales are learnable, so this is a
                # stabilization prior rather than a fixed calibration.
                message_presets = {
                    0: [0.625, 0.561, 0.540, 0.403],
                    1: [0.489, 0.745, 0.741, 0.620],
                }
                preset = list(message_presets.get(layer_idx, [0.5] * (self.ictd_fix_edge_lmax + 1)))
                if len(preset) < self.ictd_fix_edge_lmax + 1:
                    preset = preset + [0.5] * (self.ictd_fix_edge_lmax + 1 - len(preset))
                message_scale_init = preset[: self.ictd_fix_edge_lmax + 1]
                if sc_lmax == 0 and layer_idx > 0:
                    sc_scale_init = [0.342]
                elif sc_lmax > 0 and layer_idx > 0:
                    sc_scale_init = [0.342] + [0.5] * sc_lmax
            self.interactions.append(
                ICTDResidualInteractionBlock(
                    channels=self.channels,
                    lmax=self.ictd_fix_edge_lmax,
                    input_lmax=input_lmax,
                    target_lmax=self.ictd_fix_edge_lmax,
                    sc_lmax=sc_lmax,
                    number_of_basis=self.number_of_basis,
                    num_elements=self.num_elements,
                    function_type=self.function_type,
                    ictd_save_tp_mode=ictd_save_tp_mode,
                    ictd_tp_path_policy=ictd_tp_path_policy,
                    ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                    ictd_tp_backend=ictd_tp_backend,
                    equivariant_post_linear=equivariant_post_linear,
                    use_self_connection=(layer_idx > 0) or first_layer_sc,
                    self_connection_rank=self.ictd_fix_self_connection_rank,
                    self_connection_rng_offset=layer_idx,
                    avg_num_neighbors=self.avg_num_neighbors,
                    message_scale_init=message_scale_init,
                    sc_scale_init=sc_scale_init,
                    conv_tp_scale_init=self.ictd_fix_conv_tp_scale_init,
                    freeze_conv_tp_weight=self.ictd_fix_freeze_conv_tp_weight,
                    interaction_init=self.ictd_fix_interaction_init,
                    use_rms_norm=self.ictd_fix_interaction_rms_norm,
                    interaction_attn_heads=self.ictd_fix_interaction_attn_heads,
                    radial_mlp_hidden=self.radial_mlp_hidden,
                )
            )
            if effective_product_backend == "native-mace":
                self.products.append(
                    NativeMACEProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.ictd_fix_edge_lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                        use_reduced_cg=self.ictd_fix_use_reduced_cg,
                    )
                )
            elif effective_product_backend == "ictd-bridge-u":
                self.products.append(
                    ICTDBridgeUProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.ictd_fix_edge_lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                        use_reduced_cg=self.ictd_fix_use_reduced_cg,
                    )
                )
            elif effective_product_backend == "cueq":
                self.products.append(
                    CueqMACEProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.ictd_fix_edge_lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                        use_reduced_cg=self.ictd_fix_use_reduced_cg,
                    )
                )
            elif effective_product_backend == "ictd-pure-u":
                self.products.append(
                    ICTDPureUProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        # the interaction emits a full-SO(3) message up to edge_lmax; the contraction
                        # consumes it and truncates to target_lmax (== self.lmax). Mirror the exact
                        # MACE backends (native-mace/bridge-u/cueq) and pass edge_lmax as the input
                        # degree, not self.lmax, so edge_lmax != lmax (max_ell) is handled natively.
                        lmax=self.ictd_fix_edge_lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                    )
                )
            elif target_lmax == self.lmax:
                self.products.append(
                    ICTDProductBasisBlock(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        ictd_tp_path_policy=ictd_tp_path_policy,
                        ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                        internal_compute_dtype=internal_compute_dtype,
                        ictd_tp_backend=ictd_tp_backend,
                        contraction_combine=self.ictd_fix_contraction_combine,
                    )
                )
            else:
                self.products.append(
                    ICTDScalarProductBasisBlock(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        ictd_tp_path_policy=ictd_tp_path_policy,
                        ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                        internal_compute_dtype=internal_compute_dtype,
                        ictd_tp_backend=ictd_tp_backend,
                        contraction_combine=self.ictd_fix_contraction_combine,
                    )
                )
        self.layer_energy_readouts = nn.ModuleList(
            [EquivariantScalarReadoutSO3(self.channels, self.lmax, output_init_std=self.ictd_fix_layer_readout_output_init_std) for _ in range(self.num_interaction - 1)]
        )
        self.last_layer_energy_readout = MACEStyleScalarReadoutSO3(
            self.channels,
            hidden_channels=self.ictd_fix_readout_hidden_channels,
            output_init_std=self.ictd_fix_layer_readout_output_init_std,
        )
        if self.ictd_fix_readout_head_scale_trainable:
            self.readout_head_scales = nn.Parameter(
                torch.full((2,), self.ictd_fix_readout_head_scale_init, dtype=torch.get_default_dtype())
            )
        else:
            self.readout_head_scales = None

        # Fusion route removed in this baseline-only build -> these submodules are always absent.
        self.save_multiple_mix_channels = None
        self.multiple_contraction_mix = None
        self.multiple_contract_fuse = None
        self.ictd_fix_fusion_mix_backend = None
        self.fusion_readouts = None
        self.fusion_readout = None
        self.fusion_head_logits = None
        self.fusion_head_weights = None
        self.fusion_energy_scale = None
        self.fusion_input_scales = None
        self.fusion_depth_attention = None
        self.g_mix_gate = None
        self.gmix_block_rmsnorm_gamma = None
        self.gmix_energy_readout = None
        self.gmix_readout_head_scale = None

        # --- long-range interaction module (None when mode=="none"; no-op when off,
        # so the flagship's numerics + checkpoints are unchanged with long_range off).
        # Fed the final per-atom SCALAR descriptor (scalar_last for fusion /
        # layer_states[-1] for baseline, both invariant -> equivariance-safe);
        # energy_scale inits to 0 -> zero contribution at init.
        self.long_range_mode = str(long_range_mode)
        self.long_range_hidden_dim = int(long_range_hidden_dim)
        self.long_range_filter_hidden_dim = int(long_range_filter_hidden_dim)
        self.long_range_kmax = int(long_range_kmax)
        self.long_range_include_k0 = bool(long_range_include_k0)
        self.long_range_source_channels = int(long_range_source_channels)
        self.long_range_reciprocal_backend = str(long_range_reciprocal_backend)
        self.long_range_assignment = str(long_range_assignment)
        self.long_range_backend = str(long_range_backend)
        self.long_range_energy_partition = str(long_range_energy_partition)
        self.long_range_green_mode = str(long_range_green_mode)
        self.long_range_boundary = str(long_range_boundary)
        self.long_range_neutralize = bool(long_range_neutralize)
        self.long_range_mesh_fft_full_ewald = bool(long_range_mesh_fft_full_ewald)
        self.long_range_mesh_fft_reciprocal_only = bool(long_range_mesh_fft_reciprocal_only)
        self.long_range_smearing_sigma = long_range_smearing_sigma
        self.long_range_mesh_size = normalize_mesh_size(long_range_mesh_size)
        self.long_range_slab_padding_factor = int(long_range_slab_padding_factor)
        self.long_range_module = build_long_range_module(
            mode=self.long_range_mode,
            feature_dim=self.channels,
            hidden_dim=long_range_hidden_dim,
            boundary=long_range_boundary,
            neutralize=long_range_neutralize,
            filter_hidden_dim=long_range_filter_hidden_dim,
            kmax=long_range_kmax,
            mesh_size=long_range_mesh_size,
            slab_padding_factor=long_range_slab_padding_factor,
            include_k0=long_range_include_k0,
            source_channels=long_range_source_channels,
            backend=long_range_backend,
            reciprocal_backend=long_range_reciprocal_backend,
            energy_partition=long_range_energy_partition,
            green_mode=long_range_green_mode,
            assignment=long_range_assignment,
            mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
            mesh_fft_reciprocal_only=long_range_mesh_fft_reciprocal_only,
            smearing_sigma=long_range_smearing_sigma,
            max_multipole_l=long_range_max_multipole_l,
            multipole_feature_channels=self.channels,
            theta=long_range_theta,
            leaf_size=long_range_leaf_size,
            multipole_order=long_range_multipole_order,
            far_source_dim=long_range_far_source_dim,
            far_num_shells=long_range_far_num_shells,
            far_shell_growth=long_range_far_shell_growth,
            far_tail=long_range_far_tail,
            far_tail_bins=long_range_far_tail_bins,
            far_stats=long_range_far_stats,
            far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            far_source_norm=long_range_far_source_norm,
            far_gate_init=long_range_far_gate_init,
            cutoff_radius=self.max_radius,
        )
        self.long_range_exports_reciprocal_source = (
            bool(getattr(self.long_range_module, "exports_reciprocal_source", False))
            if self.long_range_module is not None
            else False
        )
        _lr = self.long_range_module
        _emit = self.long_range_exports_reciprocal_source
        self.long_range_runtime_backend = str(getattr(_lr, "runtime_backend", "none")) if _emit else "none"
        self.long_range_runtime_source_kind = str(getattr(_lr, "source_kind", "none")) if _emit else "none"
        self.long_range_runtime_source_channels = int(getattr(_lr, "source_channels", 0)) if _emit else 0
        self.long_range_runtime_source_layout = str(getattr(_lr, "source_layout", "none")) if _emit else "none"
        self.long_range_runtime_source_boundary = str(long_range_boundary)
        self.long_range_runtime_source_slab_padding_factor = int(long_range_slab_padding_factor)

        # Multipole long-range: tap the deepest full-SO(3) node features for equivariant
        # Cartesian monopole/dipole/quadrupole sources (a degree-l carrier IS a rank-l
        # multipole) and feed the mesh-FFT reciprocal kernel. OFF (max_multipole_l==0) ->
        # readout is None and the scalar latent-source path is used unchanged (byte-identical).
        self.long_range_max_multipole_l = int(long_range_max_multipole_l)
        self.long_range_multipole_gate_init = float(long_range_multipole_gate_init)
        self.multipole_readout = None
        if self.long_range_module is not None and self.long_range_max_multipole_l > 0:
            if str(long_range_reciprocal_backend) != "mesh_fft":
                raise ValueError(
                    "long_range_max_multipole_l>0 requires long_range_reciprocal_backend='mesh_fft' "
                    "(only the mesh-FFT kernel exposes multipole_energy)"
                )
            if not bool(long_range_mesh_fft_full_ewald):
                raise ValueError(
                    "long_range_max_multipole_l>0 requires long_range_mesh_fft_full_ewald=True: the "
                    "reciprocal multipole sum needs Ewald Gaussian screening for accuracy (without it "
                    "the in-cell translation error is tens of %); long_range_assignment='pcs' recommended"
                )
            if self.long_range_max_multipole_l > self.lmax:
                raise ValueError(
                    f"long_range_max_multipole_l={self.long_range_max_multipole_l} exceeds model lmax={self.lmax}"
                )
            from mace_ictc.models.multipole_readout import MultipoleReadout

            self.multipole_readout = MultipoleReadout(
                channels=self.channels,
                lmax=self.lmax,
                max_multipole_l=self.long_range_max_multipole_l,
                source_channels=int(long_range_source_channels),
                source_scale_init=self.long_range_multipole_gate_init,
            )
            if getattr(self.long_range_module, "energy_scale", None) is not None:
                with torch.no_grad():
                    self.long_range_module.energy_scale.fill_(1.0)
            # Legacy cubic Poisson models can emit packed [q|mu|Q] sources for
            # the C++ solver; other configurations keep reciprocal energy inline.
            self.long_range_exports_reciprocal_source = (
                str(long_range_green_mode) == "poisson" and long_range_smearing_sigma is None
                and isinstance(self.long_range_mesh_size, int))
            # Deploy metadata read by export_libtorch_core -> .json -> the C++ engine/solver.
            # The packed per-atom source is [q | dipole_xyz | quad_3x3] per channel (channel-major);
            # the C++ reciprocal solver rebuilds q/mu/Q from source_channels + max_multipole_l, and
            # multipole_reciprocal_energy mirrors the in-model multipole_energy (screened |S(k)|^2 PME).
            self.long_range_runtime_backend = "mesh_fft" if self.long_range_exports_reciprocal_source else "none"
            self.long_range_runtime_source_kind = "latent_multipole" if self.long_range_exports_reciprocal_source else "none"
            self.long_range_runtime_source_channels = int(long_range_source_channels) if self.long_range_exports_reciprocal_source else 0
            self.long_range_runtime_source_layout = "packed_q_dipole_quad" if self.long_range_exports_reciprocal_source else "none"
            self.long_range_runtime_source_boundary = str(long_range_boundary)
            # Deploy config the .json writer reads, taken from the built mesh kernel (the source of
            # truth) so the C++ reciprocal solver reproduces the in-model multipole_energy exactly:
            # screened |S(k)|^2 PME with green_mode/mesh_size/boundary/full_ewald all matched.
            _mp_kernel = self.long_range_module.kernel
            self.long_range_mesh_fft_full_ewald = bool(getattr(_mp_kernel, "full_ewald", bool(long_range_mesh_fft_full_ewald)))
            self.long_range_mesh_size = normalize_mesh_size(getattr(_mp_kernel, "mesh_size", 16))
            self.long_range_green_mode = str(getattr(_mp_kernel, "green_mode", "poisson"))
            self.long_range_boundary = str(getattr(_mp_kernel, "boundary", "periodic"))
            # multipole_energy always distributes e_graph/n_local uniformly (it ignores the kernel's
            # energy_partition); report that honestly so the C++ per-atom decomposition can match.
            self.long_range_energy_partition = "uniform"
            self.long_range_neutralize = bool(getattr(self.long_range_module, "neutralize", True))

        # Long-range dispersion term. OFF by default -> None -> no contribution
        # (byte-identical). The legacy boolean `long_range_dispersion=True` maps to
        # `pairwise-c6`; future MBD modes should plug into this same interface rather
        # than adding another hand-written branch in forward.
        self.long_range_dispersion_mode = normalize_dispersion_mode(
            long_range_dispersion=bool(long_range_dispersion),
            long_range_dispersion_mode=long_range_dispersion_mode,
        )
        self.long_range_dispersion = self.long_range_dispersion_mode != "none"
        self.dispersion_cutoff = float(dispersion_cutoff)
        if dispersion_max_num_neighbors is not None and int(dispersion_max_num_neighbors) < 0:
            raise ValueError("dispersion_max_num_neighbors must be >= 0 or None")
        self.dispersion_max_num_neighbors = (
            None if dispersion_max_num_neighbors is None or int(dispersion_max_num_neighbors) == 0
            else int(dispersion_max_num_neighbors)
        )
        self.dispersion_neighbor_method = str(dispersion_neighbor_method)
        self.dispersion_bruteforce_threshold = int(dispersion_bruteforce_threshold)
        self.dispersion_allow_large_bruteforce_fallback = bool(dispersion_allow_large_bruteforce_fallback)
        self.dispersion_slq_num_probes = int(dispersion_slq_num_probes)
        self.dispersion_slq_lanczos_steps = int(dispersion_slq_lanczos_steps)
        self.mbd_operator_backend = str(mbd_operator_backend)
        self.mbd_pme_mesh_size = int(mbd_pme_mesh_size)
        self.mbd_pme_assignment = str(mbd_pme_assignment)
        self.mbd_pme_k_norm_floor = float(mbd_pme_k_norm_floor)
        self.mbd_pme_assignment_window_floor = float(mbd_pme_assignment_window_floor)
        self.mbd_pme_ewald_alpha_prefactor = float(mbd_pme_ewald_alpha_prefactor)
        self.mbd_anisotropic_polarizability = bool(mbd_anisotropic_polarizability)
        self.mbd_learnable_energy_scale = bool(mbd_learnable_energy_scale)
        self.mbd_alpha_floor = float(mbd_alpha_floor)
        self.dispersion_min_cutoff = float(dispersion_min_cutoff)
        self.dispersion_switch_width = float(dispersion_switch_width)
        self.dispersion_pbc = str(long_range_boundary) == "periodic"
        self.dispersion = build_long_range_dispersion(
            mode=self.long_range_dispersion_mode,
            feature_dim=self.channels,
            cutoff=self.dispersion_cutoff,
            pbc=self.dispersion_pbc,
            neighbor_method=self.dispersion_neighbor_method,
            bruteforce_threshold=self.dispersion_bruteforce_threshold,
            allow_large_bruteforce_fallback=self.dispersion_allow_large_bruteforce_fallback,
            slq_num_probes=self.dispersion_slq_num_probes,
            slq_lanczos_steps=self.dispersion_slq_lanczos_steps,
            max_num_neighbors=self.dispersion_max_num_neighbors,
            mbd_operator_backend=self.mbd_operator_backend,
            mbd_pme_mesh_size=self.mbd_pme_mesh_size,
            mbd_pme_assignment=self.mbd_pme_assignment,
            mbd_pme_k_norm_floor=self.mbd_pme_k_norm_floor,
            mbd_pme_assignment_window_floor=self.mbd_pme_assignment_window_floor,
            mbd_pme_ewald_alpha_prefactor=self.mbd_pme_ewald_alpha_prefactor,
            mbd_anisotropic_polarizability=self.mbd_anisotropic_polarizability,
            mbd_learnable_energy_scale=self.mbd_learnable_energy_scale,
            mbd_alpha_floor=self.mbd_alpha_floor,
            dispersion_min_cutoff=self.dispersion_min_cutoff,
            dispersion_switch_width=self.dispersion_switch_width,
        )

        # MBD-source packing metadata (read by the exporter -> .json -> C++ engine/pair-style). When the
        # mbd-slq head exports a source, the deploy forward emits [electrostatic | omega, alpha]; the MBD
        # (omega,alpha) begins at offset = the electrostatic source width (0 if no electrostatic source).
        self.long_range_mbd_source_enabled = bool(
            self.dispersion is not None and self.dispersion.exports_mbd_source()
        )
        # MBD source width: isotropic [omega, alpha] = 2; anisotropic [omega, alpha_iso, 6*B] = 8.
        self.long_range_mbd_source_channels = 8 if self.mbd_anisotropic_polarizability else 2
        _l = self.long_range_max_multipole_l
        self.long_range_mbd_source_offset = (
            int(getattr(self, "long_range_runtime_source_channels", 0))
            * (1 + (3 if _l >= 1 else 0) + (9 if _l >= 2 else 0))
            if self.long_range_exports_reciprocal_source
            else 0
        )
        if self.long_range_mbd_source_enabled:
            # emit the reciprocal_source even for an MBD-only model (no electrostatics) so the C++ MBD
            # solver gets (omega, alpha); the reciprocal solver stays gated on the electrostatic channels.
            self.long_range_exports_reciprocal_source = True

        # Optional fixed scale/shift on the network (short-range) per-atom interaction energy.
        # This mirrors MACE ScaleShiftMACE: E_inter_atom = scale * readout + shift. OFF by
        # default -> None buffers are excluded from state_dict, so old checkpoints load unchanged
        # with strict=True. E0 is added afterward (outside the model) and is NOT scaled/shifted.
        self.energy_output_scale_enabled = bool(energy_output_scale_enabled)
        if self.energy_output_scale_enabled:
            # Store at full (float64) precision; cast to the compute dtype at use in forward.
            # (Storing in the default float32 would round the scale and perturb energies ~1e-8.)
            self.register_buffer(
                "energy_output_scale",
                torch.tensor(float(energy_output_scale), dtype=torch.float64),
            )
        else:
            self.register_buffer("energy_output_scale", None)
        self.energy_output_shift_enabled = bool(energy_output_shift_enabled)
        if self.energy_output_shift_enabled:
            self.register_buffer(
                "energy_output_shift",
                torch.tensor(float(energy_output_shift), dtype=torch.float64),
            )
        else:
            self.register_buffer("energy_output_shift", None)
        # Optional converter-only additive term for MACE's first interaction skip connection.
        # It is None for normal training/checkpoints. `convert_mace_to_ictd` installs a
        # tensor of shape (num_elements, channels) so the converted model can reproduce
        # mace-torch without relying on Python forward hooks, which are brittle under export.
        self.register_buffer("mace_first_layer_sc0", None)

    def _readout_head_scale(self, index: int, ref: torch.Tensor) -> torch.Tensor:
        if self.readout_head_scales is None:
            # new_zeros(()) is a device memset (no host->device copy) so this stays
            # CUDA-graph capturable; +scalar is a kernel arg. Equals the scalar.
            return ref.new_zeros(()) + float(self.ictd_fix_readout_head_scale_init)
        return self.readout_head_scales[index].to(dtype=ref.dtype, device=ref.device)

    def _fusion_input_scale(self, index: int, ref: torch.Tensor) -> torch.Tensor:
        if self.fusion_input_scales is None:
            return ref.new_zeros(()) + float(self.ictd_fix_fusion_input_scale_init)
        return self.fusion_input_scales[index].to(dtype=ref.dtype, device=ref.device)

    def _g_mix_gate(self, ref: torch.Tensor) -> torch.Tensor:
        if self.g_mix_gate is None:
            return ref.new_zeros(()) + float(self.ictd_fix_gmix_gate_init)
        return self.g_mix_gate.to(dtype=ref.dtype, device=ref.device)

    def _maybe_gmix_block_rmsnorm(self, g_mix: torch.Tensor) -> torch.Tensor:
        if self.gmix_block_rmsnorm_gamma is None:
            return g_mix
        channels = self.save_multiple_mix_channels if self.ictd_fix_fusion_readout_mixed_channels else self.channels
        return _so3_block_rmsnorm(
            g_mix,
            int(channels),
            self.ictd_fix_gmix_output_lmax,
            self.gmix_block_rmsnorm_gamma,
        )

    def install_mace_first_layer_sc0(self, sc0_by_element: torch.Tensor | None) -> None:
        """Install the first-layer element skip term used by converted mace-torch models.

        MACE's first residual block adds a pure l=0, element-conditioned self connection.
        This baseline ICTC block has no parameter slot for that exact additive constant, so
        conversion stores it as a non-trainable buffer and `forward` adds it after product[0].
        """
        if sc0_by_element is None:
            self.mace_first_layer_sc0 = None
            return
        sc0 = sc0_by_element.detach().clone()
        if sc0.shape != (self.num_elements, self.channels):
            raise ValueError(
                f"mace_first_layer_sc0 must have shape ({self.num_elements}, {self.channels}), "
                f"got {tuple(sc0.shape)}"
            )
        self.mace_first_layer_sc0 = sc0

    def _apply_e3nn_basis_fold(self) -> None:
        """Fold the FIXED angular operators (interaction Clebsch-Gordan tensors + the
        symmetric-contraction U tensors) into the e3nn/MACE spherical basis so the model
        computes its l>=1 features natively in the e3nn convention (``angular_basis="e3nn"``).

        Combined with the harmonics fold in ``forward``, this is a single global orthogonal
        change of the angular basis: every intermediate equivariant feature becomes its e3nn
        counterpart, while the energy / forces / virial (SO(3) invariants) are unchanged.
        Learnable weights are NOT touched (they index channel / path axes that the fold
        preserves), so an ``e3nn`` model is the SAME function as its ``ictd`` twin in a rotated
        basis -> bit-identical output. Idempotent; runs once (lazily on first forward)."""
        if getattr(self, "_e3nn_folded", False):
            return
        from mace_ictc.mace_basis import orthogonal_Q_blocks
        ref = next(self.parameters())
        q_blocks_cpu = orthogonal_Q_blocks(
            max(self.lmax, self.ictd_fix_edge_lmax),
            dtype=torch.float64,
            device="cpu",
        )
        self._e3nn_q_blocks = [q.to(dtype=ref.dtype, device=ref.device) for q in q_blocks_cpu]
        folded_u = False
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "fold_cg_to_e3nn"):
                module.fold_cg_to_e3nn(q_blocks_cpu)
            if hasattr(module, "enable_e3nn_basis"):
                module.enable_e3nn_basis(self._e3nn_q_blocks)
                folded_u = True
        if not folded_u:
            raise NotImplementedError(
                "angular_basis='e3nn' currently requires the symmetric-contraction backend to "
                "expose enable_e3nn_basis (currently product_backend='ictd-pure-u' or 'cueq'). The "
                f"selected backend {getattr(self, 'ictd_fix_product_backend', '?')!r} has no e3nn fold.")
        self._e3nn_folded = True

    def activate_e3nn_basis_from_folded_state_dict(self) -> None:
        """Restore runtime e3nn-basis state after loading already-folded buffers.

        Training with ``angular_basis='e3nn'`` folds fixed interaction tensors on the
        first forward. Those folded tensors are saved in ``state_dict``. Reload must
        not fold them again, but forward still needs the Q blocks for edge harmonics
        and product backends such as cuEq still need their e3nn-basis runtime flag.
        """
        if self.angular_basis != "e3nn":
            raise ValueError("activate_e3nn_basis_from_folded_state_dict requires angular_basis='e3nn'")
        if getattr(self, "_e3nn_folded", False):
            return
        from mace_ictc.mace_basis import orthogonal_Q_blocks
        ref = next(self.parameters())
        q_blocks_cpu = orthogonal_Q_blocks(
            max(self.lmax, self.ictd_fix_edge_lmax),
            dtype=torch.float64,
            device="cpu",
        )
        self._e3nn_q_blocks = [q.to(dtype=ref.dtype, device=ref.device) for q in q_blocks_cpu]
        enabled = False
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "enable_e3nn_basis"):
                module.enable_e3nn_basis(self._e3nn_q_blocks)
                enabled = True
        if not enabled:
            raise NotImplementedError(
                "Loaded angular_basis='e3nn' checkpoint requires a product backend with enable_e3nn_basis."
            )
        self._e3nn_folded = True

    def to_mace_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Re-express an equivariant feature tensor in the *original-MACE / e3nn* basis.

        ``x`` carries the ``(lmax+1)**2`` angular components in its last axis (ICTC basis); the
        result is ``x @ Q`` with the fixed block-diagonal orthogonal ``Q`` (see
        :mod:`mace_ictc.mace_basis`). Energy, forces and the virial are SO(3) invariants / physical
        tensors and are basis-independent, so this only matters for equivariant (l>=1) features.
        """
        from mace_ictc.mace_basis import to_mace_basis as _to_mace
        return _to_mace(x, self.lmax)

    def to_ictd_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`to_mace_basis` (original-MACE/e3nn -> ICTC)."""
        from mace_ictc.mace_basis import to_ictd_basis as _to_ictd
        return _to_ictd(x, self.lmax)

    def forward(
        self,
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        *,
        precomputed_edge_vec=None,
        precomputed_edge_feats=None,
        dispersion_edge_src=None,
        dispersion_edge_dst=None,
        dispersion_edge_shifts=None,
        precomputed_dispersion_edge_vec=None,
        return_combined_features: bool = False,
        layer_channel_gains: torch.Tensor | None = None,
        edge_block_callback: callable | None = None,
        pre_product_state_callback: callable | None = None,
        layer_state_callback: callable | None = None,
        sync_after_scatter: callable | None = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
        local_atom_count: int | None = None,
        compact_local_states: bool = False,
        compact_local_projection: bool = False,
        compact_local_messages: bool = False,
        distributed_atom_count: int | None = None,
        long_range_energy_callback: callable | None = None,
    ):
        if return_physical_tensors:
            raise ValueError("pure-cartesian-ictd-fix does not currently support return_physical_tensors=True")

        # Domain inference only needs the energies of the owned prefix. Trace
        # its dependencies backwards through the supplied complete halo graph;
        # later layers need progressively fewer centers. Ghost coordinates stay
        # in autograd so the caller can reverse-communicate their forces.
        # Layerwise domain decomposition supplies only owned-center edges.
        # The caller exchanges intermediate states (and their adjoints) through
        # layer_state_callback. Unlike complete-halo inference, every layer has
        # the same owned receivers; no backwards dependency closure is needed.
        distributed_mode = distributed_atom_count is not None
        if distributed_mode:
            if (local_atom_count is not None or layer_state_callback is None
                    or sync_after_scatter is not None or compact_local_states):
                raise ValueError("Distributed inference requires a state exchange callback, "
                                 "full ghost state storage, and no complete-halo pruning")
            local_atom_count = distributed_atom_count
        local_mode = local_atom_count is not None and local_atom_count < pos.shape[0]
        if local_atom_count is not None and (not isinstance(local_atom_count, int)
                or isinstance(local_atom_count, bool) or not 0 < local_atom_count <= pos.shape[0]):
            raise ValueError("local_atom_count must select a nonempty owned prefix")
        if local_mode and (self.training or return_combined_features or return_reciprocal_source
                or (self.long_range_module is not None and long_range_energy_callback is None) or self.dispersion is not None
                or self.ictd_fix_interaction_attn_heads != 0
                or layer_channel_gains is not None or edge_block_callback is not None
                or pre_product_state_callback is not None
                or (layer_state_callback is not None and not distributed_mode)
                or sync_after_scatter is not None):
            raise ValueError("Owned-prefix pruning supports short-range inference without callbacks")

        dtype = next(self.parameters()).dtype
        if self.angular_basis == "e3nn" and not self._e3nn_folded:
            self._apply_e3nn_basis_fold()
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        if not getattr(self, "preserve_edge_order", False):
            sort_idx = torch.argsort(edge_dst)
            edge_src = edge_src[sort_idx]
            edge_dst = edge_dst[sort_idx]
            edge_shifts = edge_shifts[sort_idx]
        else:
            sort_idx = None
        edge_index = torch.stack([edge_src, edge_dst], dim=0)

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec if sort_idx is None else precomputed_edge_vec[sort_idx]
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs

        edge_length = edge_vec.norm(dim=1)
        n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)
        edge_mask = (
            None
            if getattr(self, "assume_edges_within_radius", False)
            else (edge_length <= self.max_radius).to(dtype=pos.dtype).unsqueeze(-1)
        )
        Y_list = None
        deployment_harmonics=getattr(self,"_deployment_harmonics",None)
        if deployment_harmonics is not None:Y_list=deployment_harmonics(n.to(dtype=dtype))
        if Y_list is None and getattr(self,"_fused_direction_harmonics",False):
            from .generated_direction_harmonics import generated_direction_harmonics
            Y_list=generated_direction_harmonics(n.to(dtype=dtype),self.ictd_fix_edge_lmax)
        if Y_list is None:Y_list = direction_harmonics_all(n.to(dtype=dtype), self.ictd_fix_edge_lmax)
        if self.angular_basis == "e3nn":
            # fold the angular embedding into the e3nn/MACE spherical basis (Y_ictd @ Q_l = Y_e3nn)
            Y_list = [Y_list[l] @ self._e3nn_q_blocks[l].to(dtype=Y_list[l].dtype, device=Y_list[l].device)
                      for l in range(self.ictd_fix_edge_lmax + 1)]
        edge_attrs = {l: Y_list[l].to(dtype=dtype).unsqueeze(-2) for l in range(self.ictd_fix_edge_lmax + 1)}
        if precomputed_edge_feats is None:
            edge_feats = mace_radial_embedding(
                edge_length,
                r_max=self.max_radius,
                number_of_basis=self.number_of_basis,
                function_type=self.function_type,
                polynomial_cutoff_p=self.polynomial_cutoff_p,
                sqrt_num_basis_norm=self.radial_sqrt_num_basis,
            ).to(dtype=dtype)
        else:
            edge_feats = precomputed_edge_feats.to(dtype=dtype, device=pos.device)
            if sort_idx is not None:
                edge_feats = edge_feats[sort_idx]
        # Per-edge cutoff envelope for the optional interaction neighbor-attention
        # (env^2-gating keeps attention -> 0 smoothly at r_max so forces stay continuous).
        # Same MACE polynomial cutoff that is baked into edge_feats. Only built when needed.
        edge_env = (
            mace_polynomial_cutoff(
                edge_length,
                self.max_radius,
                self.polynomial_cutoff_p if self.polynomial_cutoff_p is not None else 6,
            ).to(dtype=dtype)
            if self.ictd_fix_interaction_attn_heads > 0
            else None
        )

        A_long = A.long()
        # `skip_input_validation` removes the two host syncs below (`.item()` /
        # `torch.any` + `.tolist()`) so this forward can be captured by a CUDA
        # graph. It only disables guards; the numerics (compact_idx, one_hot) are
        # unchanged. The capture wrapper validates inputs once before enabling it.
        if not getattr(self, "skip_input_validation", False):
            if int(A_long.max().item()) >= self.atomic_number_to_index.numel():
                raise ValueError(
                    f"Encountered atomic number {int(A_long.max().item())}, but compact mapping supports only up to "
                    f"{self.atomic_number_to_index.numel() - 1}. atomic_numbers={self.atomic_numbers}"
                )
        compact_idx = self.atomic_number_to_index[A_long]
        if not getattr(self, "skip_input_validation", False):
            if torch.any(compact_idx < 0):
                bad = torch.unique(A_long[compact_idx < 0]).tolist()
                raise ValueError(
                    f"Encountered atomic numbers without compact mapping: {bad}. "
                    f"Configured atomic_numbers={self.atomic_numbers}"
                )
        type_idx_only = all(
            isinstance(
                product,
                (CueqMACEProductBasisBlockSO3, ICTDBridgeUProductBasisBlockSO3),
            )
            for product in self.products
        )
        if type_idx_only:
            node_attrs = None
            embedding_weight = self.node_embedding.weight.t().to(dtype=dtype, device=pos.device)
            h = embedding_weight.index_select(0, compact_idx)
        else:
            node_attrs = F.one_hot(compact_idx, num_classes=self.num_elements).to(dtype=dtype)
            h = self.node_embedding(node_attrs)

        layer_states: List[torch.Tensor] = []
        if layer_channel_gains is not None:
            expected_gain_shape = (pos.shape[0], self.num_interaction, self.channels)
            if not getattr(self, "skip_input_validation", False):
                if tuple(layer_channel_gains.shape) != expected_gain_shape:
                    raise ValueError(
                        "layer_channel_gains must have shape "
                        f"{expected_gain_shape}, got {tuple(layer_channel_gains.shape)}"
                    )
            layer_channel_gains = layer_channel_gains.to(dtype=dtype, device=pos.device)
        last_preproduct_state: torch.Tensor | None = None
        total_energy = None
        local_layers = None
        local_row_map = None
        input_attrs, input_types = node_attrs, compact_idx
        if local_mode and distributed_mode:
            if not getattr(self, "skip_input_validation", False) and bool(
                    torch.any(edge_dst >= local_atom_count)):
                raise ValueError("Distributed edges must terminate on owned atoms")
            nodes = torch.arange(local_atom_count, device=pos.device)
            edges = torch.arange(edge_src.numel(), device=pos.device)
            local_layers = [(nodes, edges)] * self.num_interaction
        elif local_mode:
            required = torch.arange(pos.shape[0], device=pos.device) < local_atom_count
            local_layers = []
            for _ in range(self.num_interaction):
                nodes = torch.nonzero(required).flatten()
                edges = torch.nonzero(required[edge_dst]).flatten()
                local_layers.append((nodes, edges))
                previous = required.clone()
                previous[edge_src[edges]] = True
                required = previous
            local_layers.reverse()
        for layer_idx, (interaction, product) in enumerate(zip(self.interactions, self.products)):
            current_edge_block_callback = None
            if edge_block_callback is not None:
                def current_edge_block_callback(*, _layer_idx=layer_idx, **callback_kwargs):
                    return edge_block_callback(
                        _layer_idx,
                        edge_vec=edge_vec,
                        edge_sort_idx=sort_idx,
                        **callback_kwargs,
                    )
            if local_layers is not None:
                active_nodes, active_edges = local_layers[layer_idx]
            if local_layers is not None and active_edges.numel() != edge_src.numel():
                layer_edge_attrs = {l: value.index_select(0, active_edges) for l, value in edge_attrs.items()}
                layer_edge_feats = edge_feats.index_select(0, active_edges)
                layer_edge_index = edge_index.index_select(1, active_edges)
                layer_edge_mask = None if edge_mask is None else edge_mask.index_select(0, active_edges)
                layer_edge_env = None if edge_env is None else edge_env.index_select(0, active_edges)
            else:
                layer_edge_attrs, layer_edge_feats, layer_edge_index = edge_attrs, edge_feats, edge_index
                layer_edge_mask, layer_edge_env = edge_mask, edge_env
            if local_row_map is not None:
                layer_edge_index = local_row_map[layer_edge_index]
            receiver_rows = None
            if local_layers is not None and (compact_local_projection or compact_local_messages):
                receiver_rows = active_nodes if local_row_map is None else local_row_map[active_nodes]
            interaction_output = interaction(
                node_attrs=input_attrs,
                node_feats=h,
                edge_attrs=layer_edge_attrs,
                edge_feats=layer_edge_feats,
                edge_index=layer_edge_index,
                edge_mask=layer_edge_mask,
                edge_env=layer_edge_env,
                node_type_idx=input_types,
                edge_block_callback=current_edge_block_callback,
                sync_after_scatter=sync_after_scatter,
                receiver_rows=receiver_rows,
                compact_message_output=compact_local_messages,
            )
            auxiliary_message = None
            if len(interaction_output) == 3:
                message, sc, auxiliary_message = interaction_output
            else:
                message, sc = interaction_output
            if pre_product_state_callback is not None:
                if auxiliary_message is None:
                    message = pre_product_state_callback(layer_idx, message)
                else:
                    message = pre_product_state_callback(
                        layer_idx,
                        message,
                        auxiliary_message=auxiliary_message,
                    )
            product_attrs, product_types = input_attrs, input_types
            if local_layers is not None:
                product_rows = active_nodes if local_row_map is None else local_row_map[active_nodes]
                if receiver_rows is None:
                    message = message.index_select(0, product_rows)
                    sc = None if sc is None else sc.index_select(0, product_rows)
                product_attrs = None if input_attrs is None else input_attrs.index_select(0, product_rows)
                product_types = input_types.index_select(0, product_rows)
            if isinstance(
                product,
                (CueqMACEProductBasisBlockSO3, ICTDBridgeUProductBasisBlockSO3),
            ):
                h = product(node_feats=message, sc=sc, node_attrs=product_attrs, node_type_idx=product_types)
            else:
                h = product(node_feats=message, sc=sc, node_attrs=product_attrs)
            if layer_idx == 0 and self.mace_first_layer_sc0 is not None:
                add = self.mace_first_layer_sc0.to(dtype=h.dtype, device=h.device)[product_types]
                h = torch.cat((h[..., : self.channels] + add, h[..., self.channels :]), dim=-1)
            if local_layers is not None:
                if compact_local_states:
                    input_attrs, input_types = product_attrs, product_types
                    local_row_map = torch.full((pos.shape[0],), -1, dtype=torch.long, device=pos.device)
                    local_row_map[active_nodes] = torch.arange(active_nodes.numel(), device=pos.device)
                else:
                    h = h.new_zeros((pos.shape[0], h.shape[1])).index_copy(0, active_nodes, h)
            if layer_channel_gains is not None:
                gain = layer_channel_gains[:, layer_idx, :]
                target_lmax = self.lmax if layer_idx < self.num_interaction - 1 else 0
                if target_lmax == 0:
                    h = h * gain
                else:
                    blocks = _split_irreps(h, self.channels, target_lmax)
                    h = _merge_irreps(
                        {
                            l: block * gain[..., None]
                            for l, block in blocks.items()
                        },
                        self.channels,
                        target_lmax,
                    )
            if layer_state_callback is not None:
                h = layer_state_callback(layer_idx, h)
            layer_states.append(h)
            if layer_idx == self.num_interaction - 1:
                last_preproduct_state = message  # last-layer interaction output (pre-product, full SO3, 2-hop)
            if layer_idx < self.num_interaction - 1:
                e_layer = self.layer_energy_readouts[layer_idx](h[:local_atom_count] if local_mode else h)
                e_layer = self._readout_head_scale(0, e_layer) * e_layer
            else:
                e_layer = self.last_layer_energy_readout(h[:local_atom_count] if local_mode else h)
                e_layer = self._readout_head_scale(1, e_layer) * e_layer
            total_energy = e_layer if total_energy is None else (total_energy + e_layer)

        out = total_energy.sum(dim=-1, keepdim=True)
        if local_mode:
            # Ghost energies are unspecified; retain the standard [N,1] ABI.
            out = torch.cat((out, out.new_zeros((pos.shape[0] - local_atom_count, 1))))

        # Optional MACE ScaleShiftMACE-style scale/shift on the short-range interaction energy.
        # None when disabled -> byte-identical. Applied BEFORE the long-range add so it affects
        # only the network interaction energy; E0 is added outside this module.
        if self.energy_output_scale is not None:
            out = out * self.energy_output_scale.to(dtype=out.dtype, device=out.device)
        if self.energy_output_shift is not None:
            out = out + self.energy_output_shift.to(dtype=out.dtype, device=out.device)

        # --- long-range additive term (skipped entirely when module is None) ---
        reciprocal_source = None
        if self.long_range_module is not None and self.multipole_readout is not None:
            # Multipole route: equivariant Cartesian monopole/dipole/quadrupole sources fed to the
            # mesh-FFT reciprocal kernel. Source = the last layer's INTERACTION output (pre-product
            # message): the interaction always emits full-SO(3) (edge_lmax); only the PRODUCT truncates
            # the last layer to l=0. So it carries l=1/2 at every depth, single-layer included, and is
            # the deepest such feature. multipole_readout front-slices via _split_irreps, so a wider
            # edge_lmax message is safe.
            mp_feat = last_preproduct_state
            if mp_feat is None:
                raise RuntimeError(
                    "multipole long-range needs a full-SO(3) node-feature layer; none found"
                )
            if long_range_energy_callback is not None and local_atom_count is not None:
                mp_feat = mp_feat[:local_atom_count]
            monopole, dipole, quadrupole = self.multipole_readout(mp_feat)
            if long_range_energy_callback is not None:
                out = out + long_range_energy_callback(pos, batch, cell, monopole, dipole, quadrupole)
            elif return_reciprocal_source and self.long_range_exports_reciprocal_source:
                # deployment: emit the packed [q|mu|Q] per-atom source for the C++ reciprocal
                # solver (mff_reciprocal_solver) to do the long-range sum; defer the energy here.
                from mace_ictc.models.multipole_readout import pack_multipole_source

                reciprocal_source = pack_multipole_source(monopole, dipole, quadrupole)
            else:
                # training/validation: compute the reciprocal multipole energy in-model.
                long_range_energy = self.long_range_module.forward_multipole(
                    pos, batch, cell, monopole, dipole, quadrupole
                )
                if long_range_energy is not None:
                    out = out + long_range_energy
        elif self.long_range_module is not None:
            # final per-atom INVARIANT descriptor [N, channels] (in scope for both routes):
            # baseline last layer_state is already scalar; fusion last_preproduct is full-SO3 -> take l=0.
            last_state = layer_states[-1]
            # baseline: the last layer is scalar (l=0) -> the long-range source is the invariant
            # per-atom descriptor directly (no l>=1 multipole tap; that path needed the fusion route).
            if last_state.shape[-1] == self.channels:
                lr_feat = last_state
            else:
                lr_feat = _split_irreps(last_state, self.channels, self.lmax)[0].reshape(last_state.shape[0], self.channels)
            defer = False
            if long_range_energy_callback is not None:
                if local_atom_count is not None:
                    lr_feat = lr_feat[:local_atom_count]
                long_range_energy = long_range_energy_callback(
                    pos, batch, cell, self.long_range_module.emit_source(lr_feat), None, None)
            elif return_reciprocal_source and self.long_range_exports_reciprocal_source:
                # Deploy: emit ONLY the latent charge source (source_head); defer the reciprocal
                # energy + neutralization to the C++ solver. emit_source avoids the kernel's per-graph
                # torch.nonzero so make_fx/AOTI can trace it (forward(return_source=True) would not).
                reciprocal_source = self.long_range_module.emit_source(lr_feat)
                long_range_energy = None
                defer = True
            else:
                long_range_energy = self.long_range_module(
                    lr_feat, pos, batch, cell, edge_src=edge_src, edge_dst=edge_dst
                )
            if long_range_energy is not None and not defer:
                out = out + long_range_energy

        # --- pairwise dispersion additive term (invariant) ---
        if self.dispersion is not None:
            last_state = layer_states[-1]
            disp_l2 = None
            if last_state.shape[-1] == self.channels:
                disp_feat = last_state
            else:
                disp_feat = _split_irreps(last_state, self.channels, self.lmax)[0].reshape(
                    last_state.shape[0], self.channels
                )
            # ANISOTROPIC MBD: feed the equivariant l=2 node block to the dispersion head (ICTC l=2 ->
            # 3x3 anisotropic polarizability). Source = the last layer's INTERACTION output (pre-product
            # message): the interaction always emits full-SO(3) (edge_lmax); only the PRODUCT truncates
            # the last layer to l=0. So the l=2 block is present at every depth, single-layer included.
            # _split_irreps front-slices l=2, so a wider edge_lmax message stays correct. (Without an
            # l=2 source disp_l2 stays None and polarizability_factor falls back to the ISOTROPIC b0*I,
            # silently leaving the anisotropic l2_mix/l2_gate untrained.)
            if self.mbd_anisotropic_polarizability and self.lmax >= 2 and last_preproduct_state is not None:
                disp_l2 = _split_irreps(last_preproduct_state, self.channels, self.lmax)[2]  # [N, channels, 5]
            if return_reciprocal_source and self.dispersion.exports_mbd_source():
                # Deploy: emit the head's (omega, alpha) as the MBD source and DEFER the coupled-dipole
                # energy to the C++ MBD solver (no double count). PACK after any electrostatic source so
                # a COMBINED model carries both: [elec | omega, alpha]. The C++ reciprocal solver ignores
                # the trailing channels; the MBD solver reads source[:, mbd_offset:mbd_offset+2].
                mbd_source = self.dispersion.emit_source(disp_feat, disp_l2)
                reciprocal_source = (
                    mbd_source if reciprocal_source is None
                    else torch.cat([reciprocal_source, mbd_source], dim=1)
                )
            else:
                disp_edge_src = edge_src
                disp_edge_dst = edge_dst
                disp_edge_length = edge_length
                disp_edge_vec = edge_vec
                disp_cutoff = self.dispersion_cutoff
                if dispersion_edge_src is not None or dispersion_edge_dst is not None:
                    if dispersion_edge_src is None or dispersion_edge_dst is None:
                        raise ValueError("dispersion_edge_src and dispersion_edge_dst must be provided together")
                    disp_edge_src = dispersion_edge_src
                    disp_edge_dst = dispersion_edge_dst
                    disp_cutoff = 0.0
                    if precomputed_dispersion_edge_vec is not None:
                        disp_edge_vec = precomputed_dispersion_edge_vec
                    else:
                        if dispersion_edge_shifts is None:
                            raise ValueError(
                                "dispersion_edge_shifts or precomputed_dispersion_edge_vec is required "
                                "when explicit dispersion edges are provided"
                            )
                        disp_shift = dispersion_edge_shifts.to(dtype=dtype)
                        disp_cells = cell[batch[disp_edge_dst]]
                        disp_shift_vec = torch.einsum("ni,nij->nj", disp_shift, disp_cells)
                        disp_edge_vec = pos[disp_edge_dst] - pos[disp_edge_src] + disp_shift_vec
                    disp_edge_length = disp_edge_vec.norm(dim=1)
                out = out + self.dispersion(
                    disp_feat,
                    pos,
                    batch,
                    cell,
                    edge_src=disp_edge_src,
                    edge_dst=disp_edge_dst,
                    edge_lengths=disp_edge_length,
                    edge_vec=disp_edge_vec,
                    cutoff=disp_cutoff,
                    pbc=self.dispersion_pbc,
                    l2_feats=disp_l2,
                )

        if return_combined_features:
            combined_features = torch.cat(layer_states, dim=-1)
            if return_reciprocal_source:
                rs = reciprocal_source if reciprocal_source is not None else out.new_empty((out.size(0), 0))
                return out, combined_features, rs
            return out, combined_features
        if return_reciprocal_source:
            rs = reciprocal_source if reciprocal_source is not None else out.new_empty((out.size(0), 0))
            return out, rs
        return out
