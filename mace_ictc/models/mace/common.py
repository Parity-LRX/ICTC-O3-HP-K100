"""Model initialization and representation helpers."""
from __future__ import annotations
import math
from numbers import Integral
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from ictc_o3.basis.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
from ictc_o3.basis.ictd_irreps import direction_harmonics_all
from ictc_o3.basis.layout import _merge_irreps
from ictc_o3.basis.layout import _split_irreps
from ictc_o3.basis.bridge import _hidden_irreps






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
    from ictc_o3.basis.mace import orthogonal_Q_blocks

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


def _cueq_o3_e3nn_group():
    try:
        from mace.tools.cg import O3_e3nn
    except Exception:
        try:
            from ictc_o3.basis._mace_cg import O3_e3nn
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
