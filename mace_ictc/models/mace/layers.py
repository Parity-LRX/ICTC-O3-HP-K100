"""Element-conditioned linears and MACE interaction layers."""
from __future__ import annotations
import math
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from ictc_o3.basis.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
from ictc_o3.basis.ictd_irreps import EquivariantChannelLinearSO3
from mace_ictc.models.pure_cartesian_ictd_layers import SO3BlockRMSNorm
from mace_ictc.models.pure_cartesian_ictd_layers import _irreps_total_dim
from ictc_o3.basis.layout import _merge_irreps
from ictc_o3.basis.layout import _split_irreps
from mace_ictc.utils.scatter import scatter



from .common import _cueq_o3_e3nn_group, _init_element_conditioned_identity_, _init_element_conditioned_mace_style_, _init_mace_style_linear_, _init_path_tp_weight_to_one_, _init_so3_linear_identity_, _init_so3_linear_mace_style_, _mace_like_conv_tp_path_scales, _node_component_message_linear, _radial_mlp_widths, _tp_allowed_paths_from_target_lmax


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
            from ictc_o3.ops.species_linear import species_groups
            from ictc_o3.ops.species_linear import grouped_species_linear
            groups = species_groups(idx, self.num_elements)
        for l in range(self.lmax + 1):
            weight = self.weights[str(l)].to(dtype=x.dtype, device=x.device)
            if groups is not None:
                bias = None if self.bias is None else self.bias[str(l)].to(dtype=x.dtype, device=x.device)
                out_blocks[l] = grouped_species_linear(blocks[l], weight, idx, bias, groups=groups)
                continue
            if getattr(self,"_shared_species_gemm",False):
                from ictc_o3.ops.species_linear import shared_species_linear
                from ictc_o3.ops.species_linear import supported
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
            from ictc_o3.ops.fused_activation import normalized_silu
            return normalized_silu(x,self._scale)
        return F.silu(x) * self._scale


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

        from ictc_o3.basis.mace import orthogonal_Q_blocks

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
            from ictc_o3.ops.streamed_gate_message import streamed_gate_message
            if receiver_rows is not None and compact_message_output:
                receiver_map = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_dst.device)
                receiver_map[receiver_rows] = torch.arange(receiver_rows.numel(), device=edge_dst.device)
                fused_message_blocks=streamed_gate_message(self.tp,x1,edge_attrs,edge_feats,self.fc,
                    edge_src,receiver_map[edge_dst],edge_mask,output_nodes=receiver_rows.numel())
                selected_message_output = fused_message_blocks is not None
            if fused_message_blocks is None:
                fused_message_blocks=streamed_gate_message(self.tp,x1,edge_attrs,edge_feats,self.fc,edge_src,edge_dst,edge_mask)
        if fused_message_blocks is None:
            from ictc_o3.autograd.radial_checkpoint import radial_apply
            gates = radial_apply(self.fc,edge_feats,checkpoint=getattr(self.tp,'_radial_checkpoint',False),
                gemm_policy=getattr(self.tp,'_training_gate_gemm_policy','torch'),
                chunk_rows=getattr(self.tp,'_radial_chunk_rows',0))
        else:
            gates = None
        if (getattr(self, "_edge_fusion", "reference") in ("triton", "cuda", "cuda_projection", "triton_projection", "triton_csr", "triton_remat")
                and self.tp_cueq is None and self.interaction_attn_heads == 0
                and edge_block_callback is None and fused_message_blocks is None):
            from ictc_o3.ops.fused_message import aggregate_message
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
