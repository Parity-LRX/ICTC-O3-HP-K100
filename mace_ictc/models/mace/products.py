"""MACE product-block assembly around shared O3 contractions."""
from __future__ import annotations
from typing import Dict
import opt_einsum_fx
import torch
import torch.nn as nn
from e3nn import o3
from ictc_o3.basis.ictd_irreps import EquivariantChannelLinearSO3
from ictc_o3.basis.ictd_irreps import EquivariantChannelLinearSO3Rect
from ictc_o3.basis.ictd_irreps import HarmonicPathWeightedTensorProduct
from ictc_o3.basis.ictd_irreps import ictd_u_matrix_so3
from ictc_o3.basis.layout import _merge_irreps
from ictc_o3.basis.layout import _split_irreps
from mace_ictc.models.pure_cartesian_ictd_layers import apply_channel_adapter_per_l
from ictc_o3.ops._mace_symmetric_contraction import MaceSymmetricContraction
from ictc_o3.basis.bridge import _hidden_irreps
from ictc_o3.basis.bridge import _so3_flat_to_mace_features
from ictc_o3.basis.bridge import SO3ToE3NNBasisBridge
from ictc_o3.ops.symmetric_contraction import ICTDBridgeUSymmetricContractionSO3

_CONTRACTION_BATCH_EXAMPLE = 10
_CONTRACTION_ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]

from .common import _add_product_self_connection, _cueq_o3_e3nn_group, _cueq_o3_irreps, _init_contraction_basis_logits_, _init_contraction_basis_weight_, _init_contraction_path_weight_, _init_linear_identity_, _init_path_tp_weight_to_one_, _init_so3_linear_identity_, _node_type_indices, _tp_allowed_paths_to_output_l


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

ICTDMACEUSymmetricContractionSO3 = ICTDBridgeUSymmetricContractionSO3
ICTDMACEUProductBasisBlockSO3 = ICTDBridgeUProductBasisBlockSO3

