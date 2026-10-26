"""MACE architecture construction and graph orchestration."""
from __future__ import annotations
import math
from numbers import Integral
from typing import Dict, List
import opt_einsum_fx
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from ictc_o3.electrostatics.mesh import normalize_mesh_size
from ictc_o3.basis.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
from ictc_o3.basis.ictd_irreps import EquivariantChannelLinearSO3
from ictc_o3.basis.ictd_irreps import EquivariantChannelLinearSO3Rect
from ictc_o3.basis.ictd_irreps import HarmonicElementwiseProduct
from ictc_o3.basis.ictd_irreps import HarmonicPathWeightedTensorProduct
from ictc_o3.basis.ictd_irreps import direction_harmonics_all
from ictc_o3.basis.ictd_irreps import ictd_u_matrix_so3
from mace_ictc.models.radial_basis import mace_radial_embedding, mace_polynomial_cutoff
from mace_ictc.models.pure_cartesian_ictd_layers import EquivariantScalarReadoutSO3
from mace_ictc.models.pure_cartesian_ictd_layers import SO3BlockRMSNorm
from mace_ictc.models.pure_cartesian_ictd_layers import _irreps_total_dim
from ictc_o3.basis.layout import _merge_irreps
from ictc_o3.basis.layout import _split_irreps
from mace_ictc.models.pure_cartesian_ictd_layers import apply_channel_adapter_per_l
from mace_ictc.models.pure_cartesian_ictd_layers import resolve_save_multiple_mix_channels
from ictc_o3.ops._mace_symmetric_contraction import MaceSymmetricContraction
from mace_ictc.models.mlp import MainNet
from mace_ictc.utils.scatter import scatter
from mace_ictc.models.long_range import build_long_range_module
from mace_ictc.utils.legacy_dispersion import normalize_dispersion_mode
from ictc_o3.basis.bridge import _hidden_irreps
from ictc_o3.basis.bridge import _so3_flat_to_mace_features
from ictc_o3.basis.bridge import _merge_blocks_subset
from ictc_o3.basis.bridge import SO3ToE3NNBasisBridge
from ictc_o3.ops.symmetric_contraction import ICTDBridgeUSymmetricContractionSO3

_CONTRACTION_BATCH_EXAMPLE = 10
_CONTRACTION_ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]

from .common import _add_product_self_connection, _concat_so3_states_by_l, _cueq_o3_e3nn_group, _cueq_o3_irreps, _init_contraction_basis_logits_, _init_contraction_basis_weight_, _init_contraction_path_weight_, _init_element_conditioned_identity_, _init_element_conditioned_mace_style_, _init_linear_identity_, _init_mace_style_linear_, _init_path_tp_weight_to_one_, _init_so3_linear_identity_, _init_so3_linear_mace_style_, _mace_like_conv_tp_path_scales, _node_component_message_linear, _node_type_indices, _radial_mlp_widths, _resolve_internal_compute_dtype, _so3_block_rmsnorm, _tp_allowed_paths_from_target_lmax, _tp_allowed_paths_to_output_l
from .layers import CueqConvTensorProductSO3, ElementConditionedLinearSO3, ICTDResidualInteractionBlock, LowRankElementConditionedLinearSO3, PathPreservingLinearSO3, PerLScaleSO3, _Normalize2MomSiLU
from .products import CueqMACEProductBasisBlockSO3, CueqMaceSymmetricContractionSO3, ICTDBridgeUProductBasisBlockSO3, ICTDProductBasisBlock, ICTDPureUProductBasisBlockSO3, ICTDPureUSymmetricContractionSO3, ICTDScalarProductBasisBlock, ICTDScalarSymmetricContractionSO3, ICTDSymmetricContractionSO3, NativeMACEProductBasisBlockSO3, _ICTDPureUContraction
from .heads import MACEStyleScalarReadoutSO3, multipole_feature_tap


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

        # Retain disabled historical metadata; trained dispersion checkpoints fail explicitly.
        self.long_range_dispersion_mode = normalize_dispersion_mode(
            long_range_dispersion=long_range_dispersion,
            long_range_dispersion_mode=long_range_dispersion_mode,
        )
        self.long_range_dispersion = False
        self.dispersion = None
        for name, value in tuple(locals().items()):
            if name.startswith(("dispersion_", "mbd_")):
                setattr(self, name, value)
        self.dispersion_pbc = str(long_range_boundary) == "periodic"
        self.long_range_mbd_source_enabled = False
        self.long_range_mbd_source_channels = 0
        self.long_range_mbd_source_offset = 0

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
        from ictc_o3.basis.mace import orthogonal_Q_blocks
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
        from ictc_o3.basis.mace import orthogonal_Q_blocks
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
        from ictc_o3.basis.mace import to_mace_basis as _to_mace
        return _to_mace(x, self.lmax)

    def to_ictd_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`to_mace_basis` (original-MACE/e3nn -> ICTC)."""
        from ictc_o3.basis.mace import to_ictd_basis as _to_ictd
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
            from ictc_o3.ops.generated_direction_harmonics import generated_direction_harmonics
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
        last_full_so3_features: torch.Tensor | None = None
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
                last_full_so3_features = message  # last-layer interaction output (pre-product, full SO3, 2-hop)
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
            mp_feat = multipole_feature_tap(
                last_full_so3_features,
                owned_count=local_atom_count if long_range_energy_callback is not None else None,
            )
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

ICTDMACEUSymmetricContractionSO3 = ICTDBridgeUSymmetricContractionSO3
ICTDMACEUProductBasisBlockSO3 = ICTDBridgeUProductBasisBlockSO3

