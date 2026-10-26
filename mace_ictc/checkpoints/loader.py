"""Checkpoint reconstruction shared by inference, conversion and simulation."""
from __future__ import annotations

import os
from typing import List, Optional
import torch
from mace_ictc.utils.mesh import normalize_mesh_size
from .adapters import AtomForcesWrapper, _maybe_torchscript_trace_model
# Baseline build: PureCartesianICTDFix + its baseline layer are always available.
from mace_ictc.models.pure_cartesian_ictd_layers import (
    PhysicalTensorICTDRecovery,
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerSave,
)
from mace_ictc.models.pure_cartesian_ictd_fix import PureCartesianICTDFix

# Variant backbones (spherical / sparse / o3) are NOT shipped in this baseline-only build.
# Their imports are guarded so `from_checkpoint` still loads `pure-cartesian-ictd-fix`
# checkpoints; loading a non-baseline checkpoint raises a clear error at its dispatch branch.
try:
    from mace_ictc.models import E3_TransformerLayer_multi
except Exception:
    E3_TransformerLayer_multi = None
try:
    from mace_ictc.models.e3nn_layers_channelwise import (
        E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
    )
except Exception:
    E3_TransformerLayer_multi_channelwise = None
try:
    from mace_ictc.models.pure_cartesian_sparse_layers import (
        PureCartesianSparseTransformerLayer,
    )
except Exception:
    PureCartesianSparseTransformerLayer = None
try:
    from mace_ictc.models.pure_cartesian_sparse_layers_save import (
        PureCartesianSparseTransformerLayerSave,
    )
except Exception:
    PureCartesianSparseTransformerLayerSave = None
try:
    from mace_ictc.models.pure_cartesian_ictd_layers_full import PureCartesianICTDTransformerLayer
except Exception:
    PureCartesianICTDTransformerLayer = None
try:
    from mace_ictc.models.pure_cartesian_ictd_layers_full_o3 import PureCartesianICTDO3TransformerLayer
except Exception:
    PureCartesianICTDO3TransformerLayer = None
try:
    from mace_ictc.models.pure_cartesian_ictd_layers_o3 import PureCartesianICTDSaveO3TransformerLayer
except Exception:
    PureCartesianICTDSaveO3TransformerLayer = None
from mace_ictc.utils.config import ModelConfig
from mace_ictc.utils.checkpoint_metadata import (
    derive_long_range_far_max_radius_multiplier,
    get_checkpoint_e3_state_dict,
    infer_external_tensor_rank_from_state_dict,
    infer_ictd_save_final_readout_mode_from_state_dict,
    infer_ictd_save_multiple_fusion_scheme_from_state_dict,
    infer_ictd_save_multiple_order_from_state_dict,
    infer_physical_tensor_outputs_from_state_dict,
    validate_dispersion_deployment_graph_rule,
    validate_dispersion_train_deploy_graph_compatibility,
    validate_dispersion_training_graph_rule,
)
from mace_ictc.utils.external_tensor_specs import normalize_external_tensor_specs
from mace_ictc.models.zbl import maybe_wrap_model_with_zbl
from mace_ictc.utils.tensor_utils import map_tensor_values


class LoadedPotential:
    """Loaded model and E0 buffers, independent of any simulation engine.

    ``wrapper`` retains the established export ABI. Construction preserves the
    inference freeze, dtype and device behavior of historical checkpoint loads.
    """
    def __init__(self, model, element_types, max_radius, atomic_energy_keys,
                 atomic_energy_values, device="cpu", dtype=torch.float64):
        self.device, self.dtype = device, dtype
        self.element_types, self.max_radius = list(element_types), max_radius
        self.wrapper = AtomForcesWrapper(model, atomic_energy_keys, atomic_energy_values)
        self.wrapper = self.wrapper.to(dtype=dtype).to(device)

    @property
    def model(self):
        return self.wrapper.model


def load_checkpoint(
    checkpoint_path: str,
    element_types: List[str],
    max_radius: Optional[float] = None,
    atomic_energy_keys: Optional[List[int]] = None,
    atomic_energy_values: Optional[List[float]] = None,
    device: str = "cpu",
    embed_size: Optional[List[int]] = None,
    output_size: Optional[int] = None,
    tensor_product_mode: Optional[str] = None,
    num_interaction: Optional[int] = None,
    ictd_tp_path_policy: Optional[str] = None,
    ictd_tp_max_rank_other: Optional[int] = None,
    avg_num_neighbors: Optional[float] = None,
    torchscript: bool = False,
    force_naive: bool = False,
    ictd_tp_backend: Optional[str] = None,
    operator_plan: str = "auto",
    generated_layout: str = "packed",
    *, _potential_class=LoadedPotential,
) -> "LoadedPotential":
    """Reconstruct a checkpoint without importing a simulation interface.

    Supports tensor_product_mode from checkpoint or argument:
    - "spherical": E3_TransformerLayer_multi (e3nn_layers)
    - "spherical-save": E3_TransformerLayer_multi_channelwise (e3nn_layers_channelwise)
    - "spherical-save-cue": E3_TransformerLayer_multi (cue_layers_channelwise, cuEquivariance GPU)
    - "pure-cartesian-sparse": PureCartesianSparseTransformerLayer
    - "pure-cartesian-sparse-save": PureCartesianSparseTransformerLayerSave
    - "pure-cartesian-ictd": PureCartesianICTDTransformerLayer (pure_cartesian_ictd_layers_full)
    - "pure-cartesian-ictd-save": PureCartesianICTDTransformerLayer (pure_cartesian_ictd_layers)
    - "pure-cartesian-ictd-save-multiple": PureCartesianICTDTransformerLayer (pure_cartesian_ictd_layers, multi-branch contraction)
    - "pure-cartesian-ictd-save-o3": PureCartesianICTDSaveO3TransformerLayer (pure_cartesian_ictd_layers_o3)
    - "pure-cartesian-ictd-fix": PureCartesianICTDFix (pure_cartesian_ictd_fix)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    selected_state_dict, state_source = get_checkpoint_e3_state_dict(ckpt)
    if state_source == "ema":
        print("[LAMMPS_MLIAP_MFF] Using EMA weights from checkpoint")
    elif state_source == "swa":
        print("[LAMMPS_MLIAP_MFF] Using SWA weights from checkpoint")

    arch_meta = ckpt.get("model_hyperparameters", {})

    dtype_raw = ckpt.get("dtype", arch_meta.get("dtype", torch.float64))
    if isinstance(dtype_raw, str):
        dtype = torch.float64 if dtype_raw in ("float64", "double") else torch.float32
    else:
        dtype = dtype_raw

    mode = str(tensor_product_mode or ckpt.get("tensor_product_mode", "spherical"))
    # Use max_radius from checkpoint if saved; otherwise use CLI override; finally fall back to 5.0.
    radius = float(ckpt.get("max_radius", max_radius if max_radius is not None else 5.0))
    if "max_radius" in ckpt:
        print(f"[LAMMPS_MLIAP_MFF] 使用 checkpoint 中的 max_radius: {radius:.2f} Å")
    config = ModelConfig(
        dtype=dtype,
        channel_in=int(arch_meta.get("channel_in", 64)),
        channel_in2=int(arch_meta.get("channel_in2", 32)),
        channel_in3=int(arch_meta.get("channel_in3", 32)),
        channel_in4=int(arch_meta.get("channel_in4", 32)),
        channel_in5=int(arch_meta.get("channel_in5", 32)),
        max_atomvalue=int(arch_meta.get("max_atomvalue", 10)),
        embedding_dim=int(arch_meta.get("embedding_dim", 16)),
        main_hidden_sizes3=list(arch_meta.get("main_hidden_sizes3", [64, 32])),
        embed_size=embed_size if embed_size is not None else list(arch_meta.get("embed_size", [128, 128, 128])),
        output_size=int(output_size if output_size is not None else arch_meta.get("output_size", 8)),
        irreps_output_conv_channels=arch_meta.get("irreps_output_conv_channels"),
        lmax=int(arch_meta.get("lmax", 2)),
        function_type=str(arch_meta.get("function_type", "gaussian")),
        num_layers=int(arch_meta.get("num_layers", 1)),
        number_of_basis=int(arch_meta.get("number_of_basis", 8)),
        number_of_basis_main=int(arch_meta.get("number_of_basis_main", 8)),
        emb_number_main_2=list(arch_meta.get("emb_number_main_2", [64, 64, 64])),
        max_radius=radius,
        max_radius_main=float(arch_meta.get("max_radius_main", radius)),
    )
    if num_interaction is None:
        num_interaction = int(arch_meta.get("num_interaction", 2))
    invariant_channels = int(arch_meta.get("invariant_channels", 32))

    if atomic_energy_keys is not None and atomic_energy_values is not None:
        aek = torch.tensor(atomic_energy_keys, dtype=torch.long)
        aev = torch.tensor(atomic_energy_values, dtype=dtype)
    elif ckpt.get("atomic_energy_keys") is not None and ckpt.get("atomic_energy_values") is not None:
        aek_raw = ckpt.get("atomic_energy_keys")
        aev_raw = ckpt.get("atomic_energy_values")
        aek = aek_raw.detach().cpu().to(dtype=torch.long) if isinstance(aek_raw, torch.Tensor) else torch.tensor(aek_raw, dtype=torch.long)
        aev = aev_raw.detach().cpu().to(dtype=dtype) if isinstance(aev_raw, torch.Tensor) else torch.tensor(aev_raw, dtype=dtype)
        print("[LAMMPS_MLIAP_MFF] 使用 checkpoint 中的 atomic_energy_keys/atomic_energy_values")
    else:
        config.load_atomic_energies_from_file("fitted_E0.csv")
        aek = config.atomic_energy_keys
        aev = config.atomic_energy_values

    external_tensor_rank = ckpt.get("external_tensor_rank", arch_meta.get("external_tensor_rank"))
    external_tensor_irrep = ckpt.get("external_tensor_irrep", arch_meta.get("external_tensor_irrep"))
    external_tensor_specs = ckpt.get("external_tensor_specs", arch_meta.get("external_tensor_specs"))
    num_fidelity_levels = int(arch_meta.get("num_fidelity_levels", 0) or 0)
    multi_fidelity_mode = str(arch_meta.get("multi_fidelity_mode", "conditioning") or "conditioning")
    o3_irrep_preset = str(arch_meta.get("o3_irrep_preset", "auto"))
    o3_active_irreps = arch_meta.get("o3_active_irreps")
    if external_tensor_rank is None:
        external_tensor_rank = infer_external_tensor_rank_from_state_dict(selected_state_dict)
    external_tensor_specs = normalize_external_tensor_specs(
        external_tensor_specs,
        external_tensor_rank=external_tensor_rank,
        external_tensor_irrep=external_tensor_irrep,
        external_tensor_parity=None,
    )
    physical_tensor_outputs = ckpt.get("physical_tensor_outputs", arch_meta.get("physical_tensor_outputs"))
    if physical_tensor_outputs is None:
        physical_tensor_outputs = infer_physical_tensor_outputs_from_state_dict(selected_state_dict)
    long_range_mode = str(arch_meta.get("long_range_mode", "none"))
    long_range_hidden_dim = int(arch_meta.get("long_range_hidden_dim", 64))
    long_range_boundary = str(arch_meta.get("long_range_boundary", "nonperiodic"))
    long_range_neutralize = bool(arch_meta.get("long_range_neutralize", True))
    long_range_filter_hidden_dim = int(arch_meta.get("long_range_filter_hidden_dim", 64))
    long_range_kmax = int(arch_meta.get("long_range_kmax", 2))
    long_range_mesh_size = normalize_mesh_size(arch_meta.get("long_range_mesh_size", 16))
    long_range_slab_padding_factor = int(arch_meta.get("long_range_slab_padding_factor", 2))
    long_range_include_k0 = bool(arch_meta.get("long_range_include_k0", False))
    long_range_source_channels = int(arch_meta.get("long_range_source_channels", 1))
    long_range_backend = str(arch_meta.get("long_range_backend", "dense_pairwise"))
    long_range_reciprocal_backend = str(arch_meta.get("long_range_reciprocal_backend", "direct_kspace"))
    long_range_energy_partition = str(arch_meta.get("long_range_energy_partition", "potential"))
    long_range_green_mode = str(arch_meta.get("long_range_green_mode", "poisson"))
    long_range_assignment = str(arch_meta.get("long_range_assignment", "pcs"))
    long_range_mesh_fft_full_ewald = bool(arch_meta.get("long_range_mesh_fft_full_ewald", False))
    long_range_mesh_fft_reciprocal_only = bool(arch_meta.get("long_range_mesh_fft_reciprocal_only", False))
    long_range_smearing_sigma = arch_meta.get("long_range_smearing_sigma", None)
    long_range_max_multipole_l = int(arch_meta.get("long_range_max_multipole_l", 0))
    long_range_multipole_gate_init = float(arch_meta.get("long_range_multipole_gate_init", 0.1))
    long_range_dispersion_mode = str(
        arch_meta.get(
            "long_range_dispersion_mode",
            "pairwise-c6" if bool(arch_meta.get("long_range_dispersion", False)) else "none",
        )
    )
    dispersion_cutoff = float(arch_meta.get("dispersion_cutoff", 10.0))
    dispersion_max_num_neighbors_raw = arch_meta.get("dispersion_max_num_neighbors", None)
    dispersion_max_num_neighbors = (
        None
        if dispersion_max_num_neighbors_raw is None or int(dispersion_max_num_neighbors_raw) <= 0
        else int(dispersion_max_num_neighbors_raw)
    )
    dispersion_neighbor_method = str(arch_meta.get("dispersion_neighbor_method", "auto"))
    dispersion_bruteforce_threshold = int(arch_meta.get("dispersion_bruteforce_threshold", 1024))
    dispersion_allow_large_bruteforce_fallback = bool(
        arch_meta.get("dispersion_allow_large_bruteforce_fallback", False)
    )
    dispersion_slq_num_probes = int(ckpt.get("dispersion_slq_num_probes", arch_meta.get("dispersion_slq_num_probes", 8)))
    dispersion_slq_lanczos_steps = int(
        ckpt.get("dispersion_slq_lanczos_steps", arch_meta.get("dispersion_slq_lanczos_steps", 16))
    )
    mbd_operator_backend = str(ckpt.get("mbd_operator_backend", arch_meta.get("mbd_operator_backend", "edge_sparse")))
    raw_dispersion_graph_rule = ckpt.get(
        "dispersion_deployment_graph_rule",
        arch_meta.get("dispersion_deployment_graph_rule", None),
    )
    raw_dispersion_training_graph_rule = ckpt.get(
        "dispersion_training_graph_rule",
        arch_meta.get("dispersion_training_graph_rule", None),
    )
    raw_dispersion_graph_compatibility = ckpt.get(
        "dispersion_train_deploy_graph_compatibility",
        arch_meta.get("dispersion_train_deploy_graph_compatibility", None),
    )
    validate_dispersion_training_graph_rule(
        long_range_dispersion_mode=long_range_dispersion_mode,
        mbd_operator_backend=mbd_operator_backend,
        raw_rule=raw_dispersion_training_graph_rule,
        source_label="checkpoint dispersion_training_graph_rule",
    )
    validate_dispersion_deployment_graph_rule(
        long_range_dispersion_mode=long_range_dispersion_mode,
        mbd_operator_backend=mbd_operator_backend,
        raw_rule=raw_dispersion_graph_rule,
        source_label="checkpoint dispersion_deployment_graph_rule",
    )
    validate_dispersion_train_deploy_graph_compatibility(
        long_range_dispersion_mode=long_range_dispersion_mode,
        mbd_operator_backend=mbd_operator_backend,
        raw_value=raw_dispersion_graph_compatibility,
        source_label="checkpoint dispersion_train_deploy_graph_compatibility",
    )
    mbd_pme_mesh_size = int(ckpt.get("mbd_pme_mesh_size", arch_meta.get("mbd_pme_mesh_size", 16)))
    mbd_pme_assignment = str(ckpt.get("mbd_pme_assignment", arch_meta.get("mbd_pme_assignment", "pcs")))
    mbd_pme_k_norm_floor = float(ckpt.get("mbd_pme_k_norm_floor", arch_meta.get("mbd_pme_k_norm_floor", 1.0e-6)))
    mbd_pme_assignment_window_floor = float(
        ckpt.get("mbd_pme_assignment_window_floor", arch_meta.get("mbd_pme_assignment_window_floor", 1.0e-6))
    )
    mbd_pme_ewald_alpha_prefactor = float(
        ckpt.get("mbd_pme_ewald_alpha_prefactor", arch_meta.get("mbd_pme_ewald_alpha_prefactor", 5.0))
    )
    # Preserve disabled historical metadata; enabled dispersion was rejected above.
    mbd_anisotropic_polarizability = bool(
        ckpt.get("mbd_anisotropic_polarizability", arch_meta.get("mbd_anisotropic_polarizability", False))
    )
    long_range_theta = float(arch_meta.get("long_range_theta", 0.5))
    long_range_leaf_size = int(arch_meta.get("long_range_leaf_size", 32))
    long_range_multipole_order = int(arch_meta.get("long_range_multipole_order", 0))
    long_range_far_source_dim = int(arch_meta.get("long_range_far_source_dim", 16))
    long_range_far_num_shells = int(arch_meta.get("long_range_far_num_shells", 3))
    long_range_far_shell_growth = float(arch_meta.get("long_range_far_shell_growth", 2.0))
    long_range_far_tail = bool(arch_meta.get("long_range_far_tail", True))
    long_range_far_tail_bins = int(arch_meta.get("long_range_far_tail_bins", 2))
    long_range_far_stats = str(arch_meta.get("long_range_far_stats", "mean,count,mean_r,rms_r"))
    if arch_meta.get("long_range_far_max_radius_multiplier") is None:
        long_range_far_max_radius_multiplier = derive_long_range_far_max_radius_multiplier(
            long_range_far_num_shells,
            long_range_far_shell_growth,
        )
    else:
        long_range_far_max_radius_multiplier = float(arch_meta.get("long_range_far_max_radius_multiplier"))
    long_range_far_source_norm = bool(arch_meta.get("long_range_far_source_norm", True))
    long_range_far_gate_init = float(arch_meta.get("long_range_far_gate_init", 0.0))
    feature_spectral_mode = str(arch_meta.get("feature_spectral_mode", "none"))
    feature_spectral_bottleneck_dim = int(arch_meta.get("feature_spectral_bottleneck_dim", 8))
    feature_spectral_mesh_size = int(arch_meta.get("feature_spectral_mesh_size", 16))
    feature_spectral_filter_hidden_dim = int(arch_meta.get("feature_spectral_filter_hidden_dim", 64))
    feature_spectral_boundary = str(arch_meta.get("feature_spectral_boundary", "periodic"))
    feature_spectral_slab_padding_factor = int(arch_meta.get("feature_spectral_slab_padding_factor", 2))
    feature_spectral_neutralize = bool(arch_meta.get("feature_spectral_neutralize", True))
    feature_spectral_include_k0 = bool(arch_meta.get("feature_spectral_include_k0", False))
    feature_spectral_assignment = str(arch_meta.get("feature_spectral_assignment", "pcs"))
    feature_spectral_gate_init = float(arch_meta.get("feature_spectral_gate_init", 0.0))
    save_contraction_order = int(
        ckpt.get("ictd_save_contraction_order")
        or arch_meta.get("ictd_save_contraction_order")
        or arch_meta.get("save_contraction_order")
        or infer_ictd_save_multiple_order_from_state_dict(selected_state_dict)
        or 3
    )
    save_multiple_fusion_scheme = str(
        ckpt.get("ictd_save_multiple_fusion_scheme")
        or arch_meta.get("ictd_save_multiple_fusion_scheme")
        or arch_meta.get("save_multiple_fusion_scheme")
        or infer_ictd_save_multiple_fusion_scheme_from_state_dict(selected_state_dict)
        or "serial_lastmix"
    )
    save_final_readout_mode = str(
        ckpt.get("ictd_save_final_readout_mode")
        or arch_meta.get("ictd_save_final_readout_mode")
        or arch_meta.get("save_final_readout_mode")
        or infer_ictd_save_final_readout_mode_from_state_dict(selected_state_dict)
        or "direct-1"
    )
    save_multiple_mix_channels = (
        ckpt.get("ictd_save_multiple_mix_channels")
        or arch_meta.get("ictd_save_multiple_mix_channels")
        or arch_meta.get("save_multiple_mix_channels")
    )
    if save_multiple_mix_channels is not None:
        save_multiple_mix_channels = int(save_multiple_mix_channels)

    if mode == "pure-cartesian-ictd-fix":
        ictd_tp_path_policy = ictd_tp_path_policy or ckpt.get("ictd_tp_path_policy") or arch_meta.get("ictd_tp_path_policy", "full")
        ictd_tp_max_rank_other = (
            ictd_tp_max_rank_other
            if ictd_tp_max_rank_other is not None
            else ckpt.get("ictd_tp_max_rank_other", arch_meta.get("ictd_tp_max_rank_other"))
        )
        has_fusion_mix = any(k.startswith("multiple_contraction_mix") for k in selected_state_dict)
        has_softmax_heads = "fusion_head_logits" in selected_state_dict
        has_free_heads = "fusion_head_weights" in selected_state_dict
        if has_softmax_heads:
            inferred_fusion_heads = int(selected_state_dict["fusion_head_logits"].numel())
            inferred_head_mode = "softmax"
        elif has_free_heads:
            inferred_fusion_heads = int(selected_state_dict["fusion_head_weights"].numel())
            inferred_head_mode = "free"
        else:
            inferred_fusion_heads = 1
            inferred_head_mode = "softmax"
        # Equivariant neighbor-attention (commit 1319eba) is opt-in and post-dates the original
        # from_checkpoint wiring; without reconstructing it, strict load fails on the attn_* keys.
        # Prefer the saved hyperparameter, else infer the head count from attn_z_bias_raw's length
        # (per-layer shape [H]). Defaults to 0 -> no attention modules (unchanged for old ckpts).
        inferred_attn_heads = int(
            ckpt.get("ictd_fix_interaction_attn_heads")
            or arch_meta.get("ictd_fix_interaction_attn_heads")
            or (selected_state_dict["interactions.0.attn_z_bias_raw"].numel()
                if "interactions.0.attn_z_bias_raw" in selected_state_dict else 0)
        )
        # avg_num_neighbors normalizes the messages (model divides by it, pure_cartesian_ictd_fix
        # ~line 1666) so the weights are trained UNDER it -- but it is a plain Python float, NOT a
        # state_dict buffer, so a wrong value loads SILENTLY (strict load can't catch it). For
        # ictd-fix it is auto-computed from the training data, so the legacy 14.38 fallback is
        # almost always wrong. Require it (explicit arg > ckpt > arch_meta), else warn loudly.
        resolved_avg_nn = avg_num_neighbors
        if resolved_avg_nn is None:
            resolved_avg_nn = ckpt.get("avg_num_neighbors") or arch_meta.get("avg_num_neighbors")
        if resolved_avg_nn is None:
            import warnings
            warnings.warn(
                "[LAMMPS_MLIAP_MFF] avg_num_neighbors is NOT in the checkpoint and was not passed "
                "explicitly; falling back to 14.38. The model divides messages by this constant, so "
                "if training used a different (auto-computed) value the deployed energies/forces are "
                "WRONG. Pass avg_num_neighbors=<the training value> (logged as 'Computed average "
                "number of neighbors').",
                RuntimeWarning,
            )
            resolved_avg_nn = 14.38
        resolved_avg_nn = float(resolved_avg_nn)

        def _scalar_meta(name: str, default: float) -> float:
            if name in ckpt:
                return float(ckpt[name])
            if name in arch_meta:
                return float(arch_meta[name])
            if name in selected_state_dict:
                v = selected_state_dict[name]
                return float(v.detach().cpu().item() if torch.is_tensor(v) else v)
            return float(default)

        energy_output_scale_enabled = bool(
            ("energy_output_scale" in selected_state_dict)
            or ckpt.get(
                "energy_output_scale_enabled",
                arch_meta.get("energy_output_scale_enabled", False),
            )
        )
        energy_output_shift_enabled = bool(
            ("energy_output_shift" in selected_state_dict)
            or ckpt.get(
                "energy_output_shift_enabled",
                arch_meta.get("energy_output_shift_enabled", False),
            )
        )
        angular_basis = str(
            ckpt.get("angular_basis")
            or arch_meta.get("angular_basis", "ictd")
        )
        atomic_numbers = aek.detach().cpu().to(dtype=torch.long).tolist() if aek is not None else None
        model = PureCartesianICTDFix(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            atomic_numbers=atomic_numbers,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            radial_mlp_hidden=ckpt.get("radial_mlp_hidden", arch_meta.get("radial_mlp_hidden")),
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=config.internal_compute_dtype,
            # FSCETP checkpoints were trained WITH the sqrt(num_basis) radial scale -> default
            # True here for back-compat (the model default is False = byte-literal MACE radial).
            radial_sqrt_num_basis=bool(
                ckpt.get(
                    "radial_sqrt_num_basis",
                    arch_meta.get("radial_sqrt_num_basis", True),
                )
            ),
            polynomial_cutoff_p=(
                None
                if (ckpt.get("polynomial_cutoff_p", arch_meta.get("polynomial_cutoff_p", 6)) is None)
                else int(ckpt.get("polynomial_cutoff_p", arch_meta.get("polynomial_cutoff_p", 6)))
            ),
            ictd_save_tp_mode=str(
                ckpt.get("ictd_save_tp_mode")
                or arch_meta.get("ictd_save_tp_mode", "fully-connected")
            ),
            ictd_fix_route=str(
                ckpt.get("ictd_fix_route")
                or arch_meta.get("ictd_fix_route", "fusion" if has_fusion_mix else "baseline")
            ),
            ictd_fix_contraction_combine=str(
                ckpt.get("ictd_fix_contraction_combine")
                or arch_meta.get("ictd_fix_contraction_combine", "softmax")
            ),
            ictd_fix_product_backend=str(
                ckpt.get("ictd_fix_product_backend")
                or arch_meta.get("ictd_fix_product_backend", "ictd-pure-u")
            ),
            angular_basis=angular_basis,
            ictd_tp_backend=str(
                ictd_tp_backend
                or ckpt.get("ictd_tp_backend")
                or arch_meta.get("ictd_tp_backend", "pytorch")
            ),
            ictd_fix_use_reduced_cg=bool(
                ckpt.get(
                    "ictd_fix_use_reduced_cg",
                    arch_meta.get("ictd_fix_use_reduced_cg", False),
                )
            ),
            ictd_fix_first_layer_self_connection=bool(
                ckpt.get(
                    "ictd_fix_first_layer_self_connection",
                    arch_meta.get("ictd_fix_first_layer_self_connection", False),
                )
            ),
            ictd_fix_self_connection_rank=int(
                ckpt.get(
                    "ictd_fix_self_connection_rank",
                    arch_meta.get("ictd_fix_self_connection_rank", 0),
                )
            ),
            ictd_fix_conv_tp_scale_init=str(
                ckpt.get("ictd_fix_conv_tp_scale_init")
                or arch_meta.get("ictd_fix_conv_tp_scale_init", "none")
            ),
            ictd_fix_freeze_conv_tp_weight=bool(
                ckpt.get(
                    "ictd_fix_freeze_conv_tp_weight",
                    arch_meta.get("ictd_fix_freeze_conv_tp_weight", False),
                )
            ),
            ictd_fix_interaction_init=str(
                ckpt.get("ictd_fix_interaction_init")
                or arch_meta.get("ictd_fix_interaction_init", "identity")
            ),
            ictd_fix_readout_hidden_channels=int(
                ckpt.get("ictd_fix_readout_hidden_channels")
                or arch_meta.get("ictd_fix_readout_hidden_channels", 16)
            ),
            ictd_fix_edge_lmax=(
                None
                if ckpt.get("ictd_fix_edge_lmax", arch_meta.get("ictd_fix_edge_lmax", None)) is None
                else int(ckpt.get("ictd_fix_edge_lmax", arch_meta.get("ictd_fix_edge_lmax", None)))
            ),
            ictd_fix_interaction_scale=str(
                ckpt.get("ictd_fix_interaction_scale")
                or arch_meta.get("ictd_fix_interaction_scale", "none")
            ),
            ictd_fix_fusion_scale_init=float(
                ckpt.get("ictd_fix_fusion_scale_init")
                or arch_meta.get("ictd_fix_fusion_scale_init", 0.1)
            ),
            ictd_fix_fusion_heads=int(
                ckpt.get("ictd_fix_fusion_heads")
                or arch_meta.get("ictd_fix_fusion_heads")
                or inferred_fusion_heads
            ),
            ictd_fix_fusion_head_weight_mode=str(
                ckpt.get("ictd_fix_fusion_head_weight_mode")
                or arch_meta.get("ictd_fix_fusion_head_weight_mode", inferred_head_mode)
            ),
            ictd_fix_fusion_input_scale_init=float(
                ckpt.get("ictd_fix_fusion_input_scale_init")
                or arch_meta.get("ictd_fix_fusion_input_scale_init", 1.0)
            ),
            ictd_fix_fusion_input_scale_trainable=(
                "fusion_input_scales" in selected_state_dict
                or bool(arch_meta.get("ictd_fix_fusion_input_scale_trainable", False))
            ),
            ictd_fix_gmix_gate_init=float(
                ckpt.get("ictd_fix_gmix_gate_init")
                or arch_meta.get("ictd_fix_gmix_gate_init", 1.0)
            ),
            ictd_fix_gmix_gate_trainable=(
                "g_mix_gate" in selected_state_dict
                or bool(arch_meta.get("ictd_fix_gmix_gate_trainable", False))
            ),
            ictd_fix_gmix_block_rmsnorm=(
                "gmix_block_rmsnorm_gamma" in selected_state_dict
                or bool(arch_meta.get("ictd_fix_gmix_block_rmsnorm", False))
            ),
            ictd_fix_gmix_block_rmsnorm_gamma_init=float(
                ckpt.get("ictd_fix_gmix_block_rmsnorm_gamma_init")
                or arch_meta.get("ictd_fix_gmix_block_rmsnorm_gamma_init", 1.0)
            ),
            ictd_fix_readout_head_scale_init=float(
                ckpt.get("ictd_fix_readout_head_scale_init")
                or arch_meta.get("ictd_fix_readout_head_scale_init", 1.0)
            ),
            ictd_fix_readout_head_scale_trainable=(
                "readout_head_scales" in selected_state_dict
                or bool(arch_meta.get("ictd_fix_readout_head_scale_trainable", False))
            ),
            ictd_fix_fusion_readout_mixed_channels=bool(
                ckpt.get(
                    "ictd_fix_fusion_readout_mixed_channels",
                    arch_meta.get("ictd_fix_fusion_readout_mixed_channels", False),
                )
            ),
            ictd_fix_interaction_attn_heads=inferred_attn_heads,
            save_contraction_order=save_contraction_order,
            save_multiple_mix_channels=save_multiple_mix_channels,
            avg_num_neighbors=resolved_avg_nn,
            energy_output_scale_enabled=energy_output_scale_enabled,
            energy_output_scale=_scalar_meta("energy_output_scale", 1.0),
            energy_output_shift_enabled=energy_output_shift_enabled,
            energy_output_shift=_scalar_meta("energy_output_shift", 0.0),
            device=torch.device(device),
            long_range_mode=long_range_mode,
            long_range_hidden_dim=long_range_hidden_dim,
            long_range_boundary=long_range_boundary,
            long_range_neutralize=long_range_neutralize,
            long_range_filter_hidden_dim=long_range_filter_hidden_dim,
            long_range_kmax=long_range_kmax,
            long_range_mesh_size=long_range_mesh_size,
            long_range_slab_padding_factor=long_range_slab_padding_factor,
            long_range_include_k0=long_range_include_k0,
            long_range_source_channels=long_range_source_channels,
            long_range_backend=long_range_backend,
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
            long_range_mesh_fft_reciprocal_only=long_range_mesh_fft_reciprocal_only,
            long_range_smearing_sigma=long_range_smearing_sigma,
            long_range_max_multipole_l=long_range_max_multipole_l,
            long_range_multipole_gate_init=long_range_multipole_gate_init,
            long_range_dispersion_mode=long_range_dispersion_mode,
            dispersion_cutoff=dispersion_cutoff,
            dispersion_max_num_neighbors=dispersion_max_num_neighbors,
            dispersion_neighbor_method=dispersion_neighbor_method,
            dispersion_bruteforce_threshold=dispersion_bruteforce_threshold,
            dispersion_allow_large_bruteforce_fallback=dispersion_allow_large_bruteforce_fallback,
            dispersion_slq_num_probes=dispersion_slq_num_probes,
            dispersion_slq_lanczos_steps=dispersion_slq_lanczos_steps,
            mbd_operator_backend=mbd_operator_backend,
            mbd_pme_mesh_size=mbd_pme_mesh_size,
            mbd_pme_assignment=mbd_pme_assignment,
            mbd_pme_k_norm_floor=mbd_pme_k_norm_floor,
            mbd_pme_assignment_window_floor=mbd_pme_assignment_window_floor,
            mbd_pme_ewald_alpha_prefactor=mbd_pme_ewald_alpha_prefactor,
            mbd_anisotropic_polarizability=mbd_anisotropic_polarizability,
            long_range_theta=long_range_theta,
            long_range_leaf_size=long_range_leaf_size,
            long_range_multipole_order=long_range_multipole_order,
            long_range_far_source_dim=long_range_far_source_dim,
            long_range_far_num_shells=long_range_far_num_shells,
            long_range_far_shell_growth=long_range_far_shell_growth,
            long_range_far_tail=long_range_far_tail,
            long_range_far_tail_bins=long_range_far_tail_bins,
            long_range_far_stats=long_range_far_stats,
            long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            long_range_far_source_norm=long_range_far_source_norm,
            long_range_far_gate_init=long_range_far_gate_init,
            feature_spectral_mode=feature_spectral_mode,
            feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=feature_spectral_boundary,
            feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=feature_spectral_neutralize,
            feature_spectral_include_k0=feature_spectral_include_k0,
            feature_spectral_assignment=feature_spectral_assignment,
            feature_spectral_gate_init=feature_spectral_gate_init,
        ).to(device)
    elif mode == "pure-cartesian-ictd":
        model = PureCartesianICTDTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            physical_tensor_outputs=physical_tensor_outputs,
            external_tensor_rank=external_tensor_rank,
            external_tensor_irrep=external_tensor_irrep,
            external_tensor_specs=external_tensor_specs,
            num_fidelity_levels=num_fidelity_levels,
            multi_fidelity_mode=multi_fidelity_mode,
        internal_compute_dtype=config.internal_compute_dtype,
            device=torch.device(device),
            long_range_mode=long_range_mode,
            long_range_hidden_dim=long_range_hidden_dim,
            long_range_boundary=long_range_boundary,
            long_range_neutralize=long_range_neutralize,
            long_range_filter_hidden_dim=long_range_filter_hidden_dim,
            long_range_kmax=long_range_kmax,
            long_range_mesh_size=long_range_mesh_size,
            long_range_slab_padding_factor=long_range_slab_padding_factor,
            long_range_include_k0=long_range_include_k0,
            long_range_source_channels=long_range_source_channels,
            long_range_backend=long_range_backend,
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
            long_range_theta=long_range_theta,
            long_range_leaf_size=long_range_leaf_size,
            long_range_multipole_order=long_range_multipole_order,
            long_range_far_source_dim=long_range_far_source_dim,
            long_range_far_num_shells=long_range_far_num_shells,
            long_range_far_shell_growth=long_range_far_shell_growth,
            long_range_far_tail=long_range_far_tail,
            long_range_far_tail_bins=long_range_far_tail_bins,
            long_range_far_stats=long_range_far_stats,
            long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            long_range_far_source_norm=long_range_far_source_norm,
            long_range_far_gate_init=long_range_far_gate_init,
            feature_spectral_mode=feature_spectral_mode,
            feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=feature_spectral_boundary,
            feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=feature_spectral_neutralize,
            feature_spectral_include_k0=feature_spectral_include_k0,
            feature_spectral_assignment=feature_spectral_assignment,
            feature_spectral_gate_init=feature_spectral_gate_init,
        ).to(device)
    elif mode in {"pure-cartesian-ictd-o3", "pure-cartesian-ictd-save-o3"}:
        o3_model_cls = (
            PureCartesianICTDSaveO3TransformerLayer
            if mode == "pure-cartesian-ictd-save-o3"
            else PureCartesianICTDO3TransformerLayer
        )
        model = o3_model_cls(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            physical_tensor_outputs=physical_tensor_outputs,
            external_tensor_rank=external_tensor_rank,
            external_tensor_irrep=external_tensor_irrep,
            external_tensor_specs=external_tensor_specs,
            o3_irrep_preset=o3_irrep_preset,
            o3_active_irreps=o3_active_irreps,
            num_fidelity_levels=num_fidelity_levels,
            multi_fidelity_mode=multi_fidelity_mode,
        internal_compute_dtype=config.internal_compute_dtype,
            device=torch.device(device),
            long_range_mode=long_range_mode,
            long_range_hidden_dim=long_range_hidden_dim,
            long_range_boundary=long_range_boundary,
            long_range_neutralize=long_range_neutralize,
            long_range_filter_hidden_dim=long_range_filter_hidden_dim,
            long_range_kmax=long_range_kmax,
            long_range_mesh_size=long_range_mesh_size,
            long_range_slab_padding_factor=long_range_slab_padding_factor,
            long_range_include_k0=long_range_include_k0,
            long_range_source_channels=long_range_source_channels,
            long_range_backend=long_range_backend,
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
            long_range_theta=long_range_theta,
            long_range_leaf_size=long_range_leaf_size,
            long_range_multipole_order=long_range_multipole_order,
            long_range_far_source_dim=long_range_far_source_dim,
            long_range_far_num_shells=long_range_far_num_shells,
            long_range_far_shell_growth=long_range_far_shell_growth,
            long_range_far_tail=long_range_far_tail,
            long_range_far_tail_bins=long_range_far_tail_bins,
            long_range_far_stats=long_range_far_stats,
            long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            long_range_far_source_norm=long_range_far_source_norm,
            long_range_far_gate_init=long_range_far_gate_init,
            feature_spectral_mode=feature_spectral_mode,
            feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=feature_spectral_boundary,
            feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=feature_spectral_neutralize,
            feature_spectral_include_k0=feature_spectral_include_k0,
            feature_spectral_assignment=feature_spectral_assignment,
            feature_spectral_gate_init=feature_spectral_gate_init,
        ).to(device)
    elif mode in {"pure-cartesian-ictd-save", "pure-cartesian-ictd-save-multiple"}:
        ictd_tp_path_policy = ictd_tp_path_policy or ckpt.get("ictd_tp_path_policy") or arch_meta.get("ictd_tp_path_policy", "full")
        ictd_tp_max_rank_other = (
            ictd_tp_max_rank_other
            if ictd_tp_max_rank_other is not None
            else ckpt.get("ictd_tp_max_rank_other", arch_meta.get("ictd_tp_max_rank_other"))
        )
        model = PureCartesianICTDTransformerLayerSave(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            save_readout_mode="multiple-contraction" if mode == "pure-cartesian-ictd-save-multiple" else "elementwise-scalar",
            save_contraction_order=save_contraction_order,
            save_multiple_fusion_scheme=save_multiple_fusion_scheme,
            save_final_readout_mode=save_final_readout_mode,
            save_multiple_mix_channels=save_multiple_mix_channels,
            internal_compute_dtype=config.internal_compute_dtype,
            device=torch.device(device),
            long_range_mode=long_range_mode,
            long_range_hidden_dim=long_range_hidden_dim,
            long_range_boundary=long_range_boundary,
            long_range_neutralize=long_range_neutralize,
            long_range_filter_hidden_dim=long_range_filter_hidden_dim,
            long_range_kmax=long_range_kmax,
            long_range_mesh_size=long_range_mesh_size,
            long_range_slab_padding_factor=long_range_slab_padding_factor,
            long_range_include_k0=long_range_include_k0,
            long_range_source_channels=long_range_source_channels,
            long_range_backend=long_range_backend,
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
            long_range_theta=long_range_theta,
            long_range_leaf_size=long_range_leaf_size,
            long_range_multipole_order=long_range_multipole_order,
            long_range_far_source_dim=long_range_far_source_dim,
            long_range_far_num_shells=long_range_far_num_shells,
            long_range_far_shell_growth=long_range_far_shell_growth,
            long_range_far_tail=long_range_far_tail,
            long_range_far_tail_bins=long_range_far_tail_bins,
            long_range_far_stats=long_range_far_stats,
            long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            long_range_far_source_norm=long_range_far_source_norm,
            long_range_far_gate_init=long_range_far_gate_init,
            feature_spectral_mode=feature_spectral_mode,
            feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=feature_spectral_boundary,
            feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=feature_spectral_neutralize,
            feature_spectral_include_k0=feature_spectral_include_k0,
            feature_spectral_assignment=feature_spectral_assignment,
            feature_spectral_gate_init=feature_spectral_gate_init,
        ).to(device)
    elif mode in ("pure-cartesian-sparse", "pure-cartesian-sparse-save"):
        max_rank_other = int(ckpt.get("max_rank_other", arch_meta.get("max_rank_other", 1)))
        k_policy = str(ckpt.get("k_policy", arch_meta.get("k_policy", "k0")))
        sparse_cls = (
            PureCartesianSparseTransformerLayerSave
            if mode == "pure-cartesian-sparse-save"
            else PureCartesianSparseTransformerLayer
        )
        model = sparse_cls(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            lmax=config.lmax,
            max_rank_other=max_rank_other,
            k_policy=k_policy,
            physical_tensor_outputs=physical_tensor_outputs,
            external_tensor_rank=external_tensor_rank,
            external_tensor_specs=external_tensor_specs,
            num_fidelity_levels=num_fidelity_levels,
            multi_fidelity_mode=multi_fidelity_mode,
            device=torch.device(device),
            long_range_mode=long_range_mode,
            long_range_hidden_dim=long_range_hidden_dim,
            long_range_boundary=long_range_boundary,
            long_range_neutralize=long_range_neutralize,
            long_range_filter_hidden_dim=long_range_filter_hidden_dim,
            long_range_kmax=long_range_kmax,
            long_range_mesh_size=long_range_mesh_size,
            long_range_slab_padding_factor=long_range_slab_padding_factor,
            long_range_include_k0=long_range_include_k0,
            long_range_source_channels=long_range_source_channels,
            long_range_backend=long_range_backend,
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_theta=long_range_theta,
            long_range_leaf_size=long_range_leaf_size,
            long_range_multipole_order=long_range_multipole_order,
            long_range_far_source_dim=long_range_far_source_dim,
            long_range_far_num_shells=long_range_far_num_shells,
            long_range_far_shell_growth=long_range_far_shell_growth,
            long_range_far_tail=long_range_far_tail,
            long_range_far_tail_bins=long_range_far_tail_bins,
            long_range_far_stats=long_range_far_stats,
            long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            long_range_far_source_norm=long_range_far_source_norm,
            long_range_far_gate_init=long_range_far_gate_init,
            feature_spectral_mode=feature_spectral_mode,
            feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=feature_spectral_boundary,
            feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=feature_spectral_neutralize,
            feature_spectral_include_k0=feature_spectral_include_k0,
            feature_spectral_assignment=feature_spectral_assignment,
            feature_spectral_gate_init=feature_spectral_gate_init,
        ).to(device)
    elif mode == "spherical-save-cue":
        try:
            import cuequivariance_torch  # noqa: F401
        except Exception as e:
            raise ImportError(
                "tensor_product_mode='spherical-save-cue' requires cuEquivariance.\n"
                "Install: pip install cuequivariance-torch cuequivariance-ops-torch-cu12\n"
                f"Original error: {e}"
            ) from e
        from mace_ictc.models.cue_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise_cue,
        )
        model = E3_TransformerLayer_multi_channelwise_cue(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=num_interaction,
            function_type_main=config.function_type,
            device=torch.device(device),
            force_naive=force_naive,
            long_range_mode=long_range_mode,
            long_range_hidden_dim=long_range_hidden_dim,
            long_range_boundary=long_range_boundary,
            long_range_neutralize=long_range_neutralize,
            long_range_filter_hidden_dim=long_range_filter_hidden_dim,
            long_range_kmax=long_range_kmax,
            long_range_mesh_size=long_range_mesh_size,
            long_range_slab_padding_factor=long_range_slab_padding_factor,
            long_range_include_k0=long_range_include_k0,
            long_range_source_channels=long_range_source_channels,
            long_range_backend=long_range_backend,
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_theta=long_range_theta,
            long_range_leaf_size=long_range_leaf_size,
            long_range_multipole_order=long_range_multipole_order,
            long_range_far_source_dim=long_range_far_source_dim,
            long_range_far_num_shells=long_range_far_num_shells,
            long_range_far_shell_growth=long_range_far_shell_growth,
            long_range_far_tail=long_range_far_tail,
            long_range_far_tail_bins=long_range_far_tail_bins,
            long_range_far_stats=long_range_far_stats,
            long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            long_range_far_source_norm=long_range_far_source_norm,
            long_range_far_gate_init=long_range_far_gate_init,
            feature_spectral_mode=feature_spectral_mode,
            feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=feature_spectral_boundary,
            feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=feature_spectral_neutralize,
            feature_spectral_include_k0=feature_spectral_include_k0,
            feature_spectral_assignment=feature_spectral_assignment,
            feature_spectral_gate_init=feature_spectral_gate_init,
        ).to(device)
    elif mode == "spherical-save":
        model = E3_TransformerLayer_multi_channelwise(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            device=torch.device(device),
            long_range_mode=long_range_mode,
            long_range_hidden_dim=long_range_hidden_dim,
            long_range_boundary=long_range_boundary,
            long_range_neutralize=long_range_neutralize,
            long_range_filter_hidden_dim=long_range_filter_hidden_dim,
            long_range_kmax=long_range_kmax,
            long_range_mesh_size=long_range_mesh_size,
            long_range_slab_padding_factor=long_range_slab_padding_factor,
            long_range_include_k0=long_range_include_k0,
            long_range_source_channels=long_range_source_channels,
            long_range_backend=long_range_backend,
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_theta=long_range_theta,
            long_range_leaf_size=long_range_leaf_size,
            long_range_multipole_order=long_range_multipole_order,
            long_range_far_source_dim=long_range_far_source_dim,
            long_range_far_num_shells=long_range_far_num_shells,
            long_range_far_shell_growth=long_range_far_shell_growth,
            long_range_far_tail=long_range_far_tail,
            long_range_far_tail_bins=long_range_far_tail_bins,
            long_range_far_stats=long_range_far_stats,
            long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            long_range_far_source_norm=long_range_far_source_norm,
            long_range_far_gate_init=long_range_far_gate_init,
            feature_spectral_mode=feature_spectral_mode,
            feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=feature_spectral_boundary,
            feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=feature_spectral_neutralize,
            feature_spectral_include_k0=feature_spectral_include_k0,
            feature_spectral_assignment=feature_spectral_assignment,
            feature_spectral_gate_init=feature_spectral_gate_init,
        ).to(device)
    else:
        model = E3_TransformerLayer_multi(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=num_interaction,
            invariant_channels=invariant_channels,
            function_type_main=config.function_type,
            device=torch.device(device),
            long_range_mode=long_range_mode,
            long_range_hidden_dim=long_range_hidden_dim,
            long_range_boundary=long_range_boundary,
            long_range_neutralize=long_range_neutralize,
            long_range_filter_hidden_dim=long_range_filter_hidden_dim,
            long_range_kmax=long_range_kmax,
            long_range_mesh_size=long_range_mesh_size,
            long_range_slab_padding_factor=long_range_slab_padding_factor,
            long_range_include_k0=long_range_include_k0,
            long_range_source_channels=long_range_source_channels,
            long_range_backend=long_range_backend,
            long_range_reciprocal_backend=long_range_reciprocal_backend,
            long_range_energy_partition=long_range_energy_partition,
            long_range_green_mode=long_range_green_mode,
            long_range_assignment=long_range_assignment,
            long_range_theta=long_range_theta,
            long_range_leaf_size=long_range_leaf_size,
            long_range_multipole_order=long_range_multipole_order,
            long_range_far_source_dim=long_range_far_source_dim,
            long_range_far_num_shells=long_range_far_num_shells,
            long_range_far_shell_growth=long_range_far_shell_growth,
            long_range_far_tail=long_range_far_tail,
            long_range_far_tail_bins=long_range_far_tail_bins,
            long_range_far_stats=long_range_far_stats,
            long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            long_range_far_source_norm=long_range_far_source_norm,
            long_range_far_gate_init=long_range_far_gate_init,
            feature_spectral_mode=feature_spectral_mode,
            feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=feature_spectral_boundary,
            feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=feature_spectral_neutralize,
            feature_spectral_include_k0=feature_spectral_include_k0,
            feature_spectral_assignment=feature_spectral_assignment,
            feature_spectral_gate_init=feature_spectral_gate_init,
        ).to(device)
    if mode == "pure-cartesian-ictd-save-multiple":
        model_keys = set(model.state_dict().keys())
        selected_state_dict = {k: v for k, v in selected_state_dict.items() if k in model_keys}
    if mode == "pure-cartesian-ictd-fix" and "mace_first_layer_sc0" in selected_state_dict:
        if not hasattr(model, "install_mace_first_layer_sc0"):
            raise RuntimeError("checkpoint contains mace_first_layer_sc0 but model cannot install it")
        model.install_mace_first_layer_sc0(selected_state_dict["mace_first_layer_sc0"])

    folded_in_state = False
    if mode == "pure-cartesian-ictd-fix" and getattr(model, "angular_basis", "ictd") == "e3nn":
        folded_in_state = bool(
            ckpt.get(
                "angular_basis_folded_in_state_dict",
                arch_meta.get("angular_basis_folded_in_state_dict", False),
            )
        )
        if folded_in_state:
            if not hasattr(model, "activate_e3nn_basis_from_folded_state_dict"):
                raise RuntimeError(
                    "checkpoint declares angular_basis=e3nn folded state, but model cannot restore it"
                )
            # This creates the fixed bridge modules/buffers and enables the
            # runtime basis flags without folding trainable state.  It must
            # precede strict loading because folded checkpoints may contain
            # those derived bridge buffers.
            model.activate_e3nn_basis_from_folded_state_dict()

    if mode == "spherical-save-cue":
        load_result = model.load_state_dict(selected_state_dict, strict=False)
        if load_result.unexpected_keys or load_result.missing_keys:
            import warnings
            if load_result.unexpected_keys:
                warnings.warn(
                    f"spherical-save-cue: {len(load_result.unexpected_keys)} unexpected keys in checkpoint "
                    "(cuEquivariance 版本差异?), 已忽略"
                )
            if load_result.missing_keys:
                # cuEquivariance auto-generated CG coefficient buffers (e.g. .graphs.0.graph.cN)
                # are not stored in older checkpoints but get recomputed at init. Safe to skip.
                cue_auto_buffers = [k for k in load_result.missing_keys if ".graphs." in k and ".graph.c" in k]
                real_missing = [k for k in load_result.missing_keys if k not in cue_auto_buffers]
                if cue_auto_buffers:
                    warnings.warn(
                        f"spherical-save-cue: {len(cue_auto_buffers)} auto-generated CG buffers 未在 checkpoint 中"
                        "（cuEquivariance 版本差异），已由模型自动初始化"
                    )
                if real_missing:
                    raise RuntimeError(
                        f"spherical-save-cue: {len(real_missing)} missing learned keys: "
                        f"{real_missing[:10]}... checkpoint 与模型结构不匹配."
                    )
    else:
        model.load_state_dict(selected_state_dict, strict=True)

    model = maybe_wrap_model_with_zbl(model, arch_meta)

    # Optional TorchScript tracing
    use_ts = bool(torchscript) or (os.environ.get("MLIAP_USE_TORCHSCRIPT", "").lower() in ("1", "true", "yes"))
    if use_ts:
        _ts_supported = ("pure-cartesian-ictd", "pure-cartesian-ictd-o3", "pure-cartesian-ictd-save", "pure-cartesian-ictd-save-multiple", "pure-cartesian-ictd-save-o3", "pure-cartesian-ictd-fix", "spherical-save-cue")
        if mode not in _ts_supported:
            raise ValueError(f"TorchScript export is only supported for {_ts_supported}, got {mode!r}")
        model = _maybe_torchscript_trace_model(
            model,
            device=torch.device(device),
            dtype=dtype,
            enable=True,
        )

    result = _potential_class(
        model=model,
        element_types=element_types,
        max_radius=radius,
        atomic_energy_keys=aek,
        atomic_energy_values=aev,
        device=device,
        dtype=dtype,
    )
    from mace_ictc.interfaces.deployment_triton import configure_mace_deployment_plan
    if use_ts and operator_plan == "generated":
        raise ValueError("Generated inference requires eager/AOTI, not TorchScript")
    result.operator_plan_info = configure_mace_deployment_plan(
        result.wrapper.model, policy="checkpoint" if use_ts else operator_plan,
        generated_layout=generated_layout,
    )
    print(f"[LAMMPS_MLIAP_MFF] operator plan: {result.operator_plan_info}")
    return result

