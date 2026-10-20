from __future__ import annotations

import math
import os
import re
from typing import Any, Mapping

import torch


DEFAULT_MODEL_ARCHITECTURE: dict[str, Any] = {
    "dtype": "float64",
    "max_atomvalue": 10,
    "embedding_dim": 16,
    "embed_size": [64],
    "output_size": 8,
    "lmax": 2,
    "irreps_output_conv_channels": None,
    "function_type": "gaussian",
    "tensor_product_mode": "spherical",
    "num_interaction": 2,
    "invariant_channels": 32,
    "num_fidelity_levels": 0,
    "multi_fidelity_mode": "conditioning",
    "o3_irrep_preset": "auto",
    "o3_active_irreps": None,
    "max_radius": 5.0,
    "max_rank_other": 1,
    "k_policy": "k0",
    "ictd_tp_path_policy": "full",
    "ictd_tp_max_rank_other": None,
    "ictd_tp_backend": "pytorch",
    "ictd_save_tp_mode": "fully-connected",
    "ictd_fix_use_reduced_cg": False,
    "ictd_fix_edge_lmax": None,
    "save_contraction_order": 3,
    "save_multiple_fusion_scheme": "serial_lastmix",
    "save_final_readout_mode": "direct-1",
    "save_multiple_mix_channels": None,
    "polynomial_cutoff_p": 6,
    "long_range_mode": "none",
    "long_range_hidden_dim": 64,
    "long_range_boundary": "nonperiodic",
    "long_range_neutralize": True,
    "long_range_filter_hidden_dim": 64,
    "long_range_kmax": 2,
    "long_range_mesh_size": 16,
    "long_range_slab_padding_factor": 2,
    "long_range_include_k0": False,
    "long_range_source_channels": 1,
    "long_range_backend": "dense_pairwise",
    "long_range_reciprocal_backend": "direct_kspace",
    "long_range_energy_partition": "potential",
    "long_range_green_mode": "poisson",
    "long_range_assignment": "pcs",
    "long_range_mesh_fft_full_ewald": False,
    "long_range_max_multipole_l": 0,
    "long_range_dispersion_mode": "none",
    "long_range_dispersion": False,
    "dispersion_cutoff": 10.0,
    "dispersion_max_num_neighbors": None,
    "dispersion_neighbor_method": "auto",
    "dispersion_bruteforce_threshold": 1024,
    "dispersion_allow_large_bruteforce_fallback": False,
    "dispersion_training_graph_rule": "none",
    "dispersion_deployment_graph_rule": "none",
    "dispersion_train_deploy_graph_compatibility": "none",
    "dispersion_slq_num_probes": 8,
    "dispersion_slq_lanczos_steps": 16,
    "mbd_operator_backend": "edge_sparse",
    "mbd_pme_mesh_size": 16,
    "mbd_pme_assignment": "pcs",
    "mbd_pme_k_norm_floor": 1.0e-6,
    "mbd_pme_assignment_window_floor": 1.0e-6,
    "mbd_pme_ewald_alpha_prefactor": 5.0,
    "long_range_theta": 0.5,
    "long_range_leaf_size": 32,
    "long_range_multipole_order": 0,
    "long_range_far_source_dim": 16,
    "long_range_far_num_shells": 3,
    "long_range_far_shell_growth": 2.0,
    "long_range_far_tail": True,
    "long_range_far_tail_bins": 2,
    "long_range_far_stats": "mean,count,mean_r,rms_r",
    "long_range_far_max_radius_multiplier": 8.0,
    "long_range_far_source_norm": True,
    "long_range_far_gate_init": 0.0,
    "long_range_num_k": None,
    "feature_spectral_mode": "none",
    "feature_spectral_bottleneck_dim": 8,
    "feature_spectral_mesh_size": 16,
    "feature_spectral_filter_hidden_dim": 64,
    "feature_spectral_boundary": "periodic",
    "feature_spectral_slab_padding_factor": 2,
    "feature_spectral_neutralize": True,
    "feature_spectral_include_k0": False,
    "feature_spectral_assignment": "pcs",
    "feature_spectral_gate_init": 0.0,
    "zbl_enabled": False,
    "zbl_inner_cutoff": 0.8,
    "zbl_outer_cutoff": 1.2,
    "zbl_exponent": 0.23,
    "zbl_energy_scale": 1.0,
    "energy_output_scale_enabled": False,
    "energy_output_scale": 1.0,
    "energy_output_shift_enabled": False,
    "energy_output_shift": 0.0,
    "external_tensor_specs": None,
}

SUPPORTED_DISPERSION_NEIGHBOR_METHODS = {"auto", "cell", "bruteforce"}


def validate_dispersion_deployment_graph_rule(
    *,
    long_range_dispersion_mode: str,
    mbd_operator_backend: str,
    raw_rule: Any | None,
    source_label: str = "dispersion_deployment_graph_rule",
) -> str:
    """Return the expected deployment graph rule and reject inconsistent metadata."""
    from mace_ictc.models.dispersion import dispersion_deployment_graph_rule

    expected = dispersion_deployment_graph_rule(
        long_range_dispersion_mode,
        mbd_operator_backend=mbd_operator_backend,
    )
    if raw_rule is not None and str(raw_rule) != expected:
        raise ValueError(
            f"{source_label} {str(raw_rule)!r} does not match long_range_dispersion_mode="
            f"{long_range_dispersion_mode!r} and mbd_operator_backend={mbd_operator_backend!r}"
        )
    return expected


def validate_dispersion_training_graph_rule(
    *,
    long_range_dispersion_mode: str,
    mbd_operator_backend: str,
    raw_rule: Any | None,
    source_label: str = "dispersion_training_graph_rule",
) -> str:
    """Return the expected training graph rule and reject inconsistent metadata."""
    from mace_ictc.models.dispersion import dispersion_training_graph_rule

    expected = dispersion_training_graph_rule(
        long_range_dispersion_mode,
        mbd_operator_backend=mbd_operator_backend,
    )
    if raw_rule is not None and str(raw_rule) != expected:
        raise ValueError(
            f"{source_label} {str(raw_rule)!r} does not match long_range_dispersion_mode="
            f"{long_range_dispersion_mode!r} and mbd_operator_backend={mbd_operator_backend!r}"
        )
    return expected


def validate_dispersion_train_deploy_graph_compatibility(
    *,
    long_range_dispersion_mode: str,
    mbd_operator_backend: str,
    raw_value: Any | None,
    source_label: str = "dispersion_train_deploy_graph_compatibility",
) -> str:
    """Return the expected train/deploy graph compatibility label."""
    from mace_ictc.models.dispersion import dispersion_train_deploy_graph_compatibility

    expected = dispersion_train_deploy_graph_compatibility(
        long_range_dispersion_mode,
        mbd_operator_backend=mbd_operator_backend,
    )
    if raw_value is not None and str(raw_value) != expected:
        raise ValueError(
            f"{source_label} {str(raw_value)!r} does not match long_range_dispersion_mode="
            f"{long_range_dispersion_mode!r} and mbd_operator_backend={mbd_operator_backend!r}"
        )
    return expected


def resolve_save_multiple_mix_channels_default(
    channels: int,
    num_interaction: int,
) -> int:
    channels = int(channels)
    num_interaction = int(num_interaction)
    return max(1, math.ceil(channels * num_interaction / 2))

def derive_long_range_far_max_radius_multiplier(
    far_num_shells: int,
    far_shell_growth: float,
) -> float:
    far_num_shells = int(far_num_shells)
    far_shell_growth = float(far_shell_growth)
    if far_num_shells < 1:
        raise ValueError("long_range_far_num_shells must be >= 1")
    if far_shell_growth <= 1.0:
        raise ValueError("long_range_far_shell_growth must be > 1")
    return float(math.pow(far_shell_growth, far_num_shells))


def maybe_load_checkpoint(path: str | None, *, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint if isinstance(checkpoint, dict) else None


def get_checkpoint_e3_state_dict(
    checkpoint: Mapping[str, Any] | None,
    *,
    prefer_ema: bool = True,
) -> tuple[Mapping[str, Any], str]:
    """Return the model state_dict to use from a checkpoint.

    New MACE-ICTC checkpoints may carry ``default_state_source`` to choose raw,
    EMA, or SWA weights for deployment. Older checkpoints keep the historical
    behavior: prefer EMA when present, otherwise use raw weights. The second
    return value is a short source label: ``"raw"``, ``"ema"``, or ``"swa"``.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping to resolve e3trans weights.")

    raw_state = checkpoint.get("e3trans_state_dict")
    ema_state = checkpoint.get("e3trans_ema_state_dict")
    swa_state = checkpoint.get("e3trans_swa_state_dict")
    states = {
        "raw": raw_state,
        "ema": ema_state,
        "swa": swa_state,
    }

    default_source = str(checkpoint.get("default_state_source", "") or "").lower()
    if default_source in states and isinstance(states[default_source], Mapping):
        return states[default_source], default_source

    if prefer_ema and isinstance(ema_state, Mapping):
        return ema_state, "ema"

    if isinstance(raw_state, Mapping):
        return raw_state, "raw"

    if isinstance(swa_state, Mapping):
        return swa_state, "swa"

    if isinstance(ema_state, Mapping):
        return ema_state, "ema"

    raise KeyError(
        "Checkpoint does not contain e3trans_state_dict, e3trans_ema_state_dict, "
        "or e3trans_swa_state_dict."
    )


def get_arch_metadata(checkpoint: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {}
    arch_meta = checkpoint.get("model_hyperparameters", {})
    return arch_meta if isinstance(arch_meta, Mapping) else {}


def normalize_tensor_product_mode_name(mode: Any) -> str:
    return str(mode or DEFAULT_MODEL_ARCHITECTURE["tensor_product_mode"])


def infer_ictd_save_multiple_order_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    order_mix_pat = re.compile(
        r"^multiple_contraction(?:_(?:last|mix))?\.order_mix\.(\d+)\."
    )
    tp_layers_pat = re.compile(
        r"^multiple_contraction(?:_(?:last|mix))?\.tp_layers\.(\d+)\."
    )
    max_order_mix = -1
    max_tp_layers = -1
    for key in state_dict.keys():
        match = order_mix_pat.match(key)
        if match:
            max_order_mix = max(max_order_mix, int(match.group(1)))
        match = tp_layers_pat.match(key)
        if match:
            max_tp_layers = max(max_tp_layers, int(match.group(1)))
    if max_order_mix >= 0:
        return max_order_mix + 1
    if max_tp_layers >= 0:
        return max_tp_layers + 1
    return None


def infer_ictd_save_multiple_fusion_scheme_from_state_dict(state_dict: Mapping[str, Any]) -> str | None:
    keys = state_dict.keys()
    has_last = any(key.startswith("multiple_contraction_last.") for key in keys)
    has_mix = any(key.startswith("multiple_contraction_mix.") for key in keys)
    has_fuse = any(key.startswith("multiple_contract_fuse.") for key in keys)
    if has_last and has_mix and has_fuse:
        return "serial_lastmix"
    if any(key.startswith("multiple_contraction.") for key in keys):
        return "single"
    return None


def infer_ictd_save_final_readout_mode_from_state_dict(state_dict: Mapping[str, Any]) -> str | None:
    if any(key.startswith("weighted_sum.") for key in state_dict.keys()):
        return "weighted-17"
    if any(key.startswith("proj_total.") for key in state_dict.keys()):
        return "direct-1"
    return None


def normalize_dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    if value == torch.float64:
        return "float64"
    if value == torch.float32:
        return "float32"
    text = str(value).strip().lower()
    if text in {"torch.float64", "float64", "double"}:
        return "float64"
    if text in {"torch.float32", "float32", "float"}:
        return "float32"
    return text or None


def _resolve_value(
    overrides: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | None,
    arch_meta: Mapping[str, Any],
    key: str,
    default: Any,
    *,
    checkpoint_key: str | None = None,
) -> Any:
    if overrides.get(key) is not None:
        return overrides[key]
    if checkpoint_key is not None and checkpoint is not None and checkpoint.get(checkpoint_key) is not None:
        return checkpoint.get(checkpoint_key)
    if arch_meta.get(key) is not None:
        return arch_meta.get(key)
    return default


def infer_physical_tensor_outputs_from_state_dict(state_dict: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
    per_name: dict[str, dict[int, int]] = {}
    pat = re.compile(r"^physical_tensor_heads\.([^.]+)\.(\d+)\.weight$")
    for key, value in state_dict.items():
        match = pat.match(key)
        if not match:
            continue
        name = match.group(1)
        l_value = int(match.group(2))
        channels_out = int(value.shape[0]) if hasattr(value, "shape") and len(value.shape) >= 1 else 1
        per_name.setdefault(name, {})[l_value] = channels_out

    if not per_name:
        return None

    outputs: dict[str, dict[str, Any]] = {}
    for name, channels_by_l in per_name.items():
        ls = sorted(channels_by_l.keys())
        outputs[name] = {
            "ls": ls,
            "channels_out": {l_value: channels_by_l[l_value] for l_value in ls},
            "reduce": "sum",
        }
    return outputs


def infer_external_tensor_rank_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    if "e3_conv_emb.external_tensor_scale_by_l" in state_dict:
        return 1
    return None


def resolve_model_architecture(
    checkpoint: Mapping[str, Any] | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    arch_meta = get_arch_metadata(checkpoint)
    resolved: dict[str, Any] = {}

    resolved["dtype"] = (
        normalize_dtype_name(overrides.get("dtype"))
        or normalize_dtype_name(checkpoint.get("dtype") if checkpoint is not None else None)
        or normalize_dtype_name(arch_meta.get("dtype"))
        or DEFAULT_MODEL_ARCHITECTURE["dtype"]
    )

    resolved["max_radius"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "max_radius",
            DEFAULT_MODEL_ARCHITECTURE["max_radius"],
            checkpoint_key="max_radius",
        )
    )
    resolved["max_atomvalue"] = int(_resolve_value(overrides, checkpoint, arch_meta, "max_atomvalue", DEFAULT_MODEL_ARCHITECTURE["max_atomvalue"]))
    resolved["embedding_dim"] = int(_resolve_value(overrides, checkpoint, arch_meta, "embedding_dim", DEFAULT_MODEL_ARCHITECTURE["embedding_dim"]))
    resolved["embed_size"] = list(_resolve_value(overrides, checkpoint, arch_meta, "embed_size", DEFAULT_MODEL_ARCHITECTURE["embed_size"]))
    resolved["output_size"] = int(_resolve_value(overrides, checkpoint, arch_meta, "output_size", DEFAULT_MODEL_ARCHITECTURE["output_size"]))
    resolved["lmax"] = int(_resolve_value(overrides, checkpoint, arch_meta, "lmax", DEFAULT_MODEL_ARCHITECTURE["lmax"]))
    resolved["irreps_output_conv_channels"] = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "irreps_output_conv_channels",
        DEFAULT_MODEL_ARCHITECTURE["irreps_output_conv_channels"],
    )
    resolved["function_type"] = str(_resolve_value(overrides, checkpoint, arch_meta, "function_type", DEFAULT_MODEL_ARCHITECTURE["function_type"]))
    resolved["tensor_product_mode"] = normalize_tensor_product_mode_name(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "tensor_product_mode",
            DEFAULT_MODEL_ARCHITECTURE["tensor_product_mode"],
            checkpoint_key="tensor_product_mode",
        )
    )
    resolved["num_interaction"] = int(_resolve_value(overrides, checkpoint, arch_meta, "num_interaction", DEFAULT_MODEL_ARCHITECTURE["num_interaction"]))
    resolved["invariant_channels"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "invariant_channels",
            DEFAULT_MODEL_ARCHITECTURE["invariant_channels"],
        )
    )
    resolved["num_fidelity_levels"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "num_fidelity_levels",
            DEFAULT_MODEL_ARCHITECTURE["num_fidelity_levels"],
        )
    )
    resolved["multi_fidelity_mode"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "multi_fidelity_mode",
            DEFAULT_MODEL_ARCHITECTURE["multi_fidelity_mode"],
        )
    )
    resolved["o3_irrep_preset"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "o3_irrep_preset",
            DEFAULT_MODEL_ARCHITECTURE["o3_irrep_preset"],
        )
    )
    resolved["o3_active_irreps"] = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "o3_active_irreps",
        DEFAULT_MODEL_ARCHITECTURE["o3_active_irreps"],
    )
    resolved["max_rank_other"] = int(_resolve_value(overrides, checkpoint, arch_meta, "max_rank_other", DEFAULT_MODEL_ARCHITECTURE["max_rank_other"]))
    resolved["k_policy"] = str(_resolve_value(overrides, checkpoint, arch_meta, "k_policy", DEFAULT_MODEL_ARCHITECTURE["k_policy"]))
    resolved["ictd_tp_path_policy"] = str(
        _resolve_value(overrides, checkpoint, arch_meta, "ictd_tp_path_policy", DEFAULT_MODEL_ARCHITECTURE["ictd_tp_path_policy"])
    )
    resolved["ictd_tp_max_rank_other"] = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "ictd_tp_max_rank_other",
        DEFAULT_MODEL_ARCHITECTURE["ictd_tp_max_rank_other"],
    )
    resolved["ictd_tp_backend"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "ictd_tp_backend",
            DEFAULT_MODEL_ARCHITECTURE["ictd_tp_backend"],
        )
    )
    resolved["save_contraction_order"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "save_contraction_order",
            DEFAULT_MODEL_ARCHITECTURE["save_contraction_order"],
            checkpoint_key="ictd_save_contraction_order",
        )
    )
    resolved["save_multiple_fusion_scheme"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "save_multiple_fusion_scheme",
            DEFAULT_MODEL_ARCHITECTURE["save_multiple_fusion_scheme"],
            checkpoint_key="ictd_save_multiple_fusion_scheme",
        )
    )
    resolved["save_final_readout_mode"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "save_final_readout_mode",
            DEFAULT_MODEL_ARCHITECTURE["save_final_readout_mode"],
            checkpoint_key="ictd_save_final_readout_mode",
        )
    )
    resolved["save_multiple_mix_channels"] = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "save_multiple_mix_channels",
        resolve_save_multiple_mix_channels_default(
            arch_meta.get("channel_in", 64),
            resolved["num_interaction"],
        ),
        checkpoint_key="ictd_save_multiple_mix_channels",
    )
    if resolved["save_multiple_mix_channels"] is not None:
        resolved["save_multiple_mix_channels"] = int(resolved["save_multiple_mix_channels"])
    resolved["long_range_mode"] = str(
        _resolve_value(overrides, checkpoint, arch_meta, "long_range_mode", DEFAULT_MODEL_ARCHITECTURE["long_range_mode"])
    )
    resolved["long_range_hidden_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_hidden_dim",
            DEFAULT_MODEL_ARCHITECTURE["long_range_hidden_dim"],
        )
    )
    resolved["long_range_boundary"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_boundary",
            DEFAULT_MODEL_ARCHITECTURE["long_range_boundary"],
        )
    )
    resolved["long_range_neutralize"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_neutralize",
            DEFAULT_MODEL_ARCHITECTURE["long_range_neutralize"],
        )
    )
    resolved["long_range_filter_hidden_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_filter_hidden_dim",
            DEFAULT_MODEL_ARCHITECTURE["long_range_filter_hidden_dim"],
        )
    )
    resolved["long_range_kmax"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_kmax",
            DEFAULT_MODEL_ARCHITECTURE["long_range_kmax"],
        )
    )
    resolved["long_range_mesh_size"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_mesh_size",
            DEFAULT_MODEL_ARCHITECTURE["long_range_mesh_size"],
        )
    )
    resolved["long_range_slab_padding_factor"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_slab_padding_factor",
            DEFAULT_MODEL_ARCHITECTURE["long_range_slab_padding_factor"],
        )
    )
    resolved["long_range_include_k0"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_include_k0",
            DEFAULT_MODEL_ARCHITECTURE["long_range_include_k0"],
        )
    )
    resolved["long_range_source_channels"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_source_channels",
            DEFAULT_MODEL_ARCHITECTURE["long_range_source_channels"],
        )
    )
    resolved["long_range_backend"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_backend",
            DEFAULT_MODEL_ARCHITECTURE["long_range_backend"],
        )
    )
    resolved["long_range_reciprocal_backend"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_reciprocal_backend",
            DEFAULT_MODEL_ARCHITECTURE["long_range_reciprocal_backend"],
        )
    )
    resolved["long_range_energy_partition"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_energy_partition",
            DEFAULT_MODEL_ARCHITECTURE["long_range_energy_partition"],
        )
    )
    resolved["long_range_green_mode"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_green_mode",
            DEFAULT_MODEL_ARCHITECTURE["long_range_green_mode"],
        )
    )
    resolved["long_range_assignment"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_assignment",
            DEFAULT_MODEL_ARCHITECTURE["long_range_assignment"],
        )
    )
    resolved["long_range_mesh_fft_full_ewald"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_mesh_fft_full_ewald",
            DEFAULT_MODEL_ARCHITECTURE["long_range_mesh_fft_full_ewald"],
        )
    )
    resolved["long_range_max_multipole_l"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_max_multipole_l",
            DEFAULT_MODEL_ARCHITECTURE["long_range_max_multipole_l"],
        )
    )
    resolved["long_range_dispersion_mode"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_dispersion_mode",
            "pairwise-c6"
            if bool(
                _resolve_value(
                    overrides,
                    checkpoint,
                    arch_meta,
                    "long_range_dispersion",
                    DEFAULT_MODEL_ARCHITECTURE["long_range_dispersion"],
                )
            )
            else DEFAULT_MODEL_ARCHITECTURE["long_range_dispersion_mode"],
        )
    )
    resolved["long_range_dispersion"] = resolved["long_range_dispersion_mode"] != "none"
    resolved["dispersion_cutoff"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "dispersion_cutoff",
            DEFAULT_MODEL_ARCHITECTURE["dispersion_cutoff"],
        )
    )
    dispersion_max_num_neighbors = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "dispersion_max_num_neighbors",
        DEFAULT_MODEL_ARCHITECTURE["dispersion_max_num_neighbors"],
    )
    if dispersion_max_num_neighbors is not None and int(dispersion_max_num_neighbors) < 0:
        raise ValueError("dispersion_max_num_neighbors must be >= 0 or None")
    resolved["dispersion_max_num_neighbors"] = (
        None
        if dispersion_max_num_neighbors is None or int(dispersion_max_num_neighbors) == 0
        else int(dispersion_max_num_neighbors)
    )
    dispersion_neighbor_method = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "dispersion_neighbor_method",
            DEFAULT_MODEL_ARCHITECTURE["dispersion_neighbor_method"],
        )
    )
    if dispersion_neighbor_method not in SUPPORTED_DISPERSION_NEIGHBOR_METHODS:
        raise ValueError(
            f"Unsupported dispersion_neighbor_method {dispersion_neighbor_method!r}; "
            f"supported methods: {sorted(SUPPORTED_DISPERSION_NEIGHBOR_METHODS)}"
        )
    resolved["dispersion_neighbor_method"] = dispersion_neighbor_method
    resolved["dispersion_bruteforce_threshold"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "dispersion_bruteforce_threshold",
            DEFAULT_MODEL_ARCHITECTURE["dispersion_bruteforce_threshold"],
        )
    )
    resolved["dispersion_allow_large_bruteforce_fallback"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "dispersion_allow_large_bruteforce_fallback",
            DEFAULT_MODEL_ARCHITECTURE["dispersion_allow_large_bruteforce_fallback"],
        )
    )
    resolved["dispersion_slq_num_probes"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "dispersion_slq_num_probes",
            DEFAULT_MODEL_ARCHITECTURE["dispersion_slq_num_probes"],
        )
    )
    resolved["dispersion_slq_lanczos_steps"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "dispersion_slq_lanczos_steps",
            DEFAULT_MODEL_ARCHITECTURE["dispersion_slq_lanczos_steps"],
        )
    )
    resolved["mbd_operator_backend"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "mbd_operator_backend",
            DEFAULT_MODEL_ARCHITECTURE["mbd_operator_backend"],
        )
    )
    raw_dispersion_graph_rule = None
    for source in (overrides, checkpoint, arch_meta):
        if isinstance(source, Mapping) and source.get("dispersion_deployment_graph_rule") is not None:
            raw_dispersion_graph_rule = str(source["dispersion_deployment_graph_rule"])
            break
    raw_dispersion_training_graph_rule = None
    for source in (overrides, checkpoint, arch_meta):
        if isinstance(source, Mapping) and source.get("dispersion_training_graph_rule") is not None:
            raw_dispersion_training_graph_rule = str(source["dispersion_training_graph_rule"])
            break
    raw_dispersion_graph_compatibility = None
    for source in (overrides, checkpoint, arch_meta):
        if isinstance(source, Mapping) and source.get("dispersion_train_deploy_graph_compatibility") is not None:
            raw_dispersion_graph_compatibility = str(source["dispersion_train_deploy_graph_compatibility"])
            break
    resolved["dispersion_training_graph_rule"] = validate_dispersion_training_graph_rule(
        long_range_dispersion_mode=resolved["long_range_dispersion_mode"],
        mbd_operator_backend=resolved["mbd_operator_backend"],
        raw_rule=raw_dispersion_training_graph_rule,
    )
    resolved["dispersion_deployment_graph_rule"] = validate_dispersion_deployment_graph_rule(
        long_range_dispersion_mode=resolved["long_range_dispersion_mode"],
        mbd_operator_backend=resolved["mbd_operator_backend"],
        raw_rule=raw_dispersion_graph_rule,
    )
    resolved["dispersion_train_deploy_graph_compatibility"] = (
        validate_dispersion_train_deploy_graph_compatibility(
            long_range_dispersion_mode=resolved["long_range_dispersion_mode"],
            mbd_operator_backend=resolved["mbd_operator_backend"],
            raw_value=raw_dispersion_graph_compatibility,
        )
    )
    resolved["mbd_pme_mesh_size"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "mbd_pme_mesh_size",
            DEFAULT_MODEL_ARCHITECTURE["mbd_pme_mesh_size"],
        )
    )
    resolved["mbd_pme_assignment"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "mbd_pme_assignment",
            DEFAULT_MODEL_ARCHITECTURE["mbd_pme_assignment"],
        )
    )
    resolved["mbd_pme_k_norm_floor"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "mbd_pme_k_norm_floor",
            DEFAULT_MODEL_ARCHITECTURE["mbd_pme_k_norm_floor"],
        )
    )
    resolved["mbd_pme_assignment_window_floor"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "mbd_pme_assignment_window_floor",
            DEFAULT_MODEL_ARCHITECTURE["mbd_pme_assignment_window_floor"],
        )
    )
    resolved["mbd_pme_ewald_alpha_prefactor"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "mbd_pme_ewald_alpha_prefactor",
            DEFAULT_MODEL_ARCHITECTURE["mbd_pme_ewald_alpha_prefactor"],
        )
    )
    resolved["long_range_theta"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_theta",
            DEFAULT_MODEL_ARCHITECTURE["long_range_theta"],
        )
    )
    resolved["long_range_leaf_size"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_leaf_size",
            DEFAULT_MODEL_ARCHITECTURE["long_range_leaf_size"],
        )
    )
    resolved["long_range_multipole_order"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_multipole_order",
            DEFAULT_MODEL_ARCHITECTURE["long_range_multipole_order"],
        )
    )
    resolved["long_range_far_source_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_source_dim",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_source_dim"],
        )
    )
    resolved["long_range_far_num_shells"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_num_shells",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_num_shells"],
        )
    )
    resolved["long_range_far_shell_growth"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_shell_growth",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_shell_growth"],
        )
    )
    resolved["long_range_far_tail"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_tail",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_tail"],
        )
    )
    resolved["long_range_far_tail_bins"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_tail_bins",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_tail_bins"],
        )
    )
    resolved["long_range_far_stats"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_stats",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_stats"],
        )
    )
    raw_far_max_radius_multiplier = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "long_range_far_max_radius_multiplier",
        None,
    )
    if raw_far_max_radius_multiplier is None:
        resolved["long_range_far_max_radius_multiplier"] = derive_long_range_far_max_radius_multiplier(
            resolved["long_range_far_num_shells"],
            resolved["long_range_far_shell_growth"],
        )
    else:
        resolved["long_range_far_max_radius_multiplier"] = float(raw_far_max_radius_multiplier)
    resolved["long_range_far_source_norm"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_source_norm",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_source_norm"],
        )
    )
    resolved["long_range_far_gate_init"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "long_range_far_gate_init",
            DEFAULT_MODEL_ARCHITECTURE["long_range_far_gate_init"],
        )
    )
    resolved["feature_spectral_mode"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_mode",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_mode"],
        )
    )
    resolved["feature_spectral_bottleneck_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_bottleneck_dim",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_bottleneck_dim"],
        )
    )
    resolved["feature_spectral_mesh_size"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_mesh_size",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_mesh_size"],
        )
    )
    resolved["feature_spectral_filter_hidden_dim"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_filter_hidden_dim",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_filter_hidden_dim"],
        )
    )
    resolved["feature_spectral_boundary"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_boundary",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_boundary"],
        )
    )
    resolved["feature_spectral_slab_padding_factor"] = int(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_slab_padding_factor",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_slab_padding_factor"],
        )
    )
    resolved["feature_spectral_neutralize"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_neutralize",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_neutralize"],
        )
    )
    resolved["feature_spectral_include_k0"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_include_k0",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_include_k0"],
        )
    )
    resolved["feature_spectral_assignment"] = str(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_assignment",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_assignment"],
        )
    )
    resolved["feature_spectral_gate_init"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "feature_spectral_gate_init",
            DEFAULT_MODEL_ARCHITECTURE["feature_spectral_gate_init"],
        )
    )
    resolved["zbl_enabled"] = bool(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_enabled",
            DEFAULT_MODEL_ARCHITECTURE["zbl_enabled"],
        )
    )
    resolved["zbl_inner_cutoff"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_inner_cutoff",
            DEFAULT_MODEL_ARCHITECTURE["zbl_inner_cutoff"],
        )
    )
    resolved["zbl_outer_cutoff"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_outer_cutoff",
            DEFAULT_MODEL_ARCHITECTURE["zbl_outer_cutoff"],
        )
    )
    resolved["zbl_exponent"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_exponent",
            DEFAULT_MODEL_ARCHITECTURE["zbl_exponent"],
        )
    )
    resolved["zbl_energy_scale"] = float(
        _resolve_value(
            overrides,
            checkpoint,
            arch_meta,
            "zbl_energy_scale",
            DEFAULT_MODEL_ARCHITECTURE["zbl_energy_scale"],
        )
    )
    resolved["long_range_num_k"] = _resolve_value(
        overrides,
        checkpoint,
        arch_meta,
        "long_range_num_k",
        DEFAULT_MODEL_ARCHITECTURE["long_range_num_k"],
    )

    if checkpoint is not None:
        try:
            state_dict, _ = get_checkpoint_e3_state_dict(checkpoint)
        except (KeyError, ValueError):
            state_dict = {}
    else:
        state_dict = {}
    physical_tensor_outputs = checkpoint.get("physical_tensor_outputs") if checkpoint is not None else None
    if physical_tensor_outputs is None:
        physical_tensor_outputs = arch_meta.get("physical_tensor_outputs")
    if physical_tensor_outputs is None and state_dict:
        physical_tensor_outputs = infer_physical_tensor_outputs_from_state_dict(state_dict)
    resolved["physical_tensor_outputs"] = physical_tensor_outputs

    external_tensor_rank = checkpoint.get("external_tensor_rank") if checkpoint is not None else None
    if external_tensor_rank is None:
        external_tensor_rank = arch_meta.get("external_tensor_rank")
    if external_tensor_rank is None and state_dict:
        external_tensor_rank = infer_external_tensor_rank_from_state_dict(state_dict)
    resolved["external_tensor_rank"] = external_tensor_rank
    resolved["external_tensor_irrep"] = (
        checkpoint.get("external_tensor_irrep")
        if checkpoint is not None and checkpoint.get("external_tensor_irrep") is not None
        else arch_meta.get("external_tensor_irrep")
    )
    resolved["external_tensor_specs"] = (
        checkpoint.get("external_tensor_specs")
        if checkpoint is not None and checkpoint.get("external_tensor_specs") is not None
        else arch_meta.get("external_tensor_specs", DEFAULT_MODEL_ARCHITECTURE["external_tensor_specs"])
    )

    resolved["inference_output_physical_tensors"] = (
        checkpoint.get("inference_output_physical_tensors") if checkpoint is not None else None
    )

    return resolved


def get_checkpoint_atomic_energies(
    checkpoint: Mapping[str, Any] | None,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(checkpoint, Mapping):
        return None

    keys = checkpoint.get("atomic_energy_keys")
    values = checkpoint.get("atomic_energy_values")
    if keys is None or values is None:
        return None

    if isinstance(keys, torch.Tensor):
        keys_tensor = keys.detach().cpu().to(dtype=torch.long)
    else:
        keys_tensor = torch.tensor(list(keys), dtype=torch.long)

    if isinstance(values, torch.Tensor):
        values_tensor = values.detach().cpu().to(dtype=dtype)
    else:
        values_tensor = torch.tensor(list(values), dtype=dtype)

    if keys_tensor.numel() != values_tensor.numel():
        return None

    return keys_tensor, values_tensor
