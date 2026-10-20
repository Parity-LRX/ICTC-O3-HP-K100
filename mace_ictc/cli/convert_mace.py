"""Convert a mace-torch ScaleShiftMACE model into a MACE-ICTC checkpoint.

Example:
    python -m mace_ictc.cli.convert_mace --mace-model mace.model --out mace_ictc.pth
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from e3nn import o3

from mace_ictc.interfaces.mace_converter import convert_mace_to_ictd
from mace_ictc.models.pure_cartesian_ictd_fix import PureCartesianICTDFix


log = logging.getLogger(__name__)


def _load_torch_module(path: str, *, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _as_float(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _as_int(x: Any) -> int:
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)


def _gate_name(gate: Any) -> str:
    return str(getattr(gate, "__name__", getattr(gate, "__class__", type(gate)).__name__)).lower()


def _infer_uniform_hidden(hidden_irreps: o3.Irreps) -> tuple[int, int]:
    channels_by_l: dict[int, int] = {}
    for mul, ir in hidden_irreps:
        expected_parity = 1 if int(ir.l) % 2 == 0 else -1
        if int(ir.p) != expected_parity:
            raise ValueError(
                f"unsupported hidden irreps parity {hidden_irreps}; expected MACE parity l even=e, l odd=o"
            )
        channels_by_l[int(ir.l)] = int(mul)
    if not channels_by_l:
        raise ValueError("hidden_irreps is empty")
    lmax = max(channels_by_l)
    missing = [l for l in range(lmax + 1) if l not in channels_by_l]
    if missing:
        raise ValueError(
            f"unsupported hidden_irreps={hidden_irreps}; MACE-ICTC conversion needs contiguous l=0..L"
        )
    channels = channels_by_l[0]
    if any(c != channels for c in channels_by_l.values()):
        raise ValueError(
            f"unsupported hidden_irreps={hidden_irreps}; all l blocks must have the same channel count"
        )
    return channels, lmax


def _infer_scalar_mlp_hidden(mlp_irreps: o3.Irreps) -> int:
    simplified = mlp_irreps.simplify()
    if len(simplified) != 1:
        raise ValueError(f"only a single scalar MLP_irreps block is supported, got {mlp_irreps}")
    mul, ir = simplified[0]
    if ir != o3.Irrep("0e"):
        raise ValueError(f"only scalar even MLP_irreps are supported, got {mlp_irreps}")
    return int(mul)


def _extract_mace_config(mace_model) -> dict[str, Any]:
    try:
        from mace.tools.scripts_utils import extract_config_mace_model
    except Exception as exc:  # pragma: no cover - depends on optional mace-torch install
        raise RuntimeError("mace-torch must be installed to convert a native MACE model") from exc

    if mace_model.__class__.__name__ != "ScaleShiftMACE":
        raise ValueError(
            "expected a mace-torch ScaleShiftMACE object. In mace-torch, even '--model MACE' "
            "usually saves a ScaleShiftMACE instance with shift=0."
        )
    cfg = extract_config_mace_model(mace_model)
    if "error" in cfg:
        raise ValueError(str(cfg["error"]))
    cfg.setdefault("use_reduced_cg", bool(getattr(mace_model, "use_reduced_cg", False)))
    return cfg


def _validate_supported_config(cfg: dict[str, Any], *, channels: int, lmax: int) -> None:
    num_interactions = _as_int(cfg["num_interactions"])
    if num_interactions < 2:
        raise ValueError(f"num_interactions must be >= 2, got {num_interactions}")
    if _as_int(cfg["max_ell"]) < lmax:
        raise ValueError(
            f"current exact converter requires mace max_ell >= hidden lmax; got max_ell={cfg['max_ell']} "
            f"and hidden lmax={lmax}"
        )
    if str(cfg["radial_type"]) != "bessel":
        raise ValueError(f"only MACE radial_type='bessel' is supported exactly, got {cfg['radial_type']!r}")
    if list(cfg["radial_MLP"]) != [64, 64, 64]:
        raise ValueError(f"only radial_MLP=[64, 64, 64] is supported, got {cfg['radial_MLP']!r}")
    if bool(cfg.get("pair_repulsion", False)):
        raise ValueError("pair_repulsion=True is not supported by the MACE-ICTC converter")
    distance_transform = cfg.get("distance_transform", None)
    if distance_transform not in (None, "None"):
        raise ValueError(f"distance_transform={distance_transform!r} is not supported")
    if "silu" not in _gate_name(cfg.get("gate", "")):
        raise ValueError(f"only silu final readout gate is supported, got {cfg.get('gate')!r}")
    _infer_scalar_mlp_hidden(o3.Irreps(str(cfg["MLP_irreps"])))
    first_name = cfg["interaction_cls_first"].__name__
    rest_name = cfg["interaction_cls"].__name__
    if first_name not in {"RealAgnosticInteractionBlock", "RealAgnosticResidualInteractionBlock"}:
        raise ValueError(f"unsupported first interaction block {first_name!r}")
    if rest_name != "RealAgnosticResidualInteractionBlock":
        raise ValueError(f"unsupported interaction block {rest_name!r}")
    if channels <= 0:
        raise ValueError(f"invalid channel count {channels}")


def _use_reduced_cg_from_config(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("use_reduced_cg", False))


def _uniform_correlation_from_config(cfg: dict[str, Any], *, num_interactions: int) -> int:
    raw = cfg["correlation"]
    if isinstance(raw, (list, tuple)):
        vals = [_as_int(v) for v in raw]
        if len(vals) != int(num_interactions):
            raise ValueError(f"expected {num_interactions} correlation entries, got {vals}")
        if len(set(vals)) != 1:
            raise ValueError(f"per-layer correlation is not supported yet; got {vals}")
        return vals[0]
    return _as_int(raw)


def build_ictd_from_mace_config(
    cfg: dict[str, Any],
    *,
    product_backend: str,
    dtype: torch.dtype,
    device: torch.device,
) -> PureCartesianICTDFix:
    hidden_irreps = o3.Irreps(str(cfg["hidden_irreps"]))
    channels, lmax = _infer_uniform_hidden(hidden_irreps)
    readout_hidden_channels = _infer_scalar_mlp_hidden(o3.Irreps(str(cfg["MLP_irreps"])))
    _validate_supported_config(cfg, channels=channels, lmax=lmax)

    atomic_numbers = [int(z) for z in cfg["atomic_numbers"]]
    max_atomvalue = max(10, max(atomic_numbers) + 1)
    num_interactions = _as_int(cfg["num_interactions"])
    correlation = _uniform_correlation_from_config(cfg, num_interactions=num_interactions)
    model = PureCartesianICTDFix(
        max_embed_radius=_as_float(cfg["r_max"]),
        main_max_radius=_as_float(cfg["r_max"]),
        main_number_of_basis=_as_int(cfg["num_bessel"]),
        hidden_dim_conv=channels,
        hidden_dim_sh=channels,
        hidden_dim=channels,
        channel_in2=channels,
        embedding_dim=channels,
        max_atomvalue=max_atomvalue,
        atomic_numbers=atomic_numbers,
        output_size=8,
        embed_size=[128, 128, 128],
        main_hidden_sizes3=[64],
        num_layers=1,
        num_interaction=num_interactions,
        invariant_channels=channels,
        function_type_main=str(cfg["radial_type"]),
        lmax=lmax,
        ictd_fix_edge_lmax=_as_int(cfg["max_ell"]),
        ictd_save_tp_mode="fully-connected",
        ictd_fix_route="baseline",
        ictd_fix_product_backend=product_backend,
        # Converted models reproduce MACE's first-layer self-connection via the frozen
        # mace_first_layer_sc0 buffer installed below, NOT a trainable first-layer sc.
        # Pin False so this stays correct even though training now defaults it ON.
        ictd_fix_first_layer_self_connection=False,
        ictd_fix_use_reduced_cg=_use_reduced_cg_from_config(cfg),
        ictd_fix_readout_hidden_channels=readout_hidden_channels,
        save_contraction_order=correlation,
        radial_sqrt_num_basis=False,
        polynomial_cutoff_p=_as_int(cfg["num_polynomial_cutoff"]),
        avg_num_neighbors=_as_float(cfg["avg_num_neighbors"]),
        angular_basis="ictd",
        internal_compute_dtype=dtype,
        device=device,
    ).to(device=device, dtype=dtype)
    return model


def _checkpoint_hparams_from_model(model: PureCartesianICTDFix, cfg: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "channel_in": int(model.channels),
        "channel_in2": int(model.channels),
        "channel_in3": 32,
        "channel_in4": 32,
        "channel_in5": 32,
        "max_atomvalue": int(model.max_atomvalue),
        "embedding_dim": int(model.channels),
        "embed_size": [128, 128, 128],
        "output_size": 8,
        "lmax": int(model.lmax),
        "irreps_output_conv_channels": int(model.channels),
        "function_type": str(cfg["radial_type"]),
        "max_radius": _as_float(cfg["r_max"]),
        "max_radius_main": _as_float(cfg["r_max"]),
        "number_of_basis": _as_int(cfg["num_bessel"]),
        "number_of_basis_main": _as_int(cfg["num_bessel"]),
        "num_layers": 1,
        "main_hidden_sizes3": [64],
        "emb_number_main_2": [64, 64, 64],
        "num_interaction": int(model.num_interaction),
        "invariant_channels": int(model.channels),
        "ictd_fix_route": "baseline",
        "ictd_fix_product_backend": str(model.ictd_fix_product_backend),
        "ictd_fix_use_reduced_cg": bool(getattr(model, "ictd_fix_use_reduced_cg", False)),
        "ictd_fix_conv_tp_scale_init": str(getattr(model, "ictd_fix_conv_tp_scale_init", "none")),
        "ictd_fix_freeze_conv_tp_weight": bool(getattr(model, "ictd_fix_freeze_conv_tp_weight", False)),
        "ictd_fix_interaction_init": str(getattr(model, "ictd_fix_interaction_init", "identity")),
        "ictd_fix_readout_hidden_channels": int(getattr(model, "ictd_fix_readout_hidden_channels", 16)),
        "ictd_fix_edge_lmax": int(getattr(model, "ictd_fix_edge_lmax", model.lmax)),
        "save_contraction_order": _uniform_correlation_from_config(cfg, num_interactions=int(model.num_interaction)),
        "ictd_save_tp_mode": "fully-connected",
        "ictd_fix_interaction_attn_heads": 0,
        "radial_sqrt_num_basis": False,
        "polynomial_cutoff_p": _as_int(cfg["num_polynomial_cutoff"]),
        "avg_num_neighbors": _as_float(cfg["avg_num_neighbors"]),
        "energy_output_scale_enabled": bool(getattr(model, "energy_output_scale_enabled", False)),
        "energy_output_scale": (
            float(model.energy_output_scale.detach().cpu().item())
            if torch.is_tensor(getattr(model, "energy_output_scale", None))
            else 1.0
        ),
        "energy_output_shift_enabled": bool(getattr(model, "energy_output_shift_enabled", False)),
        "energy_output_shift": (
            float(model.energy_output_shift.detach().cpu().item())
            if torch.is_tensor(getattr(model, "energy_output_shift", None))
            else 0.0
        ),
    }


def convert_model(
    mace_model,
    out_path: str,
    *,
    source_label: str = "<loaded mace model>",
    product_backend: str = "ictd-bridge-u",
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    cfg = _extract_mace_config(mace_model)

    torch.set_default_dtype(dtype)
    mace_model = mace_model.to(device=device, dtype=torch.float64).eval()
    ictd_model = build_ictd_from_mace_config(
        cfg,
        product_backend=product_backend,
        dtype=torch.float64,
        device=device,
    )
    report = convert_mace_to_ictd(mace_model, ictd_model)
    ictd_model = ictd_model.to(device=device, dtype=dtype).eval()

    atomic_numbers = torch.as_tensor([int(z) for z in cfg["atomic_numbers"]], dtype=torch.long)
    atomic_energies = torch.as_tensor(np.asarray(cfg["atomic_energies"], dtype=np.float64), dtype=dtype)
    hparams = _checkpoint_hparams_from_model(ictd_model, cfg, dtype)
    ckpt = {
        "format": "mace-ictc-converted-from-mace-torch",
        "source_mace_model": str(source_label),
        "tensor_product_mode": "pure-cartesian-ictd-fix",
        "dtype": str(dtype).replace("torch.", ""),
        "max_radius": _as_float(cfg["r_max"]),
        "atomic_energy_keys": atomic_numbers,
        "atomic_energy_values": atomic_energies.detach().cpu(),
        "model_hyperparameters": hparams,
        "e3trans_state_dict": {
            k: v.detach().cpu()
            for k, v in ictd_model.state_dict().items()
        },
        "mace_conversion": {
            "product_backend": str(product_backend),
            "use_reduced_cg": bool(getattr(ictd_model, "ictd_fix_use_reduced_cg", False)),
            "max_ell": int(getattr(ictd_model, "ictd_fix_edge_lmax", ictd_model.lmax)),
            "mace_class": mace_model.__class__.__name__,
            "first_interaction": cfg["interaction_cls_first"].__name__,
            "interaction": cfg["interaction_cls"].__name__,
            "scale": float(report["scale"]),
            "shift": float(report["shift"]),
            "avg_num_neighbors": float(report["avg_num_neighbors"]),
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return ckpt["mace_conversion"]


def convert_file(
    mace_model_path: str,
    out_path: str,
    *,
    product_backend: str = "ictd-bridge-u",
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    mace_model = _load_torch_module(mace_model_path, map_location=device)
    return convert_model(
        mace_model,
        out_path,
        source_label=str(Path(mace_model_path).expanduser()),
        product_backend=product_backend,
        dtype=dtype,
        device=device,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mace-model", required=True, help="Path to a torch-saved mace-torch ScaleShiftMACE model.")
    ap.add_argument("--out", required=True, help="Output MACE-ICTC .pth checkpoint.")
    ap.add_argument("--product-backend", default="ictd-bridge-u", choices=["ictd-bridge-u", "native-mace", "cueq"])
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                    help="Stored MACE-ICTC checkpoint dtype. Use float64 for maximum parity.")
    ap.add_argument("--device", default="cpu", help="Device used while converting.")
    return ap


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    info = convert_file(
        args.mace_model,
        args.out,
        product_backend=args.product_backend,
        dtype=dtype,
        device=args.device,
    )
    log.info(
        "converted %s -> %s backend=%s first=%s scale=%g shift=%g avg_num_neighbors=%g",
        args.mace_model,
        args.out,
        info["product_backend"],
        info["first_interaction"],
        info["scale"],
        info["shift"],
        info["avg_num_neighbors"],
    )


if __name__ == "__main__":
    main()
