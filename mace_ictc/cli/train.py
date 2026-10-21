r"""Train a baseline ICTC-MACE model (energy + force) from preprocessed H5.

Standalone training entry point so MACE-ICTC can *train*, not only deploy. Supports
the make_fx-compiled second-order force step (``--train-makefx-compile``) and the
size-bucketing that makes it cheap on variable-size data (``--makefx-buckets``).

The model is constructed with the SAME ``ModelConfig`` -> kwarg mapping that the
deploy-side ``LAMMPS_MLIAP_MFF.from_checkpoint`` uses, and every architecture choice
is written into the checkpoint's ``model_hyperparameters``, so a model trained here
reloads bit-for-bit for AOTI / LAMMPS export.

Example
-------
    python -m mace_ictc.cli.train \
        --data-dir /path/to/data \           # holds processed_train.h5 (+ processed_val.h5)
        --channels 64 --lmax 2 --num-interaction 2 \
        --epochs 50 --batch-size 4 --device cuda --dtype float64 \
        --train-makefx-compile --makefx-buckets 6 \
        --checkpoint model.pth
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import os
import sys
import random

import h5py
import numpy as np
from mace_ictc.utils.mesh import normalize_mesh_size, parse_mesh_size
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mace_ictc.data import H5Dataset, collate_fn_h5, BucketBatchSampler
from mace_ictc.models.pure_cartesian_ictd_fix import PureCartesianICTDFix
from mace_ictc.utils.config import ModelConfig
from mace_ictc.training.train_loop import ForceTrainer, _DEFAULT_E0_KEYS, _DEFAULT_E0_VALUES, disable_tf32


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _setup_distributed(args):
    """Initialize torch.distributed for torchrun/Slurm-launched training.

    The CLI defaults to auto mode: a normal single-process invocation remains unchanged,
    while ``torchrun --nproc_per_node=N`` initializes DDP from ``env://``.
    """
    world_env = _env_int("WORLD_SIZE", 1)
    use_ddp = args.ddp == "on" or (args.ddp == "auto" and world_env > 1)
    if args.ddp == "on" and world_env <= 1:
        raise RuntimeError("--ddp on requires a torchrun-style WORLD_SIZE > 1 environment")
    if not use_ddp:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_main": True,
            "backend": None,
        }
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    backend = args.ddp_backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("DDP backend 'nccl' requires CUDA; use --ddp-backend gloo on CPU")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = _env_int("LOCAL_RANK", rank)
    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
        "backend": backend,
    }


def _resolve_device(device_arg: str, ddp_info) -> torch.device:
    requested = torch.device(device_arg)
    if ddp_info["enabled"] and requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested a CUDA device but CUDA is not available")
        device = torch.device(f"cuda:{ddp_info['local_rank']}")
        torch.cuda.set_device(device)
        return device
    return requested


def _mace_hidden_irreps(channels: int, lmax: int):
    from e3nn import o3

    return o3.Irreps(
        " + ".join(f"{int(channels)}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(int(lmax) + 1))
    )


def _apply_mace_compatible_random_init(
    model: PureCartesianICTDFix,
    args: argparse.Namespace,
    *,
    atomic_numbers: list[int],
    avg_num_neighbors: float,
    scale: float,
    shift: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Initialize ICTC by building the matching random mace-torch model and converting it.

    This gives a true MACE-compatible random initial function while keeping training in
    the ICTC parameterization. Atomic E0 offsets stay outside the model in ForceTrainer,
    so the temporary MACE model uses zero atomic energies and only contributes the
    interaction network plus ScaleShift block.
    """
    if args.function_type != "bessel":
        raise ValueError("--mace-compatible-random-init currently requires --function-type bessel")
    if args.polynomial_cutoff_p <= 0:
        raise ValueError("--mace-compatible-random-init requires --polynomial-cutoff-p > 0")
    if args.product_backend not in {"ictd-bridge-u", "native-mace", "cueq"}:
        raise ValueError(
            "--mace-compatible-random-init currently supports --product-backend "
            "ictd-bridge-u, native-mace, or cueq"
        )
    if args.angular_basis != "ictd":
        raise ValueError("--mace-compatible-random-init requires --angular-basis ictd")
    if args.num_interaction < 2:
        raise ValueError("--mace-compatible-random-init requires --num-interaction >= 2")
    if args.max_ell is not None and int(args.max_ell) < int(args.lmax):
        raise ValueError("--mace-compatible-random-init requires --max-ell >= --lmax")

    try:
        from e3nn import o3
        from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes
    except Exception as exc:  # pragma: no cover - optional mace-torch dependency
        raise RuntimeError(
            "--mace-compatible-random-init requires mace-torch and e3nn in the training environment"
        ) from exc

    from mace_ictc.interfaces.mace_converter import convert_mace_to_ictd

    if args.seed is not None:
        _set_global_seed(args.seed)
    max_ell = int(args.lmax if args.max_ell is None else args.max_ell)
    convert_dtype = torch.float64
    model.to(device=device, dtype=convert_dtype)
    torch.set_default_dtype(convert_dtype)
    mace_kwargs = dict(
        r_max=float(args.max_radius),
        num_bessel=int(args.num_basis),
        num_polynomial_cutoff=int(args.polynomial_cutoff_p),
        max_ell=max_ell,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=int(args.num_interaction),
        num_elements=len(atomic_numbers),
        hidden_irreps=_mace_hidden_irreps(args.channels, args.lmax),
        MLP_irreps=o3.Irreps(f"{int(args.readout_hidden_channels)}x0e"),
        atomic_energies=np.zeros(len(atomic_numbers), dtype=np.float64),
        avg_num_neighbors=float(avg_num_neighbors),
        atomic_numbers=[int(z) for z in atomic_numbers],
        correlation=int(args.correlation),
        gate=gate_dict["silu"],
        radial_type="bessel",
        radial_MLP=[64, 64, 64],
        atomic_inter_scale=float(scale) if args.scaling != "no_scaling" or args.atomic_inter_scale is not None else 1.0,
        atomic_inter_shift=float(shift)
        if (args.scaling != "no_scaling" and not args.no_atomic_inter_shift) or args.atomic_inter_shift is not None
        else 0.0,
    )
    # mace-torch <=0.3.6 always builds the full CG basis and does not expose
    # ``use_reduced_cg`` on MACE.__init__.  Newer releases accept the flag.
    # Filter it against the concrete base signature because ScaleShiftMACE
    # itself only advertises ``**kwargs``.
    from mace.modules.models import MACE as MACEModel

    if "use_reduced_cg" in inspect.signature(MACEModel.__init__).parameters:
        mace_kwargs["use_reduced_cg"] = bool(args.use_reduced_cg)
    mace_model = ScaleShiftMACE(**mace_kwargs).to(
        device=device,
        dtype=convert_dtype,
    )
    try:
        report = convert_mace_to_ictd(mace_model.eval(), model)
    finally:
        model.to(device=device, dtype=dtype)
        torch.set_default_dtype(dtype)
    logging.info(
        "initialized ICTC from matching random mace-torch model: avg_num_neighbors=%s scale=%s shift=%s",
        report.get("avg_num_neighbors"),
        report.get("scale"),
        report.get("shift"),
    )


def _set_global_seed(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _make_worker_init_fn(seed: int | None):
    if seed is None:
        return None

    def _seed_worker(worker_id: int):
        worker_seed = int(seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return _seed_worker


def _avg_num_neighbors_from_h5(path: str) -> float:
    """Total directed edges / total nodes over the whole training file.

    This is the message-normalization constant the ictd-fix model divides by; it is a
    plain float (not a state_dict buffer), so it must be recorded in the checkpoint or
    the deployed forces are silently wrong. Mirrors the FSCETP cli computation."""
    tot_e = tot_n = 0
    with h5py.File(path, "r") as f:
        for k in f.keys():
            tot_e += int(f[k]["edge_src"].shape[0])
            tot_n += int(f[k]["pos"].shape[0])
    return float(tot_e) / max(tot_n, 1)


def _sample_keys(f: h5py.File) -> list[str]:
    def _key_order(name: str):
        if name.startswith("sample_"):
            tail = name.split("_", 1)[1]
            if tail.isdigit():
                return (0, int(tail))
        return (1, name)

    return sorted(
        (
            k
            for k in f.keys()
            if isinstance(f[k], h5py.Group) and "A" in f[k] and "y" in f[k]
        ),
        key=_key_order,
    )


def _atomic_inter_scale_shift_from_h5(
    path: str,
    *,
    atomic_energy_keys,
    atomic_energy_values,
    scaling: str,
    max_samples: int | None = None,
) -> tuple[float, float]:
    """MACE-style interaction-energy statistics from a processed H5.

    ``mean`` is the per-graph average atomic interaction energy
    ``(E_total - sum(E0[Z])) / n_atoms``. For ``rms_forces_scaling`` the scale is
    the RMS of force components; for ``std_scaling`` it is the sample standard
    deviation of those per-graph atomic interaction energies.
    """
    if scaling == "no_scaling":
        return 1.0, 0.0
    if scaling not in {"std_scaling", "rms_forces_scaling"}:
        raise ValueError(f"unknown scaling mode {scaling!r}")

    e0_by_z = {int(k): float(v) for k, v in zip(atomic_energy_keys, atomic_energy_values)}
    atom_inter: list[float] = []
    force_sq_sum = 0.0
    force_count = 0

    with h5py.File(path, "r") as f:
        for i, key in enumerate(_sample_keys(f)):
            if max_samples is not None and i >= max_samples:
                break
            g = f[key]
            A = np.asarray(g["A"][:], dtype=np.int64).reshape(-1)
            missing = sorted({int(z) for z in A if int(z) not in e0_by_z})
            if missing:
                raise ValueError(
                    f"{path}:{key} contains atomic numbers {missing} without E0 values; "
                    "pass --atomic-energy-keys/--atomic-energy-values."
                )
            e0 = sum(e0_by_z[int(z)] for z in A)
            n_atoms = max(int(A.shape[0]), 1)
            atom_inter.append((float(g["y"][()]) - e0) / n_atoms)
            if scaling == "rms_forces_scaling":
                force = np.asarray(g["force"][:], dtype=np.float64)
                force_sq_sum += float(np.square(force).sum())
                force_count += int(force.size)

    if not atom_inter:
        raise ValueError(f"no samples with A/y were found in {path}")

    mean = float(sum(atom_inter) / len(atom_inter))
    if scaling == "std_scaling":
        if len(atom_inter) < 2:
            scale = 1.0
        else:
            var = sum((x - mean) ** 2 for x in atom_inter) / (len(atom_inter) - 1)
            scale = math.sqrt(max(var, 0.0))
    else:
        scale = math.sqrt(force_sq_sum / force_count) if force_count > 0 else 1.0

    if scale == 0.0:
        logging.warning("ScaleShift scale statistic is zero; using scale=1.0")
        scale = 1.0
    return float(scale), mean


def _resolve_atomic_inter_scale_shift(
    path: str,
    *,
    atomic_energy_keys,
    atomic_energy_values,
    scaling: str,
    atomic_inter_scale: float | None,
    atomic_inter_shift: float | None,
    no_atomic_inter_shift: bool = False,
    max_samples: int | None = None,
) -> tuple[float, float]:
    """Resolve output statistics without scanning H5 when fully specified."""
    if atomic_inter_scale is not None and (
        atomic_inter_shift is not None or no_atomic_inter_shift
    ):
        return float(atomic_inter_scale), (
            0.0 if no_atomic_inter_shift else float(atomic_inter_shift)
        )
    scale, shift = _atomic_inter_scale_shift_from_h5(
        path,
        atomic_energy_keys=atomic_energy_keys,
        atomic_energy_values=atomic_energy_values,
        scaling=scaling,
        max_samples=max_samples,
    )
    if no_atomic_inter_shift:
        shift = 0.0
    if atomic_inter_scale is not None:
        scale = float(atomic_inter_scale)
    if atomic_inter_shift is not None:
        shift = float(atomic_inter_shift)
    return scale, shift


def build_baseline_model(
    cfg: ModelConfig,
    *,
    avg_num_neighbors: float,
    num_interaction: int,
    route: str,
    product_backend: str,
    correlation: int,
    use_reduced_cg: bool = False,
    first_layer_self_connection: bool = True,
    self_connection_rank: int = 0,
    interaction_scale: str = "none",
    conv_tp_scale_init: str = "none",
    conv_tp_backend: str = "pytorch",
    freeze_conv_tp_weight: bool = False,
    interaction_init: str = "identity",
    readout_hidden_channels: int = 16,
    polynomial_cutoff_p: int | None = 5,
    angular_basis: str = "ictd",
    radial_sqrt_num_basis: bool,
    edge_lmax: int | None,
    attn_heads: int,
    atomic_numbers,
    ictd_save_tp_mode: str,
    invariant_channels: int,
    device,
    dtype,
    energy_output_scale: float = 1.0,
    energy_output_scale_enabled: bool = False,
    energy_output_shift: float = 0.0,
    energy_output_shift_enabled: bool = False,
    long_range_mode: str = "none",
    long_range_boundary: str = "periodic",
    long_range_reciprocal_backend: str = "direct_kspace",
    long_range_kmax: int = 2,
    long_range_mesh_size: int | tuple[int, int, int] = 16,
    long_range_source_channels: int = 1,
    long_range_hidden_dim: int = 64,
    long_range_filter_hidden_dim: int = 64,
    long_range_neutralize: bool = True,
    long_range_include_k0: bool = False,
    long_range_energy_partition: str = "potential",
    long_range_green_mode: str = "poisson",
    long_range_assignment: str = "pcs",
    long_range_slab_padding_factor: int = 2,
    long_range_mesh_fft_full_ewald: bool = False,
    long_range_mesh_fft_reciprocal_only: bool = False,
    long_range_smearing_sigma: float | None = None,
    long_range_max_multipole_l: int = 0,
    long_range_multipole_gate_init: float = 0.1,
    long_range_dispersion_mode: str = "none",
    dispersion_cutoff: float = 8.0,
    dispersion_max_num_neighbors: int | None = None,
    dispersion_neighbor_method: str = "auto",
    dispersion_bruteforce_threshold: int = 1024,
    dispersion_allow_large_bruteforce_fallback: bool = False,
    dispersion_slq_num_probes: int = 8,
    dispersion_slq_lanczos_steps: int = 16,
    mbd_operator_backend: str = "edge_sparse",
    mbd_pme_mesh_size: int = 32,
    mbd_anisotropic_polarizability: bool = False,
    mbd_learnable_energy_scale: bool = True,
    mbd_alpha_floor: float = 1.0e-4,
    dispersion_min_cutoff: float = 0.0,
    mbd_pme_assignment: str = "pcs",
    mbd_pme_k_norm_floor: float = 1.0e-6,
    mbd_pme_assignment_window_floor: float = 1.0e-6,
    mbd_pme_ewald_alpha_prefactor: float = 5.0,
) -> PureCartesianICTDFix:
    """Construct the model exactly the way from_checkpoint rebuilds it (so the saved
    weights reload into an identical module). All structural choices come from ``cfg``
    (which is saved verbatim into ``model_hyperparameters``) plus the explicit args."""
    return PureCartesianICTDFix(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        hidden_dim_conv=cfg.channel_in,
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        atomic_numbers=atomic_numbers,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=num_interaction,
        invariant_channels=invariant_channels,
        function_type_main=cfg.function_type,
        lmax=cfg.lmax,
        ictd_fix_edge_lmax=edge_lmax,
        ictd_save_tp_mode=ictd_save_tp_mode,
        ictd_fix_route=route,
        ictd_fix_product_backend=product_backend,
        ictd_fix_use_reduced_cg=bool(use_reduced_cg),
        ictd_fix_first_layer_self_connection=bool(first_layer_self_connection),
        ictd_fix_self_connection_rank=int(self_connection_rank),
        ictd_fix_interaction_scale=interaction_scale,
        ictd_fix_conv_tp_scale_init=conv_tp_scale_init,
        ictd_tp_backend=conv_tp_backend,
        ictd_fix_freeze_conv_tp_weight=bool(freeze_conv_tp_weight),
        ictd_fix_interaction_init=interaction_init,
        ictd_fix_readout_hidden_channels=int(readout_hidden_channels),
        ictd_fix_interaction_attn_heads=attn_heads,
        angular_basis=angular_basis,
        save_contraction_order=correlation,
        radial_sqrt_num_basis=radial_sqrt_num_basis,
        polynomial_cutoff_p=polynomial_cutoff_p,
        avg_num_neighbors=avg_num_neighbors,
        energy_output_scale=energy_output_scale,
        energy_output_scale_enabled=energy_output_scale_enabled,
        energy_output_shift=energy_output_shift,
        energy_output_shift_enabled=energy_output_shift_enabled,
        long_range_mode=long_range_mode,
        long_range_boundary=long_range_boundary,
        long_range_reciprocal_backend=long_range_reciprocal_backend,
        long_range_kmax=long_range_kmax,
        long_range_mesh_size=long_range_mesh_size,
        long_range_source_channels=long_range_source_channels,
        long_range_hidden_dim=long_range_hidden_dim,
        long_range_filter_hidden_dim=long_range_filter_hidden_dim,
        long_range_neutralize=long_range_neutralize,
        long_range_include_k0=long_range_include_k0,
        long_range_energy_partition=long_range_energy_partition,
        long_range_green_mode=long_range_green_mode,
        long_range_assignment=long_range_assignment,
        long_range_slab_padding_factor=long_range_slab_padding_factor,
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
        mbd_anisotropic_polarizability=mbd_anisotropic_polarizability,
        mbd_learnable_energy_scale=mbd_learnable_energy_scale,
        mbd_alpha_floor=mbd_alpha_floor,
        dispersion_min_cutoff=dispersion_min_cutoff,
        mbd_pme_assignment=mbd_pme_assignment,
        mbd_pme_k_norm_floor=mbd_pme_k_norm_floor,
        mbd_pme_assignment_window_floor=mbd_pme_assignment_window_floor,
        mbd_pme_ewald_alpha_prefactor=mbd_pme_ewald_alpha_prefactor,
        internal_compute_dtype=cfg.internal_compute_dtype,
        device=device,
    ).to(device=device, dtype=dtype)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    ap.add_argument("--data-dir", required=True,
                    help="Directory holding processed_<prefix>.h5 (built by preprocessing).")
    ap.add_argument("--train-prefix", default="train")
    ap.add_argument("--val-prefix", default="val")
    # architecture
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--max-ell", type=int, default=None,
                    help="MACE-style edge spherical harmonics max_ell. Defaults to --lmax.")
    ap.add_argument("--num-interaction", type=int, default=2)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--correlation", type=int, default=2, help="save_contraction_order (body order - 1).")
    ap.add_argument("--route", default="baseline")
    ap.add_argument("--product-backend", default="ictd-bridge-u")
    ap.add_argument("--angular-basis", default="ictd", choices=["ictd", "e3nn"],
                    help="Internal angular basis during training. Use e3nn only with fold-capable "
                         "product backends such as cueq; bridge-U has no e3nn-fold path.")
    ap.add_argument("--use-reduced-cg", action="store_true",
                    help="Use reduced-CG MACE/cuEq symmetric-contraction paths where supported.")
    ap.add_argument("--first-layer-self-connection", action=argparse.BooleanOptionalAction, default=True,
                    help="Trainable first-layer l=0 self-connection during ICTC training (default ON; use "
                         "--no-first-layer-self-connection to disable, e.g. when finetuning a converted MACE "
                         "checkpoint that already carries the frozen mace_first_layer_sc0 buffer).")
    ap.add_argument("--interaction-scale", default="none", choices=["none", "mace-rms"],
                    help="Optional learnable per-l interaction/self-connection scale initialization.")
    ap.add_argument("--conv-tp-scale-init", default="none", choices=["none", "e3nn"],
                    help="Initialize ICTC convolution TP path weights to match e3nn/MACE path scales.")
    ap.add_argument("--conv-tp-backend", default="pytorch", choices=["pytorch", "cueq"],
                    help="Interaction conv-TP compute backend: 'pytorch' reference projector path, or "
                         "'cueq' for the fused cuEquivariance kernel (CUDA only; numerically equivalent, "
                         "checkpoint-compatible). Saved as ictd_tp_backend in model_hyperparameters.")
    ap.add_argument("--freeze-conv-tp-weight", action="store_true",
                    help="Freeze ICTC convolution TP base path weights, matching MACE's external-weight parameterization more closely.")
    ap.add_argument("--operator-plan", choices=("auto", "reference", "generated"), default="auto",
                    help="Default auto enables qualified independent CUDA FP32/C64/L<=3/cor=3 operators. "
                         "Unsupported configurations keep their original route; generated requires coverage. "
                         "Precision and TP parameter trainability are never changed by this flag.")
    ap.add_argument("--backward-targets", choices=("auto", "parameters", "all"), default="auto",
                    help="auto requests all trainable parameter gradients for the generated plan; "
                         "all also computes loss gradients for leaf inputs such as positions.")
    ap.add_argument("--generated-layout", choices=("packed", "fused_pack"), default="packed",
                    help="Generated HP message layout; packed is the established default, fused_pack fuses boundary copies.")
    ap.add_argument("--training-message-linear-layout", choices=("node_component", "einsum"), default="node_component",
                    help="Generated eager training message linear layout; node_component uses contiguous "
                         "node/angular GEMM rows, einsum restores the original layout.")
    ap.add_argument("--interaction-init", default="identity", choices=["identity", "mace-random"],
                    help="Initialize ICTC interaction linear/skip maps as identity or MACE-like random maps.")
    ap.add_argument("--mace-compatible-random-init", action="store_true",
                    help="Build a matching random mace-torch ScaleShiftMACE model and convert it before training.")
    ap.add_argument("--invariant-channels", type=int, default=32)
    ap.add_argument("--ictd-save-tp-mode", default="fully-connected")
    ap.add_argument("--function-type", default="bessel", choices=["gaussian", "bessel"])
    ap.add_argument("--num-basis", type=int, default=8,
                    help="Number of radial basis functions (MACE num_bessel for bessel radial basis).")
    ap.add_argument("--polynomial-cutoff-p", type=int, default=5,
                    help="MACE PolynomialCutoff order p (default 5, matching MACE-MP-0 / MACE-OFF). MUST match "
                         "the source model when finetuning a converted checkpoint, else the radial envelope is "
                         "wrong and forces are systematically off (~1.8x). Use <=0 to disable the envelope.")
    ap.add_argument("--readout-hidden-channels", type=int, default=16,
                    help="Scalar hidden width of the final MACE-style readout (MACE MLP_irreps width).")
    ap.add_argument(
        "--self-connection-rank",
        type=int,
        default=0,
        help=(
            "Rank of the element-specific correction in each MACE self-connection; "
            "0 keeps the full per-element matrices, while 4 is a compact materials default."
        ),
    )
    ap.add_argument("--max-radius", type=float, default=5.0)
    ap.add_argument("--attn-heads", type=int, default=0)
    ap.add_argument("--radial-sqrt-num-basis", action="store_true",
                    help="Use the sqrt(num_basis) radial scale (default OFF = byte-literal MACE radial).")
    ap.add_argument("--avg-num-neighbors", type=float, default=None,
                    help="Override the message normalizer (default: auto-computed from the train H5).")
    # optional reciprocal-space long-range correction
    ap.add_argument("--long-range-mode", default="none", choices=["none", "reciprocal-spectral-v1"],
                    help="Optional scalar reciprocal-space long-range correction. Off by default.")
    ap.add_argument("--long-range-boundary", default="periodic", choices=["periodic", "slab"],
                    help="Boundary for reciprocal long-range mode. direct_kspace requires periodic.")
    ap.add_argument("--long-range-reciprocal-backend", default="direct_kspace", choices=["direct_kspace", "mesh_fft"],
                    help="direct_kspace keeps the long-range energy in the exported core; mesh_fft can export latent sources.")
    ap.add_argument("--long-range-kmax", type=int, default=2,
                    help="Integer reciprocal lattice cutoff for direct_kspace.")
    ap.add_argument("--long-range-mesh-size", type=parse_mesh_size, default=16,
                    help="FFT mesh size M or Mx,My,Mz for mesh_fft reciprocal backend.")
    ap.add_argument("--long-range-source-channels", type=int, default=1,
                    help="Number of learned latent scalar source channels.")
    ap.add_argument("--long-range-hidden-dim", type=int, default=64)
    ap.add_argument("--long-range-filter-hidden-dim", type=int, default=64)
    ap.add_argument("--no-long-range-neutralize", dest="long_range_neutralize", action="store_false",
                    help="Do not subtract per-graph mean latent source before reciprocal solve.")
    ap.set_defaults(long_range_neutralize=True)
    ap.add_argument("--long-range-include-k0", action="store_true",
                    help="Include the k=0 mode. Usually keep off with neutralized sources.")
    ap.add_argument("--long-range-energy-partition", default="potential", choices=["potential", "uniform"])
    ap.add_argument("--long-range-green-mode", default="poisson", choices=["poisson", "learned_poisson"])
    ap.add_argument("--long-range-assignment", default="pcs", choices=["cic", "tsc", "pcs"],
                    help="Mesh assignment rule for mesh_fft backend.")
    ap.add_argument("--long-range-slab-padding-factor", type=int, default=2)
    ap.add_argument("--long-range-mesh-fft-full-ewald", action="store_true",
                    help="Use full-Ewald-style mesh FFT correction terms for mesh_fft.")
    ap.add_argument("--long-range-mesh-fft-reciprocal-only", action="store_true",
                    help="Retain only the screened reciprocal functional; omit real/self/background terms.")
    ap.add_argument("--long-range-smearing-sigma", type=float, default=None,
                    help="Fixed Gaussian length in coordinate units: exp(-sigma^2*k^2/2). "
                         "Requires mesh_fft and --long-range-mesh-fft-full-ewald. "
                         "Omission preserves legacy box-dependent screening.")
    ap.add_argument("--electrostatics-backend", default="auto", choices=["auto", "hip", "reference"],
                    help="Mesh execution backend; hip requires eager K100 FP32/FP64.")
    ap.add_argument("--long-range-max-multipole-l", type=int, default=0,
                    help="Maximum learned multipole rank emitted for mesh_fft reciprocal long-range export.")
    ap.add_argument("--long-range-multipole-gate-init", type=float, default=0.1,
                    help="Initial value of the multipole long-range gentle-start gate (a learnable scalar that "
                         "multiplies the emitted q/mu/Q sources). The reciprocal energy is quadratic in the sources, "
                         "so a fresh multipole head starts at ~gate^2 of full strength: small (e.g. 0.1) keeps the "
                         "head gentle when WARM-STARTING a trained backbone (--finetune) while staying trainable; "
                         "set 1.0 for the ungated from-scratch behavior. The gate is folded into the exported source, "
                         "so train and deploy stay consistent.")
    ap.add_argument("--long-range-dispersion-mode", default="none", choices=["none", "pairwise-c6", "mbd", "mbd-slq"],
                    help="Optional long-range dispersion term. pairwise-c6 is the existing learned C6/R0 model; "
                         "mbd is a dense QHO many-body baseline; mbd-slq is the matrix-free stochastic-Lanczos "
                         "cutoff approximation.")
    ap.add_argument("--long-range-dispersion", dest="long_range_dispersion_mode",
                    action="store_const", const="pairwise-c6",
                    help="Deprecated alias for --long-range-dispersion-mode pairwise-c6.")
    ap.add_argument("--dispersion-cutoff", type=float, default=8.0,
                    help="Cutoff (A) for the long-range dispersion neighbor list. ~8 gives ~5%% MBD-energy "
                         "convergence (energy decays ~1/r^6). Use 0 to reuse the input edge list.")
    ap.add_argument("--dispersion-max-num-neighbors", type=int, default=0,
                    help="Optional max neighbors per atom for torch_cluster dispersion radius search. "
                         "0 estimates a conservative cap from density and cutoff.")
    ap.add_argument("--dispersion-neighbor-method", default="auto", choices=["auto", "cell", "bruteforce"],
                    help="Dispersion neighbor-list builder used during training when explicit dispersion edges "
                         "are absent. auto uses dense only below --dispersion-bruteforce-threshold and otherwise "
                         "uses torch_cluster; cell is an exact sorted Python cell-list path for nearest-image cases.")
    ap.add_argument("--dispersion-bruteforce-threshold", type=int, default=1024,
                    help="Largest per-graph atom count that auto may send to the dense O(N^2) dispersion builder.")
    ap.add_argument("--dispersion-allow-large-bruteforce-fallback", action="store_true",
                    help="Allow auto to fall back to the dense O(N^2) dispersion builder when torch_cluster is "
                         "missing above the threshold. Default is to fail loudly.")
    ap.add_argument("--dispersion-slq-num-probes", type=int, default=8,
                    help="Number of deterministic Hutchinson probes for --long-range-dispersion-mode mbd-slq.")
    ap.add_argument("--dispersion-slq-lanczos-steps", type=int, default=16,
                    help="Lanczos steps per probe for --long-range-dispersion-mode mbd-slq.")
    ap.add_argument("--mbd-operator-backend", default="edge_sparse", choices=["edge_sparse", "pme_fft"],
                    help="SLQ-MBD matrix-vector backend. edge_sparse uses the explicit cutoff dispersion graph "
                         "(direct, fast for small/medium systems); pme_fft is a reciprocal-only torch.fft matvec "
                         "that bypasses the real-space graph. BOTH deploy: AOTI/LAMMPS export the (omega,alpha) "
                         "head and the C++ MBD solver runs the matching operator (edge_sparse=direct, pme_fft="
                         "use_fft). Match the backend across train and deploy for consistent forces.")
    ap.add_argument("--mbd-pme-mesh-size", type=int, default=32,
                    help="PME mesh size for --mbd-operator-backend pme_fft (finer = more accurate at larger boxes).")
    ap.add_argument("--mbd-anisotropic", dest="mbd_anisotropic_polarizability", action="store_true",
                    help="Anisotropic (l=2 tensor) MBD polarizability: the ICTC l=2 node block sets a per-atom 3x3 "
                         "polarizability alpha^{1/2}=B (coupling W=omega*B) instead of a scalar. Emits an [N,8] "
                         "(omega, alpha_iso, 6*B) deploy source the C++ MBD solver consumes. Default off (scalar [N,2]).")
    ap.add_argument("--mbd-fixed-energy-scale", dest="mbd_learnable_energy_scale", action="store_false",
                    help="Pin the MBD dispersion.term.energy_scale at a fixed 1.0 (registered as a buffer, not a "
                         "trainable Parameter) instead of letting gradient descent freely learn it. Diagnostic: the "
                         "learnable default let energy_scale settle at ~0.056 on carbon (heads starved even after "
                         "the ramp) -- pinning forces the alpha/omega/l2 heads to adapt around a mandatory "
                         "full-strength MBD term instead of being able to suppress it. Default learnable (unchanged).")
    ap.add_argument("--dispersion-min-cutoff", type=float, default=0.0,
                    help="Exclude dispersion edges shorter than this (default 0.0 = no exclusion, all edges "
                         "0..dispersion-cutoff included -- opt in explicitly per run). For --long-range-"
                         "dispersion-mode mbd-slq/mbd, the coupled-oscillator formula's PD-safety spectral "
                         "clamp is a global per-graph budget that bonded/intralayer edges can dominate, "
                         "starving genuinely long-range pairs (2026-07-01 carbon diagnosis) -- setting this "
                         "to --max-radius removes that competition. Tried as an always-on default (2026-07-02) "
                         "but the payoff was too thin/inconsistent to justify auto-applying (turned an "
                         "exactly-flat carbon isolation-test signal into a non-flat but still accuracy-"
                         "irrelevant one; didn't prevent MD22's mbd-slq arm from being net-negative), so it's "
                         "opt-in. Not motivated at all for pairwise-c6 (a plain per-edge additive term with "
                         "its own Becke-Johnson short-range damping, no shared-budget mechanism) -- only pass "
                         "this there to deliberately mimic an mbd-slq isolation design for comparison.")
    ap.add_argument("--mbd-alpha-floor", type=float, default=1.0e-4,
                    help="Floor added to softplus(alpha_head(...)) for the MBD polarizability (default 1e-4, "
                         "matches ManyBodyDispersionSLQ's own default). Root-cause fix (2026-07-01 carbon "
                         "diagnosis): alpha collapsing near this floor shrinks the Tang-Toennies damping radius "
                         "below real covalent bond lengths, so damp saturates to ~undamped even at bond length, "
                         "a few short edges get huge raw coupling blocks, and the PD positive-definite safety "
                         "clamp (pd_rescale/pd_margin) reacts by suppressing the ENTIRE coupling operator by "
                         "~1e4x -- killing the genuine long-range/interlayer signal as collateral damage. Raising "
                         "this (e.g. to 1.0, so the damping radius exceeds a C-C bond length once beta is applied) "
                         "should let short bonds damp properly and keep the PD clamp from engaging so aggressively.")
    ap.add_argument("--mbd-pme-assignment", default="pcs", choices=["ngp", "cic", "pcs"],
                    help="Mesh assignment for experimental --mbd-operator-backend pme_fft.")
    ap.add_argument("--mbd-pme-k-norm-floor", type=float, default=1.0e-6,
                    help="Small-k floor for experimental MBD PME matvec.")
    ap.add_argument("--mbd-pme-assignment-window-floor", type=float, default=1.0e-6,
                    help="Assignment deconvolution floor for experimental MBD PME matvec.")
    ap.add_argument("--mbd-pme-ewald-alpha-prefactor", type=float, default=5.0,
                    help="Ewald alpha prefactor for experimental MBD PME matvec.")
    # atomic-energy E0 offset
    ap.add_argument("--atomic-energy-keys", default=None, help='e.g. "1,6,7,8"')
    ap.add_argument("--atomic-energy-values", default=None, help='e.g. "-430.5,-821.0,-1488.2,-2044.4"')
    ap.add_argument("--scaling", default="rms_forces_scaling",
                    choices=["std_scaling", "rms_forces_scaling", "no_scaling"],
                    help="MACE-style ScaleShiftMACE statistics for per-atom interaction energy.")
    ap.add_argument("--scaling-max-samples", type=int, default=None,
                    help="Cap the ScaleShift statistics scan at the first N training samples (force RMS / "
                         "mean atomic-inter energy converge far below the full set). None=scan all (default). "
                         "Set this for huge datasets where a full single-threaded scan over external-link H5 "
                         "is slow; samples are read in shard order so the prefix is representative.")
    ap.add_argument("--atomic-inter-scale", type=float, default=None,
                    help="Override the ScaleShiftMACE interaction-energy scale.")
    ap.add_argument("--atomic-inter-shift", type=float, default=None,
                    help="Override the ScaleShiftMACE per-atom interaction-energy shift.")
    ap.add_argument("--no-atomic-inter-shift", action="store_true",
                    help="Use a zero interaction-energy shift while keeping the selected scale statistic.")
    # optimization
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for Python, NumPy, PyTorch, DataLoader shuffle, and bucket sampler.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Optional hard cap on optimizer steps. If omitted, train for --epochs.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1,
                    help="Accumulate this many micro-batches per optimizer step. This permits a "
                         "smaller memory-resident --batch-size while preserving the effective batch size.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    ap.add_argument(
        "--optimizer-param-groups",
        default="flat",
        choices=["flat", "mace", "mace-o2"],
        help="Use flat AdamW groups or MACE-style decay/no-decay module groups.",
    )
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--amsgrad", action="store_true")
    ap.add_argument("--lr-scheduler", default="cosine",
                    choices=["cosine", "plateau", "exp", "step", "none", "ReduceLROnPlateau", "ExponentialLR"],
                    help="LR schedule. plateau/exp follow mace-torch naming; step is a legacy alias.")
    ap.add_argument("--warmup-batches", type=int, default=0)
    ap.add_argument("--warmup-start-ratio", type=float, default=0.1)
    ap.add_argument("--lr-factor", type=float, default=0.8,
                    help="ReduceLROnPlateau factor when --lr-scheduler plateau.")
    ap.add_argument("--scheduler-patience", type=int, default=50,
                    help="ReduceLROnPlateau patience in epochs.")
    ap.add_argument("--lr-scheduler-gamma", type=float, default=0.9993,
                    help="ExponentialLR gamma when --lr-scheduler exp.")
    ap.add_argument("--lr-decay-step", type=int, default=1000,
                    help="Legacy step scheduler interval in optimizer steps.")
    ap.add_argument("--lr-decay-factor", type=float, default=0.98,
                    help="Legacy step scheduler gamma.")
    ap.add_argument("--loss", default="smooth_l1", choices=["smooth_l1", "mse"],
                    help="Loss kernel used for each enabled target.")
    ap.add_argument("--loss-beta", type=float, default=0.5,
                    help="SmoothL1 beta for energy/force/stress when --loss smooth_l1.")
    ap.add_argument("--energy-weight", type=float, default=1.0)
    ap.add_argument("--force-weight", type=float, default=10.0)
    ap.add_argument("--stress-weight", type=float, default=0.0,
                    help="Weight of the stress/virial loss (0 = energy+force only; >0 enables it).")
    ap.add_argument("--force-shift-value", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=10.0,
                    help="Gradient norm clip applied before each optimizer step (<=0 disables). "
                         "Default 10.0: without it, unbounded terms (e.g. pairwise-c6's s6/c6_head, "
                         "which have no PD-safety clamp) can take one freak-batch gradient spike and "
                         "permanently poison Adam's running moments into NaN.")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="Enable EMA when >0. Example: 0.999. Saved as e3trans_ema_state_dict.")
    ap.add_argument("--ema-start-step", type=int, default=0)
    ap.add_argument("--swa", "--stage-two", action="store_true", dest="stage_two",
                    help="Enable mace-torch-style Stage Two/SWA: switch loss weights, lower LR, and average weights.")
    ap.add_argument("--swa-start-epoch", "--start-swa", "--start-stage-two",
                    type=int, default=-1, dest="swa_start_epoch",
                    help="-1 disables epoch trigger; >=0 starts Stage Two/SWA from that epoch.")
    ap.add_argument("--swa-start-step", type=int, default=-1,
                    help="-1 disables step trigger; >=0 starts Stage Two/SWA from that optimizer step.")
    ap.add_argument("--swa-lr", "--stage-two-lr", type=float, default=1e-3, dest="swa_lr",
                    help="Stage Two/SWA learning rate. Must satisfy --min-lr <= swa_lr <= --lr.")
    ap.add_argument("--swa-energy-weight", "--stage-two-energy-weight",
                    type=float, default=1000.0, dest="swa_energy_weight")
    ap.add_argument("--swa-force-weight", "--stage-two-force-weight",
                    type=float, default=100.0, dest="swa_force_weight")
    ap.add_argument("--swa-stress-weight", "--stage-two-stress-weight",
                    type=float, default=None, dest="swa_stress_weight",
                    help="Stage Two stress weight. Default: 0 if stress is off, else 10.")
    ap.add_argument("--swa-anneal-epochs", type=int, default=1,
                    help="SWALR annealing epochs after Stage Two starts.")
    ap.add_argument("--swa-anneal-strategy", default="linear", choices=["linear", "cos"],
                    help="SWALR annealing strategy.")
    ap.add_argument("--checkpoint-state-source", default="auto", choices=["auto", "raw", "ema", "swa"],
                    help="Which saved weights deploy loaders should use. auto prefers EMA, then SWA, then raw.")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    ap.set_defaults(shuffle=True)
    # make_fx + bucketing
    ap.add_argument("--train-makefx-compile", action="store_true",
                    help="Flatten forward+force-autograd and torch.compile it (faster force-loss training).")
    ap.add_argument("--makefx-buckets", default=None,
                    help='Size-bucket samples for make_fx: an int K (quantile buckets) or a comma '
                         'list of atom-count bounds (e.g. "64,128,256"). One compile per bucket.')
    ap.add_argument("--makefx-max-slots", type=int, default=8)
    ap.add_argument("--pad-nodes-to-max", action="store_true")
    ap.add_argument("--pad-edges-to-max", action="store_true")
    # misc
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--ddp", default="auto", choices=["auto", "on", "off"],
                    help="DistributedDataParallel training mode. auto enables DDP when WORLD_SIZE>1.")
    ap.add_argument("--ddp-backend", default="auto", choices=["auto", "nccl", "gloo"],
                    help="torch.distributed backend for DDP training.")
    ap.add_argument("--ddp-find-unused-parameters", action="store_true",
                    help="Pass find_unused_parameters=True to DistributedDataParallel.")
    ap.add_argument("--checkpoint", default="model.pth")
    ap.add_argument("--resume-checkpoint", default=None,
                    help="Load model weights from a previous MACE-ICTC checkpoint before training.")
    ap.add_argument("--resume-training-state", action="store_true",
                    help="With --resume-checkpoint, also restore optimizer/global_step and continue epoch numbering.")
    ap.add_argument("--finetune", action="store_true",
                    help="Warm-start: non-strict load of --resume-checkpoint (load matching backbone weights, "
                         "leave any NEWLY-ADDED long-range head freshly initialized). Fresh optimizer (do not "
                         "combine with --resume-training-state). Train the backbone first, then add native "
                         "long-range physics on top: MBD dispersion (--long-range-dispersion-mode), reciprocal "
                         "electrostatics (--long-range-mode reciprocal-spectral-v1), or multipole electrostatics "
                         "(add --long-range-reciprocal-backend mesh_fft --long-range-max-multipole-l {1,2} "
                         "--long-range-mesh-fft-full-ewald). Each head starts gentle (the electrostatics "
                         "energy_scale and the multipole gate init near zero; MBD has built-in spectral guards), "
                         "so the warm-started backbone is not destabilized. See USER_MANUAL section 12.6.")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--evals-per-epoch", type=int, default=1,
                    help="Validation passes per epoch (1=epoch-end only; 2=mid-epoch + end). "
                         "Mid-epoch passes run on all ranks and barrier, so DDP stays in sync.")
    ap.add_argument("--validation-interval-steps", type=int, default=0,
                    help="Run full validation every N optimizer steps and allow those points to "
                         "update the best checkpoint. 0 keeps epoch-based validation only.")
    ap.add_argument("--keep-checkpoints", type=int, default=0,
                    help="Retain the N most-recent VALIDATION checkpoints (saved at every val: "
                         "mid-epoch + epoch-end), rolling -- oldest deleted. 0=disabled (only the "
                         "best-on-improvement checkpoint is kept). Files: <ckpt-stem>.e<E>s<STEP>.pth.")
    ap.add_argument("--checkpoint-interval-steps", type=int, default=0,
                    help="Write <checkpoint-stem>.recovery.pth every N optimizer steps, including "
                         "optimizer/SWA state. 0 disables periodic recovery checkpoints.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Load --resume-checkpoint, run ONE validation pass, print Frmse/Ermse, and exit (no training).")
    return ap


def _override_args_from_checkpoint(args, *, preserve: set[str] | None = None):
    """Finetuning a converted checkpoint: align the model build to the checkpoint's stored
    ``model_hyperparameters`` so EVERY architecture knob (polynomial_cutoff_p, avg_num_neighbors,
    channels, lmax, readout, ...) matches the source model exactly. Prevents silent config
    mismatches that load cleanly (no missing/unexpected keys) yet corrupt the forward -- e.g. a
    wrong cutoff p applies the source radial weights through the wrong envelope (~1.8x force error).
    Only architecture knobs are aligned; optimizer/LR/data/long-range args are left as set."""
    if not getattr(args, "resume_checkpoint", None):
        return
    try:
        ck = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # best-effort peek; fall back to CLI args
        logging.warning("could not read checkpoint hyperparameters (%s); using CLI args as-is", exc)
        return
    hp = ck.get("model_hyperparameters") if isinstance(ck, dict) else None
    if not isinstance(hp, dict):
        return
    preserve = set() if preserve is None else set(preserve)
    mapping = {
        "polynomial_cutoff_p": "polynomial_cutoff_p",
        "avg_num_neighbors": "avg_num_neighbors",
        "number_of_basis": "num_basis",
        "ictd_fix_edge_lmax": "max_ell",
        "lmax": "lmax",
        "num_interaction": "num_interaction",
        "save_contraction_order": "correlation",
        "channel_in": "channels",
        "max_radius": "max_radius",
        "ictd_fix_readout_hidden_channels": "readout_hidden_channels",
        "ictd_fix_product_backend": "product_backend",
        "ictd_tp_backend": "conv_tp_backend",
        "ictd_fix_self_connection_rank": "self_connection_rank",
    }
    changed = []
    for hk, ak in mapping.items():
        if ak in preserve:
            continue
        if hk not in hp or not hasattr(args, ak):
            continue
        old, new = getattr(args, ak), hp[hk]
        if isinstance(old, bool):
            new = bool(new)
        elif isinstance(old, int):
            try:
                new = int(round(float(new)))
            except (TypeError, ValueError):
                continue
        elif isinstance(old, float):
            try:
                new = float(new)
            except (TypeError, ValueError):
                continue
        if old != new:
            setattr(args, ak, new)
            changed.append(f"{ak}: {old} -> {new}")
    # Converted models reproduce MACE's first-layer self-connection via the frozen
    # mace_first_layer_sc0 buffer, not a trainable sc -> disable the trainable one if present.
    state = ck.get("e3trans_state_dict", ck) if isinstance(ck, dict) else {}
    if (isinstance(state, dict) and "mace_first_layer_sc0" in state
            and getattr(args, "first_layer_self_connection", False)):
        args.first_layer_self_connection = False
        changed.append("first_layer_self_connection: True -> False (checkpoint carries mace_first_layer_sc0)")
    if changed:
        logging.info("resume-checkpoint: aligned model hyperparameters to the source checkpoint:")
        for c in changed:
            logging.info("    %s", c)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(raw_argv)
    explicit_conv_tp_backend = any(
        token == "--conv-tp-backend" or token.startswith("--conv-tp-backend=")
        for token in raw_argv
    )
    _override_args_from_checkpoint(
        args,
        preserve={"conv_tp_backend"} if explicit_conv_tp_backend else None,
    )
    disable_tf32()

    ddp_info = _setup_distributed(args)
    if not ddp_info["is_main"]:
        logging.getLogger().setLevel(logging.WARNING)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = _resolve_device(args.device, ddp_info)
    generator = _set_global_seed(args.seed)
    worker_init_fn = _make_worker_init_fn(args.seed)
    logging.info(
        "seed = %s ddp=%s rank=%d/%d local_rank=%d backend=%s device=%s",
        args.seed,
        ddp_info["enabled"],
        ddp_info["rank"],
        ddp_info["world_size"],
        ddp_info["local_rank"],
        ddp_info["backend"],
        device,
    )
    if args.loss == "smooth_l1" and args.loss_beta <= 0:
        raise ValueError("--loss-beta must be positive for --loss smooth_l1")
    if args.min_lr > args.lr:
        raise ValueError("--min-lr must be <= --lr")
    if args.swa_start_epoch >= 0 or args.swa_start_step >= 0:
        args.stage_two = True
    if args.stage_two and args.swa_start_epoch < 0 and args.swa_start_step < 0:
        args.swa_start_epoch = max(1, args.epochs // 4 * 3)
    if args.swa_stress_weight is None:
        args.swa_stress_weight = 10.0 if args.stress_weight > 0.0 else 0.0
    if args.stage_two and not (args.min_lr <= args.swa_lr <= args.lr):
        raise ValueError("--swa-lr must satisfy --min-lr <= --swa-lr <= --lr")

    # parse make_fx bucket spec: "6" -> int K, "64,128,256" -> explicit bounds
    makefx_buckets = None
    if args.makefx_buckets:
        mb = str(args.makefx_buckets).strip()
        makefx_buckets = [int(x) for x in mb.split(",") if x.strip()] if "," in mb else int(mb)
    node_padding_enabled = bool(args.pad_nodes_to_max or makefx_buckets is not None)
    nonlocal_physics_enabled = (
        args.long_range_mode != "none" or args.long_range_dispersion_mode != "none"
    )
    if node_padding_enabled and nonlocal_physics_enabled:
        raise ValueError(
            "node padding/makefx bucketing is not currently safe with long-range or dispersion "
            "physics: dummy atoms can enter nonlocal interactions. Disable node padding/bucketing "
            "or disable the nonlocal term."
        )

    # atomic-energy E0 (None -> trainer's H/C/N/O default)
    aek = aev = None
    if args.atomic_energy_keys and args.atomic_energy_values:
        aek = [int(x) for x in args.atomic_energy_keys.split(",") if x.strip()]
        aev = [float(x) for x in args.atomic_energy_values.split(",") if x.strip()]

    # datasets (bucketing forces fixed-shape padding inside H5Dataset)
    train_ds = H5Dataset(
        prefix=args.train_prefix, data_dir=args.data_dir,
        makefx_buckets=makefx_buckets,
        pad_nodes_to_max=args.pad_nodes_to_max,
        pad_edges_to_max=args.pad_edges_to_max,
        expected_max_radius=args.max_radius,
    )
    val_h5 = os.path.join(args.data_dir, f"processed_{args.val_prefix}.h5")
    val_ds = H5Dataset(prefix=args.val_prefix, data_dir=args.data_dir,
                       expected_max_radius=args.max_radius) if os.path.exists(val_h5) else None
    requests_stress = args.stress_weight > 0.0 or (
        args.stage_two and args.swa_stress_weight is not None and args.swa_stress_weight > 0.0
    )
    if requests_stress and getattr(train_ds, "num_stress_labels", None) == 0:
        raise ValueError(
            "stress loss was requested, but the training data contains no stress/virial labels; "
            "missing stress metadata is no longer treated as a physical zero-stress target"
        )

    # avg_num_neighbors (the message normalizer baked into the weights)
    train_h5 = os.path.join(args.data_dir, f"processed_{args.train_prefix}.h5")
    ann = args.avg_num_neighbors if args.avg_num_neighbors is not None else _avg_num_neighbors_from_h5(train_h5)
    logging.info("avg_num_neighbors = %.4f", ann)

    atomic_numbers = aek if aek is not None else [1, 6, 7, 8]
    atomic_energy_keys = aek if aek is not None else _DEFAULT_E0_KEYS
    atomic_energy_values = aev if aev is not None else _DEFAULT_E0_VALUES
    scale, shift = _resolve_atomic_inter_scale_shift(
        train_h5,
        atomic_energy_keys=atomic_energy_keys,
        atomic_energy_values=atomic_energy_values,
        scaling=args.scaling,
        atomic_inter_scale=args.atomic_inter_scale,
        atomic_inter_shift=args.atomic_inter_shift,
        no_atomic_inter_shift=args.no_atomic_inter_shift,
        max_samples=args.scaling_max_samples,
    )
    if args.angular_basis == "e3nn" and args.product_backend not in {"cueq", "ictd-pure-u"}:
        raise ValueError(
            "--angular-basis e3nn requires --product-backend cueq or ictd-pure-u. "
            "bridge-U has no e3nn-fold path; use bridge-U for canonical parity or cueq for the "
            "e3nn-folded product path."
        )
    if args.long_range_mode != "none":
        if args.long_range_reciprocal_backend == "direct_kspace" and args.long_range_boundary != "periodic":
            raise ValueError("--long-range-reciprocal-backend direct_kspace requires --long-range-boundary periodic")
        if args.long_range_kmax < 0:
            raise ValueError("--long-range-kmax must be >= 0")
        normalize_mesh_size(args.long_range_mesh_size)
        if args.long_range_source_channels <= 0:
            raise ValueError("--long-range-source-channels must be > 0")
        if args.long_range_max_multipole_l < 0:
            raise ValueError("--long-range-max-multipole-l must be >= 0")
        if args.long_range_max_multipole_l > args.lmax:
            raise ValueError("--long-range-max-multipole-l must be <= --lmax")
        if args.long_range_max_multipole_l > 0:
            if args.long_range_reciprocal_backend != "mesh_fft":
                raise ValueError("--long-range-max-multipole-l > 0 requires --long-range-reciprocal-backend mesh_fft")
            if not args.long_range_mesh_fft_full_ewald:
                raise ValueError("--long-range-max-multipole-l > 0 requires --long-range-mesh-fft-full-ewald")
    if args.long_range_dispersion_mode != "none" and args.dispersion_cutoff < 0:
        raise ValueError("--dispersion-cutoff must be >= 0")
    # Dispersion self-builds its own neighbor list at runtime (NOT baked in the H5, so it is outside
    # the rcut/H5 guard). But its cutoff should extend BEYOND the short-range graph -- a dispersion
    # range shorter than --max-radius is almost always a misconfiguration.
    if args.long_range_dispersion_mode != "none" and 0 <= args.dispersion_cutoff < args.max_radius:
        logging.warning(
            "--dispersion-cutoff (%.2f A) < --max-radius (%.2f A): the dispersion neighbor list is "
            "SHORTER than the short-range message-passing graph. Dispersion is meant to add the "
            "long-range tail BEYOND the local cutoff; a shorter range is almost certainly wrong.",
            args.dispersion_cutoff, args.max_radius,
        )
    if int(args.dispersion_max_num_neighbors) < 0:
        raise ValueError("--dispersion-max-num-neighbors must be >= 0")
    if int(args.dispersion_bruteforce_threshold) < 0:
        raise ValueError("--dispersion-bruteforce-threshold must be >= 0")
    if args.long_range_dispersion_mode == "mbd-slq":
        if int(args.dispersion_slq_num_probes) <= 0:
            raise ValueError("--dispersion-slq-num-probes must be > 0")
        if int(args.dispersion_slq_lanczos_steps) <= 0:
            raise ValueError("--dispersion-slq-lanczos-steps must be > 0")
        if args.mbd_operator_backend == "pme_fft":
            if int(args.mbd_pme_mesh_size) <= 0:
                raise ValueError("--mbd-pme-mesh-size must be > 0")
            if float(args.mbd_pme_k_norm_floor) <= 0.0:
                raise ValueError("--mbd-pme-k-norm-floor must be > 0")
            if float(args.mbd_pme_assignment_window_floor) <= 0.0:
                raise ValueError("--mbd-pme-assignment-window-floor must be > 0")
            if float(args.mbd_pme_ewald_alpha_prefactor) <= 0.0:
                raise ValueError("--mbd-pme-ewald-alpha-prefactor must be > 0")
            logging.warning(
                "--mbd-operator-backend pme_fft is an experimental reciprocal-only torch.fft "
                "backend. AOTI/LAMMPS export runs the matching C++ reciprocal PME operator; "
                "keep mbd_operator_backend, mesh size, assignment, and alpha prefactor matched."
            )
    energy_output_scale_enabled = args.scaling != "no_scaling" or args.atomic_inter_scale is not None
    energy_output_shift_enabled = (
        (args.scaling != "no_scaling" and not args.no_atomic_inter_shift)
        or args.atomic_inter_shift is not None
    )
    logging.info(
        "ScaleShiftMACE-style interaction scaling: mode=%s scale=%g%s shift=%g%s",
        args.scaling,
        scale,
        " (disabled)" if not energy_output_scale_enabled else "",
        shift,
        " (disabled)" if not energy_output_shift_enabled else "",
    )

    # model config (everything from_checkpoint reads back is captured here or in extra_hparams)
    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = args.channels
    cfg.irreps_output_conv_channels = args.channels
    cfg.lmax = args.lmax
    cfg.num_layers = args.num_layers
    cfg.max_radius = args.max_radius
    cfg.max_radius_main = args.max_radius
    cfg.number_of_basis = args.num_basis
    cfg.number_of_basis_main = args.num_basis
    cfg.function_type = args.function_type
    cfg.internal_compute_dtype = dtype

    model = build_baseline_model(
        cfg, avg_num_neighbors=ann, num_interaction=args.num_interaction,
        route=args.route, product_backend=args.product_backend, correlation=args.correlation,
        use_reduced_cg=args.use_reduced_cg,
        first_layer_self_connection=args.first_layer_self_connection,
        self_connection_rank=args.self_connection_rank,
        interaction_scale=args.interaction_scale,
        conv_tp_scale_init=args.conv_tp_scale_init,
        conv_tp_backend=args.conv_tp_backend,
        freeze_conv_tp_weight=args.freeze_conv_tp_weight,
        interaction_init=args.interaction_init,
        readout_hidden_channels=args.readout_hidden_channels,
        polynomial_cutoff_p=(None if args.polynomial_cutoff_p <= 0 else args.polynomial_cutoff_p),
        angular_basis=args.angular_basis,
        radial_sqrt_num_basis=args.radial_sqrt_num_basis, edge_lmax=args.max_ell,
        attn_heads=args.attn_heads,
        atomic_numbers=atomic_numbers, ictd_save_tp_mode=args.ictd_save_tp_mode,
        invariant_channels=args.invariant_channels,
        energy_output_scale=scale,
        energy_output_scale_enabled=energy_output_scale_enabled,
        energy_output_shift=shift,
        energy_output_shift_enabled=energy_output_shift_enabled,
        long_range_mode=args.long_range_mode,
        long_range_boundary=args.long_range_boundary,
        long_range_reciprocal_backend=args.long_range_reciprocal_backend,
        long_range_kmax=args.long_range_kmax,
        long_range_mesh_size=args.long_range_mesh_size,
        long_range_source_channels=args.long_range_source_channels,
        long_range_hidden_dim=args.long_range_hidden_dim,
        long_range_filter_hidden_dim=args.long_range_filter_hidden_dim,
        long_range_neutralize=args.long_range_neutralize,
        long_range_include_k0=args.long_range_include_k0,
        long_range_energy_partition=args.long_range_energy_partition,
        long_range_green_mode=args.long_range_green_mode,
        long_range_assignment=args.long_range_assignment,
        long_range_slab_padding_factor=args.long_range_slab_padding_factor,
        long_range_mesh_fft_full_ewald=args.long_range_mesh_fft_full_ewald,
        long_range_mesh_fft_reciprocal_only=args.long_range_mesh_fft_reciprocal_only,
        long_range_smearing_sigma=args.long_range_smearing_sigma,
        long_range_max_multipole_l=args.long_range_max_multipole_l,
        long_range_multipole_gate_init=args.long_range_multipole_gate_init,
        long_range_dispersion_mode=args.long_range_dispersion_mode,
        dispersion_cutoff=args.dispersion_cutoff,
        dispersion_max_num_neighbors=(
            int(args.dispersion_max_num_neighbors) if int(args.dispersion_max_num_neighbors) > 0 else None
        ),
        dispersion_neighbor_method=args.dispersion_neighbor_method,
        dispersion_bruteforce_threshold=int(args.dispersion_bruteforce_threshold),
        dispersion_allow_large_bruteforce_fallback=bool(args.dispersion_allow_large_bruteforce_fallback),
        dispersion_slq_num_probes=args.dispersion_slq_num_probes,
        dispersion_slq_lanczos_steps=args.dispersion_slq_lanczos_steps,
        mbd_operator_backend=args.mbd_operator_backend,
        mbd_pme_mesh_size=args.mbd_pme_mesh_size,
        mbd_anisotropic_polarizability=args.mbd_anisotropic_polarizability,
        mbd_learnable_energy_scale=args.mbd_learnable_energy_scale,
        mbd_alpha_floor=args.mbd_alpha_floor,
        dispersion_min_cutoff=args.dispersion_min_cutoff,
        mbd_pme_assignment=args.mbd_pme_assignment,
        mbd_pme_k_norm_floor=args.mbd_pme_k_norm_floor,
        mbd_pme_assignment_window_floor=args.mbd_pme_assignment_window_floor,
        mbd_pme_ewald_alpha_prefactor=args.mbd_pme_ewald_alpha_prefactor,
        device=device, dtype=dtype,
    )
    if args.mace_compatible_random_init:
        _apply_mace_compatible_random_init(
            model,
            args,
            atomic_numbers=atomic_numbers,
            avg_num_neighbors=ann,
            scale=scale,
            shift=shift,
            device=device,
            dtype=dtype,
        )
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("model: %s route=%s channels=%d lmax=%d num_interaction=%d params=%d",
                 type(model).__name__, args.route, args.channels, args.lmax, args.num_interaction, n_params)

    if ddp_info["enabled"]:
        if device.type != "cuda":
            model = DistributedDataParallel(
                model,
                broadcast_buffers=False,
                find_unused_parameters=bool(args.ddp_find_unused_parameters),
            )
        else:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=bool(args.ddp_find_unused_parameters),
            )

    # dataloaders
    if makefx_buckets is not None:
        sampler = BucketBatchSampler(
            train_ds.sample_bucket, batch_size=args.batch_size,
            shuffle=args.shuffle, drop_last=True,
            num_replicas=ddp_info["world_size"],
            rank=ddp_info["rank"],
            seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  collate_fn=collate_fn_h5, num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)
        logging.info("bucketing ON: %d buckets, bounds=%s",
                     len(set(train_ds.sample_bucket)), getattr(train_ds, "_bucket_bounds", None))
    else:
        sampler = (
            DistributedSampler(
                train_ds,
                num_replicas=ddp_info["world_size"],
                rank=ddp_info["rank"],
                shuffle=args.shuffle,
                seed=args.seed,
                drop_last=False,
            )
            if ddp_info["enabled"] else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(args.shuffle and sampler is None),
            sampler=sampler,
            drop_last=False,
            collate_fn=collate_fn_h5,
            num_workers=args.num_workers,
            generator=(generator if sampler is None else None),
            worker_init_fn=worker_init_fn,
        )
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn_h5, num_workers=args.num_workers,
                             worker_init_fn=worker_init_fn)
                  if val_ds is not None else None)

    # the architecture choices from_checkpoint reads that are NOT ModelConfig fields
    extra_hparams = dict(
        num_interaction=args.num_interaction,
        invariant_channels=args.invariant_channels,
        ictd_fix_route=args.route,
        ictd_fix_product_backend=args.product_backend,
        ictd_fix_use_reduced_cg=bool(args.use_reduced_cg),
        ictd_fix_first_layer_self_connection=bool(args.first_layer_self_connection),
        ictd_fix_interaction_scale=args.interaction_scale,
        ictd_fix_conv_tp_scale_init=args.conv_tp_scale_init,
        ictd_tp_backend=args.conv_tp_backend,
        ictd_fix_freeze_conv_tp_weight=bool(args.freeze_conv_tp_weight),
        ictd_fix_interaction_init=args.interaction_init,
        mace_compatible_random_init=bool(args.mace_compatible_random_init),
        ictd_fix_readout_hidden_channels=int(args.readout_hidden_channels),
        angular_basis=args.angular_basis,
        ictd_fix_edge_lmax=(args.lmax if args.max_ell is None else args.max_ell),
        save_contraction_order=args.correlation,
        ictd_save_tp_mode=args.ictd_save_tp_mode,
        ictd_fix_interaction_attn_heads=args.attn_heads,
        radial_sqrt_num_basis=bool(args.radial_sqrt_num_basis),
        polynomial_cutoff_p=(None if args.polynomial_cutoff_p <= 0 else int(args.polynomial_cutoff_p)),
        optimizer_param_groups=args.optimizer_param_groups,
        avg_num_neighbors=float(ann),
        energy_output_scale_enabled=bool(energy_output_scale_enabled),
        energy_output_scale=float(scale),
        energy_output_shift_enabled=bool(energy_output_shift_enabled),
        energy_output_shift=float(shift),
        long_range_mode=args.long_range_mode,
        long_range_boundary=args.long_range_boundary,
        long_range_neutralize=bool(args.long_range_neutralize),
        long_range_hidden_dim=int(args.long_range_hidden_dim),
        long_range_filter_hidden_dim=int(args.long_range_filter_hidden_dim),
        long_range_kmax=int(args.long_range_kmax),
        long_range_mesh_size=normalize_mesh_size(args.long_range_mesh_size),
        long_range_slab_padding_factor=int(args.long_range_slab_padding_factor),
        long_range_include_k0=bool(args.long_range_include_k0),
        long_range_source_channels=int(args.long_range_source_channels),
        long_range_reciprocal_backend=args.long_range_reciprocal_backend,
        long_range_energy_partition=args.long_range_energy_partition,
        long_range_green_mode=args.long_range_green_mode,
        long_range_assignment=args.long_range_assignment,
        long_range_mesh_fft_full_ewald=bool(args.long_range_mesh_fft_full_ewald),
        long_range_mesh_fft_reciprocal_only=bool(args.long_range_mesh_fft_reciprocal_only),
        long_range_smearing_sigma=args.long_range_smearing_sigma,
        long_range_max_multipole_l=int(args.long_range_max_multipole_l),
        long_range_multipole_gate_init=float(args.long_range_multipole_gate_init),
        long_range_dispersion_mode=args.long_range_dispersion_mode,
        long_range_dispersion=bool(args.long_range_dispersion_mode != "none"),
        dispersion_cutoff=float(args.dispersion_cutoff),
        dispersion_max_num_neighbors=(
            int(args.dispersion_max_num_neighbors) if int(args.dispersion_max_num_neighbors) > 0 else None
        ),
        dispersion_neighbor_method=args.dispersion_neighbor_method,
        dispersion_bruteforce_threshold=int(args.dispersion_bruteforce_threshold),
        dispersion_allow_large_bruteforce_fallback=bool(args.dispersion_allow_large_bruteforce_fallback),
        dispersion_slq_num_probes=int(args.dispersion_slq_num_probes),
        dispersion_slq_lanczos_steps=int(args.dispersion_slq_lanczos_steps),
        mbd_operator_backend=args.mbd_operator_backend,
        mbd_pme_mesh_size=int(args.mbd_pme_mesh_size),
        mbd_anisotropic_polarizability=bool(args.mbd_anisotropic_polarizability),
        mbd_learnable_energy_scale=bool(args.mbd_learnable_energy_scale),
        mbd_alpha_floor=float(args.mbd_alpha_floor),
        dispersion_min_cutoff=float(args.dispersion_min_cutoff),
        mbd_pme_assignment=args.mbd_pme_assignment,
        mbd_pme_k_norm_floor=float(args.mbd_pme_k_norm_floor),
        mbd_pme_assignment_window_floor=float(args.mbd_pme_assignment_window_floor),
        mbd_pme_ewald_alpha_prefactor=float(args.mbd_pme_ewald_alpha_prefactor),
    )

    trainer = ForceTrainer(
        model, train_loader, val_loader=val_loader, device=device, config=cfg,
        dtype=dtype, max_radius=args.max_radius,
        energy_weight=args.energy_weight, force_weight=args.force_weight,
        stress_weight=args.stress_weight, force_shift_value=args.force_shift_value,
        loss_type=args.loss, loss_beta=args.loss_beta,
        atomic_energy_keys=aek, atomic_energy_values=aev,
        learning_rate=args.lr, min_learning_rate=args.min_lr, weight_decay=args.weight_decay,
        optimizer_type=args.optimizer, optimizer_param_groups=args.optimizer_param_groups,
        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps, amsgrad=args.amsgrad, lr_scheduler=args.lr_scheduler,
        warmup_batches=args.warmup_batches, warmup_start_ratio=args.warmup_start_ratio,
        lr_factor=args.lr_factor, scheduler_patience=args.scheduler_patience,
        lr_scheduler_gamma=args.lr_scheduler_gamma,
        lr_decay_step=args.lr_decay_step, lr_decay_factor=args.lr_decay_factor,
        epochs=args.epochs, max_steps=args.max_steps, max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ema_decay=args.ema_decay, ema_start_step=args.ema_start_step,
        stage_two_enabled=args.stage_two,
        swa_start_epoch=args.swa_start_epoch, swa_start_step=args.swa_start_step,
        swa_lr=args.swa_lr, swa_energy_weight=args.swa_energy_weight,
        swa_force_weight=args.swa_force_weight, swa_stress_weight=args.swa_stress_weight,
        swa_anneal_epochs=args.swa_anneal_epochs, swa_anneal_strategy=args.swa_anneal_strategy,
        checkpoint_state_source=args.checkpoint_state_source,
        train_makefx_compile=args.train_makefx_compile,
        operator_plan=args.operator_plan,
        generated_layout=args.generated_layout,
        training_message_linear_layout=args.training_message_linear_layout,
        electrostatics_backend=args.electrostatics_backend,
        backward_targets=args.backward_targets,
        require_train_makefx_compile=bool(args.train_makefx_compile and args.product_backend == "cueq"),
        makefx_max_slots=args.makefx_max_slots,
        train_sampler=sampler, checkpoint_path=args.checkpoint, log_interval=args.log_interval,
        evals_per_epoch=args.evals_per_epoch,
        validation_interval_steps=args.validation_interval_steps,
        keep_checkpoints=args.keep_checkpoints,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        extra_hparams=extra_hparams,
        distributed=ddp_info["enabled"],
        rank=ddp_info["rank"],
        world_size=ddp_info["world_size"],
        main_process=ddp_info["is_main"],
    )
    start_epoch = 0
    if args.resume_checkpoint:
        start_epoch = trainer.load_checkpoint(
            args.resume_checkpoint,
            training_state=bool(args.resume_training_state),
            strict=not args.finetune,
        )
    if args.eval_only:
        va = trainer._val_pass()
        stress_text = (
            f"\n  Stress: RMSE={va['stress_rmse']:.6g} eV/A^3   "
            f"MAE={va['stress_mae']:.6g} eV/A^3"
            if args.stress_weight > 0.0 else ""
        )
        print(f"[EVAL-ONLY] ckpt={args.resume_checkpoint}\n"
              f"  val total_loss={va['total_loss']:.6g}\n"
              f"  RMSE: Frmse={va['force_rmse']:.6g} eV/A   Ermse={va['energy_rmse_avg']:.6g} eV/atom\n"
              f"  MAE:  Fmae={va['force_mae']:.6g} eV/A   Emae={va['energy_mae_avg']:.6g} eV/atom"
              f"{stress_text}",
              flush=True)
        return
    try:
        best = trainer.fit(start_epoch=start_epoch)
        logging.info("done. best loss = %.6f. checkpoint -> %s", best, args.checkpoint)
        if ddp_info["enabled"] and dist.is_initialized():
            dist.barrier()
    finally:
        if ddp_info["enabled"] and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
