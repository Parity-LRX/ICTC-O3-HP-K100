"""Compact energy+force(+stress) trainer for MACE-ICTC (make_fx compile + size-bucketing).

This is the baseline energy+force training path extracted from the FSCETP trainer,
kept numerically faithful but stripped of the modes MACE-ICTC does not ship
(physical-tensor heads / multi-fidelity / external field / CUDA-graph /
compiled-autograd). What it preserves, verbatim in math:

  * energy:  ``E_mol = scatter_sum(E_per_atom, batch_idx) + scatter_sum(E0[A], batch_idx)``
             where ``E0`` is the per-type atomic-energy offset (the dataset stores
             energies already-baseline-subtracted; we add it back here).
  * forces:  ``f = -dE_mol/dpos`` via ``autograd.grad(E_mol.sum(), pos, create_graph=training)``.
             ``E0`` is pos-independent, so this equals ``-dE_conv/dpos``.
  * stress:  optional (``stress_weight > 0``) -- ``sigma = dE/dstrain / V`` via the
             strain-derivative trick: deform positions+cell by ``I + sym(strain)`` and
             differentiate w.r.t. ``strain`` at ``strain=0`` (FSCETP convention).
  * loss:    per-atom-normalized ``SmoothL1(beta=loss_beta)`` on energy/force/stress
             by default (or MSE when requested), ``force_ref *= force_shift_value``;
             ``total = a*E + b*F (+ c*stress)``.

The make_fx fast path flattens ``forward + inner force-autograd`` (and the strain
derivative when stress is on) into one FX graph
and ``torch.compile``s it, so the outer ``loss.backward()`` is a single ordinary
backward (sidestepping the 2nd-order-through-autograd.grad limit). It is keyed per
input shape by :class:`CompiledForceCache`; pair it with the ``BucketBatchSampler``
so each size-bucket is exactly one fixed shape = one compile.

Node padding (``pad_nodes_to_max`` / bucketing) is a numeric no-op: dummy edges get
length >> max_radius (masked by the model), and ``atom_mask`` zeros dummy atoms in the
energy sum and excludes them from the loss denominators.
"""

from __future__ import annotations

import logging
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.swa_utils import SWALR

from mace_ictc.utils.scatter import scatter
from mace_ictc.utils.tensor_utils import map_tensor_values
from mace_ictc.utils.fidelity import smooth_l1_loss_stats

log = logging.getLogger(__name__)

# H/C/N/O defaults — the same fallback the FSCETP trainer uses when no E0 is supplied.
_DEFAULT_E0_KEYS = [1, 6, 7, 8]
_DEFAULT_E0_VALUES = [-430.53299511, -821.03326787, -1488.18856918, -2044.3509823]


def disable_tf32() -> None:
    """Force full IEEE float32 matmul/convolution precision on CUDA."""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


class ForceTrainer:
    """Energy+force trainer with optional make_fx compile + size-bucketing."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        *,
        val_loader=None,
        device="cuda",
        config=None,
        dtype: torch.dtype | None = None,
        max_radius: float = 5.0,
        tensor_product_mode: str = "pure-cartesian-ictd-fix",
        # loss
        energy_weight: float = 1.0,
        force_weight: float = 10.0,
        stress_weight: float = 0.0,
        force_shift_value: float = 1.0,
        loss_type: str = "smooth_l1",
        loss_beta: float = 0.5,
        atomic_energy_keys=None,
        atomic_energy_values=None,
        # optimization
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-6,
        weight_decay: float = 0.0,
        optimizer_type: str = "adamw",
        optimizer_param_groups: str = "flat",
        neutral_learning_rate_scale: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        amsgrad: bool = False,
        lr_scheduler: str = "cosine",
        warmup_batches: int = 0,
        warmup_start_ratio: float = 0.1,
        lr_factor: float = 0.8,
        scheduler_patience: int = 50,
        lr_scheduler_gamma: float = 0.9993,
        lr_decay_step: int = 1000,
        lr_decay_factor: float = 0.98,
        epochs: int = 1,
        max_steps: int | None = None,
        max_grad_norm: float | None = None,
        gradient_accumulation_steps: int = 1,
        # averaged weights
        ema_decay: float = 0.0,
        ema_start_step: int = 0,
        stage_two_enabled: bool = False,
        swa_start_epoch: int = -1,
        swa_start_step: int = -1,
        swa_lr: float | None = None,
        swa_energy_weight: float | None = None,
        swa_force_weight: float | None = None,
        swa_stress_weight: float | None = None,
        swa_anneal_epochs: int = 1,
        swa_anneal_strategy: str = "linear",
        checkpoint_state_source: str = "auto",
        # make_fx
        train_makefx_compile: bool = False,
        require_train_makefx_compile: bool = False,
        makefx_max_slots: int = 8,
        operator_plan: str = "auto",
        generated_layout: str = "packed",
        training_message_linear_layout: str = "node_component",
        electrostatics_backend: str | None = None,
        backward_targets: str = "auto",
        # plumbing
        train_sampler=None,
        log_interval: int = 10,
        evals_per_epoch: int = 1,
        validation_interval_steps: int = 0,
        eval_top_k_errors: int = 0,
        checkpoint_path: str | None = None,
        keep_checkpoints: int = 0,
        checkpoint_interval_steps: int = 0,
        extra_hparams: dict | None = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        main_process: bool = True,
    ):
        disable_tf32()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.config = config
        self.dtype = dtype if dtype is not None else next(model.parameters()).dtype
        self.max_radius = float(max_radius)
        self.tensor_product_mode = tensor_product_mode

        self.a = float(energy_weight)
        self.b = float(force_weight)
        self.c = float(stress_weight)  # stress/virial loss weight; 0 disables the stress path
        self.force_shift_value = float(force_shift_value)
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"smooth_l1", "mse"}:
            raise ValueError(f"unsupported loss_type {loss_type!r}; use 'smooth_l1' or 'mse'")
        self.loss_beta = float(loss_beta)
        if self.loss_type == "smooth_l1" and self.loss_beta <= 0:
            raise ValueError("loss_beta must be positive for smooth_l1 loss")

        keys = atomic_energy_keys if atomic_energy_keys is not None else _DEFAULT_E0_KEYS
        values = atomic_energy_values if atomic_energy_values is not None else _DEFAULT_E0_VALUES
        self.keys = torch.as_tensor(list(keys), dtype=torch.long, device=self.device)
        self.values = torch.as_tensor(list(values), dtype=self.dtype, device=self.device)

        self.learning_rate = float(learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must be non-negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be <= learning_rate")
        self.weight_decay = float(weight_decay)
        self.optimizer_type = str(optimizer_type).lower()
        self.optimizer_param_groups = str(optimizer_param_groups or "flat").lower().replace("_", "-")
        if self.optimizer_param_groups not in {"flat", "mace", "mace-o2"}:
            raise ValueError(
                "optimizer_param_groups must be 'flat', 'mace', or 'mace-o2'"
            )
        self.neutral_learning_rate_scale = float(neutral_learning_rate_scale)
        if self.neutral_learning_rate_scale <= 0.0:
            raise ValueError("neutral_learning_rate_scale must be positive")
        if (
            self.neutral_learning_rate_scale != 1.0
            and self.optimizer_param_groups not in {"mace", "mace-o2"}
        ):
            raise ValueError(
                "neutral_learning_rate_scale requires MACE-style optimizer groups"
            )
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        if not (0.0 <= self.adam_beta1 < 1.0 and 0.0 <= self.adam_beta2 < 1.0):
            raise ValueError("Adam betas must satisfy 0 <= beta < 1")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        self.amsgrad = bool(amsgrad)
        self.lr_scheduler_kind = self._normalize_scheduler_kind(lr_scheduler)
        self.warmup_batches = int(max(0, warmup_batches))
        self.warmup_start_ratio = float(warmup_start_ratio)
        if self.warmup_start_ratio < 0.0 or self.warmup_start_ratio > 1.0:
            raise ValueError("warmup_start_ratio must be in [0, 1]")
        self.lr_factor = float(lr_factor)
        if self.lr_factor <= 0.0 or self.lr_factor >= 1.0:
            raise ValueError("lr_factor must satisfy 0 < lr_factor < 1")
        self.scheduler_patience = int(max(0, scheduler_patience))
        self.lr_scheduler_gamma = float(lr_scheduler_gamma)
        if self.lr_scheduler_gamma <= 0.0 or self.lr_scheduler_gamma > 1.0:
            raise ValueError("lr_scheduler_gamma must satisfy 0 < gamma <= 1")
        self.lr_decay_step = int(lr_decay_step)
        self.lr_decay_factor = float(lr_decay_factor)

        self.epochs = int(epochs)
        self.max_steps = int(max_steps) if max_steps is not None and int(max_steps) > 0 else None
        self.global_step = 0
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.log_interval = int(log_interval)
        self.evals_per_epoch = max(1, int(evals_per_epoch))
        self.validation_interval_steps = max(0, int(validation_interval_steps))
        self.best_validation_loss = math.inf
        self.eval_top_k_errors = max(0, int(eval_top_k_errors))
        self.last_val_top_errors: list[dict[str, float | int | str]] = []
        self.checkpoint_path = checkpoint_path
        self.keep_checkpoints = max(0, int(keep_checkpoints))
        self.checkpoint_interval_steps = max(0, int(checkpoint_interval_steps))
        self._rolling_ckpts: list[str] = []
        self.metrics_csv_path = (
            os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "loss.csv")
            if checkpoint_path else None
        )
        self.train_sampler = train_sampler
        self.distributed = bool(distributed)
        self.rank = int(rank)
        self.world_size = int(max(world_size, 1))
        self.main_process = bool(main_process)
        # Construction choices the deploy-side from_checkpoint reads but that are NOT
        # ModelConfig fields (save_contraction_order / ictd_save_tp_mode / invariant_channels /
        # radial_sqrt_num_basis / avg_num_neighbors ...). Merged into model_hyperparameters
        # so the saved checkpoint rebuilds the exact same architecture.
        self.extra_hparams = dict(extra_hparams or {})

        # make_fx state (cache held outside the module tree so its duplicated flat
        # param views stay out of parameter discovery / DDP).
        self.train_makefx_compile = bool(train_makefx_compile)
        if self.train_makefx_compile:
            # The MBD-SLQ block-Gershgorin skip gate uses a data-dependent Python bool that make_fx
            # cannot trace; switch the dispersion modules to the branchless always-power-iteration
            # path (clamp(max=1) gives the identical zero-bias result, and the power iteration is
            # compiled here anyway, so the eager skip-gate is unnecessary under make_fx).
            for _m in self.model.modules():
                if hasattr(_m, "pd_amortize"):
                    _m.pd_amortize = False
                if hasattr(_m, "mbd_static_batch"):
                    _m.mbd_static_batch = True   # equal-size (bucketing-padded) graphs -> reshape, no .nonzero
        self.require_train_makefx_compile = bool(require_train_makefx_compile)
        from mace_ictc.models.operator_plans import configure_mace_runtime_plan
        self.operator_plan_info = configure_mace_runtime_plan(
            self.model, policy=operator_plan, training=True, capture=self.train_makefx_compile,
            generated_layout=generated_layout,
            training_message_linear_layout=training_message_linear_layout,
        )
        if electrostatics_backend == "hip" and self.train_makefx_compile:
            raise ValueError("Native HIP electrostatics requires eager training")
        mesh_backend = "reference" if operator_plan == "reference" or self.train_makefx_compile else electrostatics_backend
        if mesh_backend is not None:
            from mace_ictc.hp.electrostatics import configure_electrostatics
            self.operator_plan_info["electrostatics"] = configure_electrostatics(self.model, backend=mesh_backend)
        if backward_targets not in ("auto", "parameters", "all"):
            raise ValueError("backward_targets must be auto, parameters or all")
        self.backward_targets = (
            "parameters" if self.operator_plan_info["selected"] == "generated" else "all"
        ) if backward_targets == "auto" else backward_targets
        log.info("operator plan: %s; backward targets: %s; fallback reasons: %s",
                 self.operator_plan_info["selected"], self.backward_targets,
                 "; ".join(self.operator_plan_info["reasons"]) or "none")
        self._makefx_max_slots = int(makefx_max_slots)
        self._makefx_disabled = False
        object.__setattr__(self, "_makefx_cache", None)
        self._dispersion_graph_observation = {
            "observed_batches": 0,
            "observed_edges": 0,
            "single_image_cell_batches": 0,
            "non_single_image_cell_batches": 0,
            "self_image_edges": 0,
            "multi_image_shift_edges": 0,
            "builder_explicit_batches": 0,
            "builder_auto_bruteforce_batches": 0,
            "builder_auto_torch_cluster_batches": 0,
            "builder_auto_cell_batches": 0,
            "builder_auto_bruteforce_fallback_batches": 0,
            "builder_bruteforce_batches": 0,
            "builder_cell_batches": 0,
            "builder_unknown_batches": 0,
        }
        self._dispersion_graph_observation_base = {
            "observed_batches": 0,
            "observed_edges": 0,
            "single_image_cell_batches": 0,
            "non_single_image_cell_batches": 0,
            "self_image_edges": 0,
            "multi_image_shift_edges": 0,
            "builder_explicit_batches": 0,
            "builder_auto_bruteforce_batches": 0,
            "builder_auto_torch_cluster_batches": 0,
            "builder_auto_cell_batches": 0,
            "builder_auto_bruteforce_fallback_batches": 0,
            "builder_bruteforce_batches": 0,
            "builder_cell_batches": 0,
            "builder_unknown_batches": 0,
        }
        self._dispersion_graph_observation_synced = None
        self._last_dispersion_neighbor_build_info = None

        self.ema_decay = float(ema_decay)
        if self.ema_decay < 0.0 or self.ema_decay >= 1.0:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        self.ema_start_step = max(0, int(ema_start_step))
        self._ema_state = None

        self.swa_start_epoch = int(swa_start_epoch)
        self.swa_start_step = int(swa_start_step)
        if self.swa_start_epoch < -1:
            raise ValueError("swa_start_epoch must be -1 (disabled) or >= 0")
        if self.swa_start_step < -1:
            raise ValueError("swa_start_step must be -1 (disabled) or >= 0")
        self.stage_two_enabled = bool(
            stage_two_enabled or self.swa_start_epoch >= 0 or self.swa_start_step >= 0
        )
        if self.stage_two_enabled and self.swa_start_epoch < 0 and self.swa_start_step < 0:
            self.swa_start_epoch = max(1, self.epochs // 4 * 3)
        if self.stage_two_enabled and self.swa_start_epoch > self.epochs and self.max_steps is None:
            log.warning(
                "stage two start epoch %d is greater than total epochs %d; stage two will not activate",
                self.swa_start_epoch,
                self.epochs,
            )
        self.swa_lr = self.learning_rate if swa_lr is None else float(swa_lr)
        if self.swa_lr <= 0:
            raise ValueError("swa_lr must be positive")
        if self.stage_two_enabled and not (self.min_learning_rate <= self.swa_lr <= self.learning_rate):
            raise ValueError("swa_lr must satisfy min_learning_rate <= swa_lr <= learning_rate")
        self.stage_two_energy_weight = (
            self.a if swa_energy_weight is None else float(swa_energy_weight)
        )
        self.stage_two_force_weight = (
            self.b if swa_force_weight is None else float(swa_force_weight)
        )
        self.stage_two_stress_weight = (
            self.c if swa_stress_weight is None else float(swa_stress_weight)
        )
        self.swa_anneal_epochs = int(max(1, swa_anneal_epochs))
        self.swa_anneal_strategy = str(swa_anneal_strategy)
        if self.swa_anneal_strategy not in {"linear", "cos"}:
            raise ValueError("swa_anneal_strategy must be 'linear' or 'cos'")
        self._stage_two_active = False
        self._stage_two_activated_epoch = None
        self._stage_two_activated_step = None
        self._swa_scheduler = None
        self._swa_state = None
        self._swa_n = 0
        self.checkpoint_state_source = str(checkpoint_state_source or "auto").lower()
        if self.checkpoint_state_source not in {"auto", "raw", "ema", "swa"}:
            raise ValueError("checkpoint_state_source must be one of auto/raw/ema/swa")

        self._build_optimizer(self.optimizer_type, self.learning_rate, self.weight_decay, self.amsgrad)
        self._build_scheduler(
            self.lr_scheduler_kind, self.warmup_batches, self.warmup_start_ratio,
            self.lr_decay_step, self.lr_decay_factor, self.min_learning_rate,
        )

    # ------------------------------------------------------------------ setup
    @property
    def _raw_model(self) -> torch.nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _dist_ready(self) -> bool:
        return (
            self.distributed
            and self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )

    @staticmethod
    def _new_metric_sums() -> dict[str, float]:
        return {
            "num_batches": 0.0,
            "energy_loss_sum": 0.0, "energy_count": 0.0,
            "force_loss_sum": 0.0, "force_count": 0.0,
            "stress_loss_sum": 0.0, "stress_count": 0.0,
            "energy_sq_sum": 0.0, "energy_abs_sum": 0.0,
            "force_sq_sum": 0.0, "force_abs_sum": 0.0,
            "stress_sq_sum": 0.0, "stress_abs_sum": 0.0,
        }

    @staticmethod
    def _accumulate_metric_sums(sums: dict[str, float], out: dict) -> None:
        sums["num_batches"] += 1.0
        for key, value in out["_metric_sums"].items():
            sums[key] += float(value)

    def _finalize_metric_sums(self, sums: dict[str, float], *, require_data: bool) -> dict[str, float]:
        if sums["num_batches"] <= 0:
            if require_data:
                raise ValueError("validation loader produced no batches")
            return {
                "total_loss": 0.0, "energy_loss": 0.0, "force_loss": 0.0,
                "stress_loss": 0.0, "force_rmse": 0.0, "energy_rmse_avg": 0.0,
                "force_mae": 0.0, "energy_mae_avg": 0.0,
                "stress_rmse": 0.0, "stress_mae": 0.0,
            }

        def mean(sum_key: str, count_key: str) -> float:
            count = sums[count_key]
            return sums[sum_key] / count if count > 0 else 0.0

        energy_loss = mean("energy_loss_sum", "energy_count")
        force_loss = mean("force_loss_sum", "force_count")
        stress_loss = mean("stress_loss_sum", "stress_count")
        energy_mse = mean("energy_sq_sum", "energy_count")
        force_mse = mean("force_sq_sum", "force_count")
        stress_mse = mean("stress_sq_sum", "stress_count")
        return {
            "total_loss": self.a * energy_loss + self.b * force_loss + self.c * stress_loss,
            "energy_loss": energy_loss,
            "force_loss": force_loss,
            "stress_loss": stress_loss,
            "force_rmse": math.sqrt(max(force_mse, 0.0)),
            "energy_rmse_avg": math.sqrt(max(energy_mse, 0.0)),
            "force_mae": mean("force_abs_sum", "force_count"),
            "energy_mae_avg": mean("energy_abs_sum", "energy_count"),
            "stress_rmse": math.sqrt(max(stress_mse, 0.0)),
            "stress_mae": mean("stress_abs_sum", "stress_count"),
        }

    def _reduce_epoch_metrics(self, sums: dict[str, float]) -> dict[str, float]:
        if self._dist_ready():
            keys = list(sums)
            tensor = torch.tensor([sums[k] for k in keys], dtype=torch.float64, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            sums = {key: float(tensor[i].item()) for i, key in enumerate(keys)}
        return self._finalize_metric_sums(sums, require_data=False)

    def _broadcast_float(self, value: float | None) -> float:
        if not self._dist_ready():
            return float(value if value is not None else math.inf)
        v = float(value if (self.main_process and value is not None) else math.inf)
        t = torch.tensor([v], dtype=torch.float64, device=self.device)
        dist.broadcast(t, src=0)
        return float(t.item())

    def _mace_style_param_groups(
        self,
        lr: float,
        weight_decay: float,
        *,
        decay_o2_modules: bool = False,
    ):
        templates = {
            "embedding": {"params": [], "weight_decay": 0.0, "lr": lr},
            "interactions_decay": {"params": [], "weight_decay": weight_decay, "lr": lr},
            "interactions_no_decay": {"params": [], "weight_decay": 0.0, "lr": lr},
            "products": {"params": [], "weight_decay": weight_decay, "lr": lr},
            "readouts": {"params": [], "weight_decay": 0.0, "lr": lr},
            "other_no_decay": {"params": [], "weight_decay": 0.0, "lr": lr},
        }
        groups = {name: dict(group) for name, group in templates.items()}
        seen: set[int] = set()
        for name, param in self._raw_model.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            is_neutral = name.startswith("neutral_backbone.")
            classified_name = (
                name.removeprefix("neutral_backbone.") if is_neutral else name
            )
            if classified_name.startswith(
                ("node_embedding", "density_species_embedding")
            ):
                key = "embedding"
            elif classified_name.startswith("products."):
                key = "products"
            elif decay_o2_modules and classified_name.startswith(
                ("density_layers.", "refinement_densities.")
            ):
                key = (
                    "other_no_decay"
                    if classified_name.endswith(
                        ("residual_scale", ".bias")
                    )
                    else "products"
                )
            elif decay_o2_modules and classified_name.startswith(
                ("edge_layers.", "refinement_edge_layers.")
            ):
                if (
                    classified_name.endswith(".bias")
                    or "norm" in classified_name
                    or "scale" in classified_name
                ):
                    key = "interactions_no_decay"
                else:
                    key = "interactions_decay"
            elif classified_name.startswith(("layer_energy_readouts", "last_layer_energy_readout", "readouts")):
                key = "readouts"
            elif classified_name.startswith("interactions."):
                if classified_name.endswith(".bias") or "norm" in classified_name or "scale" in classified_name:
                    key = "interactions_no_decay"
                else:
                    key = "interactions_decay"
            else:
                key = "other_no_decay"
            if is_neutral:
                neutral_key = f"neutral_{key}"
                if neutral_key not in groups:
                    groups[neutral_key] = {
                        **templates[key],
                        "params": [],
                        "lr": lr * self.neutral_learning_rate_scale,
                    }
                key = neutral_key
            groups[key]["params"].append(param)
        return [dict(name=name, **group) for name, group in groups.items() if group["params"]]

    def _build_optimizer(self, optimizer_type, lr, weight_decay, amsgrad):
        if self.optimizer_param_groups in {"mace", "mace-o2"}:
            params = self._mace_style_param_groups(
                lr,
                weight_decay,
                decay_o2_modules=self.optimizer_param_groups == "mace-o2",
            )
            base_weight_decay = 0.0
        else:
            params = list(self.model.parameters())
            base_weight_decay = weight_decay
        kind = str(optimizer_type).lower()
        common = dict(lr=lr, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps,
                      weight_decay=base_weight_decay, amsgrad=bool(amsgrad))
        if kind == "adam":
            self.optimizer = torch.optim.Adam(params, **common)
        else:
            self.optimizer = torch.optim.AdamW(params, **common)

    @staticmethod
    def _normalize_scheduler_kind(kind):
        raw = str(kind or "none").strip().lower().replace("_", "").replace("-", "")
        aliases = {
            "none": "none",
            "off": "none",
            "cosine": "cosine",
            "cosineannealinglr": "cosine",
            "exp": "exp",
            "exponential": "exp",
            "exponentiallr": "exp",
            "plateau": "plateau",
            "reducelronplateau": "plateau",
            "step": "step",
            "steplr": "step",
        }
        if raw not in aliases:
            raise ValueError("lr_scheduler must be one of none/cosine/exp/plateau/step")
        return aliases[raw]

    def _build_scheduler(self, kind, warmup_batches, warmup_start_ratio,
                         decay_step, decay_factor, min_lr):
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = max(1, self.max_steps or (self.epochs * steps_per_epoch))
        self._warmup_start_lr = max(min_lr, self.learning_rate * float(warmup_start_ratio))
        self._post_warmup_step = 0
        self._cosine_total_steps = max(1, total_steps - int(max(0, warmup_batches)))
        self._scheduler = None
        self._scheduler_kind = kind
        self._scheduler_interval = "none"
        if kind == "plateau":
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.lr_factor,
                patience=self.scheduler_patience,
                min_lr=min_lr,
            )
            self._scheduler_interval = "epoch"
        elif kind == "exp":
            self._scheduler_interval = "epoch"
        elif kind in {"cosine", "step"}:
            self._scheduler_interval = "step"
        if warmup_batches > 0:
            self._set_optimizer_lrs(self._warmup_start_lr)
        else:
            self._set_optimizer_lrs(self.learning_rate)

    def _set_optimizer_lrs(self, lr):
        lr = min(self.learning_rate, max(self.min_learning_rate, float(lr)))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _clamp_optimizer_lrs(self):
        for group in self.optimizer.param_groups:
            group["lr"] = min(self.learning_rate, max(self.min_learning_rate, float(group["lr"])))

    def _step_scheduler_after_batch(self):
        if self._stage_two_active:
            return
        if self.warmup_batches > 0 and self.global_step <= self.warmup_batches:
            frac = min(float(self.global_step) / max(1, self.warmup_batches), 1.0)
            lr = self._warmup_start_lr + (self.learning_rate - self._warmup_start_lr) * frac
            self._set_optimizer_lrs(lr)
            return
        if self.warmup_batches > 0 and self.global_step == self.warmup_batches + 1:
            self._set_optimizer_lrs(self.learning_rate)
        if self._scheduler_kind == "cosine":
            self._post_warmup_step += 1
            t = min(self._post_warmup_step, self._cosine_total_steps)
            frac = 0.5 * (1.0 + math.cos(math.pi * t / self._cosine_total_steps))
            lr = self.min_learning_rate + (self.learning_rate - self.min_learning_rate) * frac
            self._set_optimizer_lrs(lr)
        elif self._scheduler_kind == "step" and self.lr_decay_step > 0:
            if self.global_step % self.lr_decay_step == 0:
                for group in self.optimizer.param_groups:
                    group["lr"] = float(group["lr"]) * float(self.lr_decay_factor)
                self._clamp_optimizer_lrs()

    def _step_scheduler_after_epoch(self, metric):
        if self._stage_two_active:
            if self._swa_scheduler is not None:
                self._swa_scheduler.step()
                self._clamp_optimizer_lrs()
            return
        if self._scheduler_kind == "plateau" and self._scheduler is not None:
            self._scheduler.step(metric)
            self._clamp_optimizer_lrs()
        elif self._scheduler_kind == "exp":
            for group in self.optimizer.param_groups:
                group["lr"] = float(group["lr"]) * self.lr_scheduler_gamma
            self._clamp_optimizer_lrs()

    def _maybe_activate_stage_two(self, epoch: int):
        if self._stage_two_active or not self.stage_two_enabled:
            return
        by_epoch = self.swa_start_epoch >= 0 and int(epoch) >= self.swa_start_epoch
        by_step = self.swa_start_step >= 0 and self.global_step >= self.swa_start_step
        if not (by_epoch or by_step):
            return
        self._stage_two_active = True
        self._stage_two_activated_epoch = int(epoch)
        self._stage_two_activated_step = int(self.global_step)
        self.a = self.stage_two_energy_weight
        self.b = self.stage_two_force_weight
        self.c = self.stage_two_stress_weight
        self._set_optimizer_lrs(self.swa_lr)
        self._swa_scheduler = SWALR(
            optimizer=self.optimizer,
            swa_lr=self.swa_lr,
            anneal_epochs=self.swa_anneal_epochs,
            anneal_strategy=self.swa_anneal_strategy,
        )
        if self.main_process:
            log.info(
                "Stage Two/SWA activated at epoch=%d step=%d: weights E=%g F=%g S=%g lr=%g",
                epoch,
                self.global_step,
                self.a,
                self.b,
                self.c,
                self.swa_lr,
            )

    # ----------------------------------------------------------------- make_fx
    def _complete_mbd_training_ramp_for_makefx(self) -> None:
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        for module in raw_model.modules():
            complete = getattr(module, "complete_training_ramp", None)
            if callable(complete):
                complete()

    def _makefx_forward(
        self,
        pos,
        A,
        batch_idx,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        strain=None,
        dispersion_edge_src=None,
        dispersion_edge_dst=None,
        dispersion_edge_shifts=None,
        precomputed_triplet_in=None,
        precomputed_triplet_out=None,
    ):
        """make_fx-compiled ``(E_per_atom, dE/dpos)`` -- and, when ``strain`` is given,
        also ``dE/dstrain`` for the stress/virial loss (positions and cell deformed by
        ``I + sym(strain)``, traced at ``strain=0`` so the forward is undeformed).

        Compiled once per (training, input-shape) signature and cached outside the
        nn.Module tree. With a BucketBatchSampler the distinct shapes == the bucket
        count, so the compile count is bounded and each batch hits the cache."""
        from mace_ictc.training.makefx_compile import CompiledForceCache

        if self.model.training:
            self._complete_mbd_training_ramp_for_makefx()

        if getattr(self, "_makefx_cache", None) is None:
            object.__setattr__(self, "_makefx_cache",
                               CompiledForceCache(self.model, max_slots=self._makefx_max_slots))

        stress = strain is not None
        dispersion_edge_args = (dispersion_edge_src, dispersion_edge_dst, dispersion_edge_shifts)
        has_dispersion_edges = all(arg is not None for arg in dispersion_edge_args)
        if any(arg is not None for arg in dispersion_edge_args) and not has_dispersion_edges:
            raise ValueError(
                "dispersion_edge_src, dispersion_edge_dst, and dispersion_edge_shifts must be provided together"
            )
        triplet_args = (precomputed_triplet_in, precomputed_triplet_out)
        has_precomputed_triplets = all(arg is not None for arg in triplet_args)
        if any(arg is not None for arg in triplet_args) and not has_precomputed_triplets:
            raise ValueError(
                "precomputed_triplet_in and precomputed_triplet_out must be provided together"
            )
        if has_precomputed_triplets and (stress or has_dispersion_edges):
            raise NotImplementedError(
                "compiled precomputed triplets currently support energy/force training without stress or dispersion"
            )

        def _factory(model, *, training):
            if has_precomputed_triplets:
                def compute_fn(
                    pos,
                    A,
                    batch_idx,
                    edge_src,
                    edge_dst,
                    edge_shifts,
                    cell,
                    triplet_in,
                    triplet_out,
                ):
                    p = pos.detach().requires_grad_(True)
                    e_atom = model(
                        p,
                        A,
                        batch_idx,
                        edge_src,
                        edge_dst,
                        edge_shifts,
                        cell,
                        precomputed_triplet_in=triplet_in,
                        precomputed_triplet_out=triplet_out,
                    )
                    if isinstance(e_atom, tuple):
                        e_atom = e_atom[0]
                    grad = torch.autograd.grad(
                        e_atom.sum(), p, create_graph=training
                    )[0]
                    return e_atom, grad
                return compute_fn

            if stress and has_dispersion_edges:
                def compute_fn(
                    pos,
                    strain,
                    A,
                    batch_idx,
                    edge_src,
                    edge_dst,
                    edge_shifts,
                    cell,
                    dispersion_edge_src,
                    dispersion_edge_dst,
                    dispersion_edge_shifts,
                ):
                    p = pos.detach().requires_grad_(True)
                    s = strain.detach().requires_grad_(True)
                    sym = 0.5 * (s + s.transpose(-1, -2))
                    defo = torch.eye(3, dtype=p.dtype, device=p.device) + sym  # [B,3,3]
                    pos_in = torch.einsum("ni,nij->nj", p, defo[batch_idx])
                    cell_in = torch.bmm(cell, defo)
                    e_atom = model(
                        pos_in,
                        A,
                        batch_idx,
                        edge_src,
                        edge_dst,
                        edge_shifts,
                        cell_in,
                        dispersion_edge_src=dispersion_edge_src,
                        dispersion_edge_dst=dispersion_edge_dst,
                        dispersion_edge_shifts=dispersion_edge_shifts,
                    )
                    if isinstance(e_atom, tuple):
                        e_atom = e_atom[0]
                    g = torch.autograd.grad(e_atom.sum(), [p, s], create_graph=training)
                    return e_atom, g[0], g[1]
                return compute_fn

            if stress:
                def compute_fn(pos, strain, A, batch_idx, edge_src, edge_dst, edge_shifts, cell):
                    p = pos.detach().requires_grad_(True)
                    s = strain.detach().requires_grad_(True)
                    sym = 0.5 * (s + s.transpose(-1, -2))
                    defo = torch.eye(3, dtype=p.dtype, device=p.device) + sym  # [B,3,3]
                    pos_in = torch.einsum("ni,nij->nj", p, defo[batch_idx])
                    cell_in = torch.bmm(cell, defo)
                    e_atom = model(pos_in, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_in)
                    if isinstance(e_atom, tuple):
                        e_atom = e_atom[0]
                    g = torch.autograd.grad(e_atom.sum(), [p, s], create_graph=training)
                    return e_atom, g[0], g[1]
                return compute_fn

            if has_dispersion_edges:
                def compute_fn(
                    pos,
                    A,
                    batch_idx,
                    edge_src,
                    edge_dst,
                    edge_shifts,
                    cell,
                    dispersion_edge_src,
                    dispersion_edge_dst,
                    dispersion_edge_shifts,
                ):
                    p = pos.detach().requires_grad_(True)
                    e_atom = model(
                        p,
                        A,
                        batch_idx,
                        edge_src,
                        edge_dst,
                        edge_shifts,
                        cell,
                        dispersion_edge_src=dispersion_edge_src,
                        dispersion_edge_dst=dispersion_edge_dst,
                        dispersion_edge_shifts=dispersion_edge_shifts,
                    )
                    if isinstance(e_atom, tuple):
                        e_atom = e_atom[0]
                    grad = torch.autograd.grad(e_atom.sum(), p, create_graph=training)[0]
                    return e_atom, grad
                return compute_fn

            def compute_fn(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell):
                p = pos.detach().requires_grad_(True)
                e_atom = model(p, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
                if isinstance(e_atom, tuple):
                    e_atom = e_atom[0]
                # E_per_atom.sum() == E_mean.sum() (the per-type offset is pos-independent),
                # so this gradient is the exact eager dE/dpos.
                grad = torch.autograd.grad(e_atom.sum(), p, create_graph=training)[0]
                return e_atom, grad
            return compute_fn

        if stress:
            example_inputs = (pos, strain, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
        else:
            example_inputs = (pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
        if has_dispersion_edges:
            example_inputs = example_inputs + (dispersion_edge_src, dispersion_edge_dst, dispersion_edge_shifts)
        if has_precomputed_triplets:
            example_inputs = example_inputs + triplet_args
        compiled = self._makefx_cache.get(
            example_inputs,
            training=bool(self.model.training),
            compute_fn_factory=_factory,
        )
        return compiled(*example_inputs)

    def _model_topology_kwargs(self, edge_src, edge_dst, num_nodes):
        """Prepare optional discrete topology before entering eager/autograd graphs."""
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        prepare = getattr(raw_model, "prepare_precomputed_topology", None)
        if not callable(prepare):
            return {}
        kwargs = prepare(edge_src, edge_dst, int(num_nodes))
        if not isinstance(kwargs, dict):
            raise TypeError("prepare_precomputed_topology must return a dict")
        return kwargs

    # --------------------------------------------------------------- per-batch
    def _unpack(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 11:
            (pos, A, batch_idx, force_ref, target_energies,
             edge_src, edge_dst, edge_shifts, cell, stress_ref, extras) = batch
        else:
            (pos, A, batch_idx, force_ref, target_energies,
             edge_src, edge_dst, edge_shifts, cell, stress_ref) = batch
            extras = {}
        fd, ld = self.dtype, torch.long
        dev = self.device
        return dict(
            pos=pos.to(dev, fd), A=A.to(dev, ld), batch_idx=batch_idx.to(dev, ld),
            force_ref=force_ref.to(dev, fd), target_energies=target_energies.to(dev, fd),
            edge_src=edge_src.to(dev, ld), edge_dst=edge_dst.to(dev, ld),
            edge_shifts=edge_shifts.to(dev, fd), cell=cell.to(dev, fd),
            stress_ref=stress_ref.to(dev, fd), extras=extras,
        )

    def _dispersion_edges_for_batch(self, pos, batch_idx, cell, extras):
        self._last_dispersion_neighbor_build_info = None
        model = self.model.module if hasattr(self.model, "module") else self.model
        if getattr(model, "dispersion", None) is not None:
            raise ValueError("Dispersion is not supported by this K100 distribution")
        return None

    def _uses_mesh_fft_long_range(self) -> bool:
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if getattr(raw_model, "long_range_module", None) is None:
            return False
        if str(getattr(raw_model, "long_range_reciprocal_backend", "")) == "mesh_fft":
            return True
        if str(getattr(raw_model, "long_range_runtime_backend", "")) == "mesh_fft":
            return True
        lr_module = getattr(raw_model, "long_range_module", None)
        return str(getattr(lr_module, "reciprocal_backend", "")) == "mesh_fft"

    def _model_dispersion_kwargs(self, dispersion_edges):
        if dispersion_edges is None:
            return {}
        return {
            "dispersion_edge_src": dispersion_edges[0],
            "dispersion_edge_dst": dispersion_edges[1],
            "dispersion_edge_shifts": dispersion_edges[2],
        }

    @torch.no_grad()
    def _record_dispersion_graph_observation(self, batch_idx, cell, dispersion_edges):
        if dispersion_edges is not None:
            raise ValueError("Dispersion graphs are not supported by this K100 distribution")

    def _dispersion_graph_observation_keys(self) -> tuple[str, ...]:
        return (
            "observed_batches",
            "observed_edges",
            "single_image_cell_batches",
            "non_single_image_cell_batches",
            "self_image_edges",
            "multi_image_shift_edges",
            "builder_explicit_batches",
            "builder_auto_bruteforce_batches",
            "builder_auto_torch_cluster_batches",
            "builder_auto_cell_batches",
            "builder_auto_bruteforce_fallback_batches",
            "builder_bruteforce_batches",
            "builder_cell_batches",
            "builder_unknown_batches",
        )

    def _zero_dispersion_graph_observation(self) -> dict:
        return {k: 0 for k in self._dispersion_graph_observation_keys()}

    def _restore_dispersion_graph_observation(self, value) -> None:
        if not isinstance(value, dict):
            return
        restored = self._zero_dispersion_graph_observation()
        for key in self._dispersion_graph_observation_keys():
            raw = value.get(key, 0)
            try:
                restored[key] = max(0, int(raw))
            except (TypeError, ValueError):
                restored[key] = 0
        self._dispersion_graph_observation_base = restored
        self._dispersion_graph_observation = self._zero_dispersion_graph_observation()
        self._dispersion_graph_observation_synced = None

    @torch.no_grad()
    def _sync_dispersion_graph_observation(self) -> None:
        if not self._dist_ready():
            return
        keys = self._dispersion_graph_observation_keys()
        values = [float(self._dispersion_graph_observation[k]) for k in keys]
        packed = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        reduced = {k: int(round(float(v))) for k, v in zip(keys, packed.detach().cpu().tolist())}
        self._dispersion_graph_observation_synced = reduced

    def _dispersion_graph_observation_summary(self) -> dict:
        source = self._dispersion_graph_observation_synced or self._dispersion_graph_observation
        obs = {
            k: int(self._dispersion_graph_observation_base[k]) + int(source[k])
            for k in self._dispersion_graph_observation_keys()
        }
        if obs["observed_batches"] <= 0:
            status = "not_observed"
        elif (
            obs["non_single_image_cell_batches"] > 0
            or obs["self_image_edges"] > 0
            or obs["multi_image_shift_edges"] > 0
        ):
            status = "observed_not_single_image_deployable"
        else:
            status = "observed_single_image_deployable"
        obs["status"] = status
        if obs["observed_batches"] <= 0:
            builder_status = "not_observed"
        elif obs["builder_auto_bruteforce_fallback_batches"] > 0:
            builder_status = "observed_large_dense_fallback"
        elif obs["builder_auto_bruteforce_batches"] > 0 or obs["builder_bruteforce_batches"] > 0:
            builder_status = "observed_dense_builder"
        elif (
            obs["builder_unknown_batches"] > 0
            or (
                obs["builder_explicit_batches"]
                + obs["builder_auto_torch_cluster_batches"]
                + obs["builder_auto_cell_batches"]
                + obs["builder_cell_batches"]
            )
            <= 0
        ):
            builder_status = "observed_unknown_builder"
        else:
            builder_status = "observed_sparse_or_explicit_builder"
        obs["neighbor_builder_status"] = builder_status
        obs["scope"] = "distributed" if self._dispersion_graph_observation_synced is not None else "local"
        return obs

    def _eager_strain_gradient(
        self,
        pos,
        A,
        batch_idx,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        dispersion_edges,
    ):
        num_mol = cell.shape[0]
        pos_leaf = pos.detach().requires_grad_(True)
        strain = torch.zeros((num_mol, 3, 3), dtype=pos.dtype, device=self.device, requires_grad=True)
        sym = 0.5 * (strain + strain.transpose(-1, -2))
        defo = torch.eye(3, dtype=pos.dtype, device=self.device) + sym
        pos_in = torch.einsum("ni,nij->nj", pos_leaf, defo[batch_idx])
        cell_in = torch.bmm(cell, defo)
        out = self.model(
            pos_in,
            A,
            batch_idx,
            edge_src,
            edge_dst,
            edge_shifts,
            cell_in,
            **self._model_dispersion_kwargs(dispersion_edges),
        )
        E_per_atom = out[0] if isinstance(out, tuple) else out
        return torch.autograd.grad(E_per_atom.sum(), [pos_leaf, strain], create_graph=True, retain_graph=True)[1]

    def _compute(self, batch, *, training):
        """Forward + force-autograd + energy/force loss. Mirrors the FSCETP baseline."""
        b = self._unpack(batch)
        pos, A, batch_idx = b["pos"], b["A"], b["batch_idx"]
        edge_src, edge_dst, edge_shifts, cell = b["edge_src"], b["edge_dst"], b["edge_shifts"], b["cell"]
        force_ref = b["force_ref"]
        target_energies = b["target_energies"].view(-1)
        stress_ref, extras = b["stress_ref"], b["extras"]
        dispersion_edges = self._dispersion_edges_for_batch(pos, batch_idx, cell, extras)
        topology_kwargs = self._model_topology_kwargs(
            edge_src, edge_dst, pos.shape[0]
        )
        if training:
            self._record_dispersion_graph_observation(batch_idx, cell, dispersion_edges)

        atom_mask = None
        if isinstance(extras, dict) and torch.is_tensor(extras.get("atom_mask", None)):
            atom_mask = extras["atom_mask"].to(device=self.device, dtype=pos.dtype).view(-1)
        stress_mask = None
        if isinstance(extras, dict) and torch.is_tensor(extras.get("stress_mask", None)):
            stress_mask = extras["stress_mask"].to(device=self.device, dtype=torch.bool).view(-1)

        def _mol_sum(x):
            if atom_mask is not None:
                x = x * atom_mask.view([-1] + [1] * (x.dim() - 1))
            return scatter(x, batch_idx, dim=0, reduce="sum")

        mapped_A = map_tensor_values(A, self.keys, self.values).to(self.device)
        E_offset_mol = _mol_sum(mapped_A)
        compute_stress = self.c > 0
        num_mol = cell.shape[0]

        use_makefx = (
            training
            and self.train_makefx_compile
            and not self._makefx_disabled
        )
        grad0 = grad_strain = None
        if use_makefx:
            try:
                if compute_stress:
                    if self._uses_mesh_fft_long_range():
                        # Inductor currently hits a complex conjugate-view dtype bug when the
                        # cuFFT long-range path is compiled inside the strain-gradient graph.
                        # Keep the dominant energy/force path make_fx-compiled, and compute only
                        # the virial strain derivative eagerly with the same fixed topology.
                        E_per_atom, grad0 = self._makefx_forward(
                            pos,
                            A,
                            batch_idx,
                            edge_src,
                            edge_dst,
                            edge_shifts,
                            cell,
                            dispersion_edge_src=None if dispersion_edges is None else dispersion_edges[0],
                            dispersion_edge_dst=None if dispersion_edges is None else dispersion_edges[1],
                            dispersion_edge_shifts=None if dispersion_edges is None else dispersion_edges[2],
                            **topology_kwargs,
                        )
                        grad_strain = self._eager_strain_gradient(
                            pos,
                            A,
                            batch_idx,
                            edge_src,
                            edge_dst,
                            edge_shifts,
                            cell,
                            dispersion_edges,
                        )
                    else:
                        strain0 = torch.zeros((num_mol, 3, 3), dtype=pos.dtype, device=self.device)
                        E_per_atom, grad0, grad_strain = self._makefx_forward(
                            pos,
                            A,
                            batch_idx,
                            edge_src,
                            edge_dst,
                            edge_shifts,
                            cell,
                            strain=strain0,
                            dispersion_edge_src=None if dispersion_edges is None else dispersion_edges[0],
                            dispersion_edge_dst=None if dispersion_edges is None else dispersion_edges[1],
                            dispersion_edge_shifts=None if dispersion_edges is None else dispersion_edges[2],
                            **topology_kwargs,
                        )
                else:
                    E_per_atom, grad0 = self._makefx_forward(
                        pos,
                        A,
                        batch_idx,
                        edge_src,
                        edge_dst,
                        edge_shifts,
                        cell,
                        dispersion_edge_src=None if dispersion_edges is None else dispersion_edges[0],
                        dispersion_edge_dst=None if dispersion_edges is None else dispersion_edges[1],
                        dispersion_edge_shifts=None if dispersion_edges is None else dispersion_edges[2],
                        **topology_kwargs,
                    )
            except Exception as e:
                object.__setattr__(self, "_makefx_disabled", True)
                if self.require_train_makefx_compile:
                    raise RuntimeError(
                        "train_makefx_compile is required for this run, but tracing/compilation failed"
                    ) from e
                log.warning("train_makefx_compile failed (%s: %s); falling back to eager.",
                            type(e).__name__, e)
                use_makefx = False

        if not use_makefx:
            pos_leaf = pos.detach().requires_grad_(True)
            if compute_stress:
                # virial via the strain derivative: deform positions + cell by I + sym(strain),
                # evaluate at strain=0, read stress = dE/dstrain / V (FSCETP convention).
                strain = torch.zeros((num_mol, 3, 3), dtype=pos.dtype, device=self.device,
                                     requires_grad=True)
                sym = 0.5 * (strain + strain.transpose(-1, -2))
                defo = torch.eye(3, dtype=pos.dtype, device=self.device) + sym
                pos_in = torch.einsum("ni,nij->nj", pos_leaf, defo[batch_idx])
                cell_in = torch.bmm(cell, defo)
                grad_targets = [pos_leaf, strain]
            else:
                pos_in, cell_in, grad_targets = pos_leaf, cell, [pos_leaf]
            # Epoch validation runs on rank zero. Bypass DDP forward hooks,
            # including bucket rebuilds that require every rank to participate.
            forward_model = self.model if training else self._raw_model
            out = forward_model(
                pos_in,
                A,
                batch_idx,
                edge_src,
                edge_dst,
                edge_shifts,
                cell_in,
                **self._model_dispersion_kwargs(dispersion_edges),
                **topology_kwargs,
            )
            E_per_atom = out[0] if isinstance(out, tuple) else out
            E_conv_mol = _mol_sum(E_per_atom).squeeze(-1)
            E_mean = E_conv_mol + E_offset_mol
            grads = torch.autograd.grad(
                E_mean.sum(), grad_targets, create_graph=training, retain_graph=training)
            grad0 = grads[0]
            grad_strain = grads[1] if compute_stress else None
        else:
            E_conv_mol = _mol_sum(E_per_atom).squeeze(-1)
            E_mean = E_conv_mol + E_offset_mol

        f_pred = -grad0  # restore_force is identity in this pipeline
        force_ref_scaled = force_ref * self.force_shift_value

        if atom_mask is not None:
            mb = atom_mask.bool()
            f_pred_l, force_ref_l = f_pred[mb], force_ref_scaled[mb]
            num_atoms_per_mol = scatter(atom_mask, batch_idx, dim=0, reduce="sum")
        else:
            f_pred_l, force_ref_l = f_pred, force_ref_scaled
            num_atoms_per_mol = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce="sum")

        force_loss, force_loss_sum, force_count = self._loss_term_stats(f_pred_l, force_ref_l)

        E_avg_pred = E_mean / num_atoms_per_mol
        target_energy_avg = target_energies / num_atoms_per_mol
        energy_loss, energy_loss_sum, energy_count = self._loss_term_stats(E_avg_pred, target_energy_avg)

        total_loss = self.a * energy_loss + self.b * force_loss

        # Stress / virial loss: σ = dE/dε / V, SmoothL1 vs the reference stress (extra
        # term, total = a·E + b·F + c·σ). makefx path also produces dE/dε; the eager path
        # reads it from the second grad target.
        if compute_stress:
            volume = torch.abs(torch.det(cell)).clamp(min=1e-10)
            stress_pred = grad_strain / volume.view(-1, 1, 1)  # [B,3,3]
            if stress_mask is not None:
                stress_pred_l, stress_ref_l = stress_pred[stress_mask], stress_ref[stress_mask]
            else:
                stress_pred_l, stress_ref_l = stress_pred, stress_ref
            if stress_pred_l.numel():
                stress_loss, stress_loss_sum, stress_count = self._loss_term_stats(
                    stress_pred_l, stress_ref_l
                )
            else:
                stress_loss = stress_pred.sum() * 0.0
                stress_loss_sum = stress_loss.detach()
                stress_count = stress_loss.new_zeros(())
            total_loss = total_loss + self.c * stress_loss
        else:
            stress_loss = torch.zeros((), device=self.device)
            stress_loss_sum = torch.zeros((), device=self.device)
            stress_count = torch.zeros((), device=self.device)

        with torch.no_grad():
            # Loss may intentionally use the legacy scaled force target, but reported
            # metrics must remain errors against the physical dataset labels.
            force_ref_metric = force_ref[mb] if atom_mask is not None else force_ref
            force_diff = (f_pred_l - force_ref_metric).reshape(-1)
            energy_diff = (E_avg_pred - target_energy_avg).reshape(-1)
            if compute_stress:
                stress_diff = (stress_pred_l - stress_ref_l).reshape(-1)
            else:
                stress_diff = torch.empty(0, dtype=pos.dtype, device=self.device)

            force_rmse = force_diff.square().mean().sqrt()
            energy_rmse_avg = energy_diff.square().mean().sqrt()
            force_mae = force_diff.abs().mean()
            energy_mae_avg = energy_diff.abs().mean()
            stress_rmse = (
                stress_diff.square().mean().sqrt()
                if stress_diff.numel() else torch.zeros((), device=self.device)
            )
            stress_mae = (
                stress_diff.abs().mean()
                if stress_diff.numel() else torch.zeros((), device=self.device)
            )
            metric_sums = {
                "energy_loss_sum": energy_loss_sum.detach(),
                "energy_count": energy_count.detach(),
                "force_loss_sum": force_loss_sum.detach(),
                "force_count": force_count.detach(),
                "stress_loss_sum": stress_loss_sum.detach(),
                "stress_count": stress_count.detach(),
                "energy_sq_sum": energy_diff.square().sum(),
                "energy_abs_sum": energy_diff.abs().sum(),
                "force_sq_sum": force_diff.square().sum(),
                "force_abs_sum": force_diff.abs().sum(),
                "stress_sq_sum": stress_diff.square().sum(),
                "stress_abs_sum": stress_diff.abs().sum(),
            }
            per_graph_diagnostics = None
            if self.eval_top_k_errors > 0:
                # Keep this path opt-in: per-structure diagnostics require a
                # few additional reductions but must not burden ordinary
                # training or validation.  force_ref_metric and f_pred_l have
                # already had padded atoms removed above.
                force_diff_by_atom = f_pred_l - force_ref_metric
                force_batch = batch_idx[mb] if atom_mask is not None else batch_idx
                force_sq_by_atom = force_diff_by_atom.square().sum(dim=-1)
                force_abs_by_atom = force_diff_by_atom.abs().sum(dim=-1)
                force_component_count = scatter(
                    torch.full_like(force_sq_by_atom, 3.0),
                    force_batch,
                    dim=0,
                    dim_size=num_mol,
                    reduce="sum",
                ).clamp_min(1.0)
                per_graph_diagnostics = {
                    "force_rmse": (
                        scatter(
                            force_sq_by_atom,
                            force_batch,
                            dim=0,
                            dim_size=num_mol,
                            reduce="sum",
                        )
                        / force_component_count
                    ).sqrt(),
                    "force_mae": scatter(
                        force_abs_by_atom,
                        force_batch,
                        dim=0,
                        dim_size=num_mol,
                        reduce="sum",
                    )
                    / force_component_count,
                    "energy_abs_per_atom": energy_diff.abs(),
                    "num_atoms": num_atoms_per_mol,
                }
        return {
            "total_loss": total_loss,
            "energy_loss": energy_loss.detach(),
            "force_loss": force_loss.detach(),
            "stress_loss": stress_loss.detach(),
            "force_rmse": force_rmse,
            "energy_rmse_avg": energy_rmse_avg,
            "force_mae": force_mae,
            "energy_mae_avg": energy_mae_avg,
            "stress_rmse": stress_rmse,
            "stress_mae": stress_mae,
            "_metric_sums": metric_sums,
            "_per_graph_diagnostics": per_graph_diagnostics,
        }

    def _loss_term(self, pred, target):
        return self._loss_term_stats(pred, target)[0]

    def _loss_term_stats(self, pred, target):
        if self.loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction="none")
            loss_sum = loss.sum()
            count = loss.new_tensor(float(loss.numel()))
            return loss_sum / count.clamp_min(1.0), loss_sum, count
        return smooth_l1_loss_stats(pred, target, beta=self.loss_beta)

    @torch.no_grad()
    def _update_ema_state(self):
        state = self._raw_model.state_dict()
        if self._ema_state is None:
            self._ema_state = {k: v.detach().clone() for k, v in state.items()}
            return
        decay = self.ema_decay
        one_minus = 1.0 - decay
        for key, val in state.items():
            cur = val.detach()
            old = self._ema_state.get(key)
            if old is None or old.shape != cur.shape:
                self._ema_state[key] = cur.clone()
                continue
            cur = cur.to(device=old.device, dtype=old.dtype)
            if torch.is_floating_point(old):
                old.mul_(decay).add_(cur, alpha=one_minus)
            else:
                old.copy_(cur)

    @torch.no_grad()
    def _update_swa_state(self):
        state = self._raw_model.state_dict()
        if self._swa_state is None:
            self._swa_state = {k: v.detach().clone() for k, v in state.items()}
            self._swa_n = 1
            return
        self._swa_n += 1
        n = float(self._swa_n)
        for key, val in state.items():
            cur = val.detach()
            old = self._swa_state.get(key)
            if old is None or old.shape != cur.shape:
                self._swa_state[key] = cur.clone()
                continue
            cur = cur.to(device=old.device, dtype=old.dtype)
            if torch.is_floating_point(old):
                old.add_(cur - old, alpha=1.0 / n)
            else:
                old.copy_(cur)

    def _update_averaged_states(self, epoch: int):
        if self.ema_decay > 0.0 and self.global_step >= self.ema_start_step:
            self._update_ema_state()
        if self._stage_two_active:
            self._update_swa_state()

    def _reached_max_steps(self) -> bool:
        return self.max_steps is not None and self.global_step >= self.max_steps

    # ------------------------------------------------------------------- loops
    def _backward_loss(self, loss):
        if self.backward_targets == "parameters":
            # Accumulate ordinary leaf .grad and DDP reducer hooks. Force-loss
            # mixed derivatives are still computed for every trainable parameter.
            # The optimizer does not need d(loss)/d(input positions).
            parameters = tuple(p for p in self.model.parameters() if p.requires_grad)
            loss.backward(inputs=parameters)
        else:
            loss.backward()

    def train_epoch(self, epoch):
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        self._maybe_activate_stage_two(epoch)
        n_batches = len(self.train_loader)
        metric_sums = self._new_metric_sums()
        t0 = time.time()
        seen = 0
        _val_pts = set()
        if self.evals_per_epoch > 1 and self.val_loader is not None:
            for _k in range(1, self.evals_per_epoch):
                _val_pts.add((n_batches * _k) // self.evals_per_epoch)
        self.optimizer.zero_grad(set_to_none=True)
        accumulation_count = 0
        for i, batch in enumerate(self.train_loader):
            if self._reached_max_steps():
                break
            self._maybe_activate_stage_two(epoch)
            out = self._compute(batch, training=True)
            loss = out["total_loss"] / self.gradient_accumulation_steps
            self._backward_loss(loss)
            accumulation_count += 1
            optimizer_step = accumulation_count == self.gradient_accumulation_steps
            if optimizer_step:
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                accumulation_count = 0
                self._step_scheduler_after_batch()
                self._update_averaged_states(epoch)
                self._save_recovery_checkpoint(epoch)
            self._accumulate_metric_sums(metric_sums, out)
            seen += 1
            step_validation = (
                optimizer_step
                and self.val_loader is not None
                and self.validation_interval_steps > 0
                and self.global_step % self.validation_interval_steps == 0
            )
            if step_validation:
                self._validate_and_checkpoint(epoch, label="step")
                if self._dist_ready():
                    dist.barrier()
                self.model.train()
            if self.main_process and self.log_interval and (i % self.log_interval == 0):
                lr = self.optimizer.param_groups[0]["lr"]
                s = f" S={float(out['stress_loss']):.4f}" if self.c > 0 else ""
                phase = "stage2" if self._stage_two_active else "stage1"
                log.info("epoch %d step %d batch %d/%d phase=%s loss=%.4f E=%.4f F=%.4f%s Frmse=%.4f lr=%.2e",
                         epoch, self.global_step, i, n_batches, phase, float(out["total_loss"]),
                         float(out["energy_loss"]), float(out["force_loss"]), s,
                         float(out["force_rmse"]), lr)
            if i in _val_pts:
                _va = self._val_pass()
                if self.main_process:
                    log.info("epoch %d step %d MID-VAL loss=%.6g Frmse=%.6g Ermse=%.6g Fmae=%.6g Emae=%.6g",
                             epoch, self.global_step, _va["total_loss"], _va["force_rmse"], _va["energy_rmse_avg"],
                             _va["force_mae"], _va["energy_mae_avg"])
                    _stress_metrics = (
                        f" Srmse={_va['stress_rmse']:.6g} Smae={_va['stress_mae']:.6g}"
                        if self.c > 0 else ""
                    )
                    print(f"[epoch {epoch} step {self.global_step}] MID val loss={_va['total_loss']:.6g} "
                          f"Frmse={_va['force_rmse']:.6g} Ermse={_va['energy_rmse_avg']:.6g} "
                          f"Fmae={_va['force_mae']:.6g} Emae={_va['energy_mae_avg']:.6g}"
                          f"{_stress_metrics}", flush=True)
                    self._append_loss_csv(epoch, self.global_step, "mid", _va)
                self._save_rolling_checkpoint(epoch)
                if self._dist_ready():
                    dist.barrier()
                self.model.train()
        avg = self._reduce_epoch_metrics(metric_sums)
        avg["time"] = time.time() - t0
        avg["steps"] = int(seen)
        return avg

    @torch.no_grad()
    def _val_pass(self):
        # force needs grad of energy wrt pos even at eval -> enable grad locally.
        self.model.eval()
        metric_sums = self._new_metric_sums()
        diagnostic_rows: list[dict[str, float | int | str]] = []
        sample_offset = 0
        for batch in self.val_loader:
            with torch.enable_grad():
                out = self._compute(batch, training=False)
            self._accumulate_metric_sums(metric_sums, out)
            per_graph = out.get("_per_graph_diagnostics")
            if per_graph is not None:
                count = int(per_graph["force_rmse"].numel())
                dataset = getattr(self.val_loader, "dataset", None)
                for graph_index in range(count):
                    source_index = sample_offset + graph_index
                    sample_key = (
                        dataset._sample_key(source_index)
                        if dataset is not None and hasattr(dataset, "_sample_key")
                        else f"sample_{source_index}"
                    )
                    diagnostic_rows.append(
                        {
                            "dataset_index": source_index,
                            "sample_key": str(sample_key),
                            "num_atoms": int(per_graph["num_atoms"][graph_index].item()),
                            "force_rmse": float(per_graph["force_rmse"][graph_index].item()),
                            "force_mae": float(per_graph["force_mae"][graph_index].item()),
                            "energy_abs_per_atom": float(
                                per_graph["energy_abs_per_atom"][graph_index].item()
                            ),
                        }
                    )
                sample_offset += count
        if self.eval_top_k_errors > 0:
            diagnostic_rows.sort(key=lambda row: float(row["force_rmse"]), reverse=True)
            self.last_val_top_errors = diagnostic_rows[: self.eval_top_k_errors]
        else:
            self.last_val_top_errors = []
        return self._finalize_metric_sums(metric_sums, require_data=True)

    def _append_loss_csv(self, epoch, step, kind, val, tr=None):
        """Append one validation's losses/errors to loss.csv (next to the checkpoint)."""
        if not self.metrics_csv_path or not self.main_process:
            return
        cols = ["epoch", "step", "kind", "train_loss", "train_force_rmse",
                "val_loss", "val_energy_loss", "val_force_loss",
                "val_stress_loss", "val_force_rmse", "val_energy_rmse",
                "val_force_mae", "val_energy_mae", "val_stress_rmse", "val_stress_mae"]
        def g(d, k):
            return f"{float(d[k]):.6f}" if (d is not None and k in d) else ""
        new_file = not os.path.exists(self.metrics_csv_path)
        values = {
            "epoch": str(epoch), "step": str(step), "kind": str(kind),
            "train_loss": g(tr, "total_loss"), "train_force_rmse": g(tr, "force_rmse"),
            "val_loss": g(val, "total_loss"), "val_energy_loss": g(val, "energy_loss"),
            "val_force_loss": g(val, "force_loss"), "val_stress_loss": g(val, "stress_loss"),
            "val_force_rmse": g(val, "force_rmse"),
            "val_energy_rmse": g(val, "energy_rmse_avg"),
            "val_force_mae": g(val, "force_mae"), "val_energy_mae": g(val, "energy_mae_avg"),
            "val_stress_rmse": g(val, "stress_rmse"), "val_stress_mae": g(val, "stress_mae"),
        }
        if not new_file:
            with open(self.metrics_csv_path, "r") as f:
                existing_header = f.readline().strip()
            if existing_header:
                # Keep resumed legacy CSV files rectangular. New files receive the
                # extended stress columns; old files retain their original schema.
                cols = existing_header.split(",")
        row = [values.get(col, "") for col in cols]
        with open(self.metrics_csv_path, "a") as f:
            if new_file:
                f.write(",".join(cols) + "\n")
            f.write(",".join(row) + "\n")

    def load_checkpoint(self, path, *, training_state: bool = False, strict: bool = True) -> int:
        ckpt = torch.load(path, map_location=self.device)
        state_source = "raw"
        state = ckpt.get("e3trans_state_dict", ckpt)
        if isinstance(ckpt, dict) and not training_state:
            requested = self.checkpoint_state_source
            if requested == "auto":
                requested = str(
                    ckpt.get("default_state_source", "raw") or "raw"
                ).lower()
            state_key = {
                "raw": "e3trans_state_dict",
                "ema": "e3trans_ema_state_dict",
                "swa": "e3trans_swa_state_dict",
            }.get(requested)
            if state_key is None:
                log.warning(
                    "checkpoint requested unknown state_source=%s; using raw",
                    requested,
                )
            elif isinstance(ckpt.get(state_key), dict):
                state = ckpt[state_key]
                state_source = requested
            elif requested != "raw":
                log.warning(
                    "checkpoint state_source=%s requested but %s is absent; "
                    "using raw",
                    requested,
                    state_key,
                )
        # Converted MACE checkpoints carry the first-layer element self-connection that the
        # ICTC first interaction structurally omits (compensated via the mace_first_layer_sc0
        # buffer, added back in the forward). A freshly-built model leaves that buffer as None,
        # so a plain load_state_dict drops the checkpoint's value as an "unexpected" key and the
        # forward silently skips the self-connection (large force error). Install it first so the
        # buffer slot exists and gets loaded -- mirrors the deploy/LAMMPS load path.
        sc0 = state.get("mace_first_layer_sc0")
        if sc0 is not None and hasattr(self._raw_model, "install_mace_first_layer_sc0"):
            self._raw_model.install_mace_first_layer_sc0(sc0)
        missing, unexpected = self._raw_model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            log.warning("checkpoint load_state_dict missing=%s unexpected=%s", missing, unexpected)
        start_epoch = 0
        if training_state:
            if "optimizer_state_dict" not in ckpt:
                raise KeyError(f"{path} does not contain optimizer_state_dict")
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for group in self.optimizer.param_groups:
                for key, value in list(group.items()):
                    if torch.is_tensor(value):
                        group[key] = value.to(self.device)
            self.global_step = int(ckpt.get("global_step", 0))
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            if isinstance(ckpt.get("e3trans_ema_state_dict"), dict):
                self._ema_state = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in ckpt["e3trans_ema_state_dict"].items()
                }
            if isinstance(ckpt.get("e3trans_swa_state_dict"), dict):
                self._swa_state = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in ckpt["e3trans_swa_state_dict"].items()
                }
                self._swa_n = int(ckpt.get("swa_n_averaged", self._swa_n))
            self._stage_two_active = bool(
                ckpt.get("training_hyperparameters", {}).get("stage_two_active", self._stage_two_active)
            )
            self._stage_two_activated_epoch = ckpt.get(
                "training_hyperparameters", {}
            ).get("stage_two_activated_epoch", self._stage_two_activated_epoch)
            self._stage_two_activated_step = ckpt.get(
                "training_hyperparameters", {}
            ).get("stage_two_activated_step", self._stage_two_activated_step)
            self._restore_dispersion_graph_observation(
                ckpt.get("training_hyperparameters", {}).get("dispersion_training_graph_observation", None)
            )
            log.warning(
                "resumed optimizer/global_step from %s at epoch=%d step=%d; "
                "scheduler history is not serialized, so stateful schedulers resume approximately",
                path, start_epoch, self.global_step,
            )
        else:
            log.info(
                "loaded model weights from %s (state_source=%s)",
                path,
                state_source,
            )
        return start_epoch

    def fit(self, epochs=None, start_epoch: int = 0):
        epochs = int(epochs) if epochs is not None else self.epochs
        for epoch in range(int(start_epoch), epochs):
            if self._reached_max_steps():
                break
            self._maybe_activate_stage_two(epoch)
            tr = self.train_epoch(epoch)
            sterm = f" S={tr['stress_loss']:.4f}" if self.c > 0 else ""
            phase = "stage2" if self._stage_two_active else "stage1"
            msg = (f"[epoch {epoch} step {self.global_step} {phase}] train loss={tr['total_loss']:.6g} "
                   f"E={tr['energy_loss']:.6g} F={tr['force_loss']:.6g}{sterm} "
                   f"Frmse={tr['force_rmse']:.6g} ({tr['time']:.1f}s)")
            if self.val_loader is not None and self.main_process:
                va = self._val_pass()
                stress_metrics = (
                    f" Srmse={va['stress_rmse']:.6g} Smae={va['stress_mae']:.6g}"
                    if self.c > 0 else ""
                )
                msg += (f" | val loss={va['total_loss']:.6g} "
                        f"Frmse={va['force_rmse']:.6g} Ermse={va['energy_rmse_avg']:.6g} "
                        f"Fmae={va['force_mae']:.6g} Emae={va['energy_mae_avg']:.6g}"
                        f"{stress_metrics}")
                self._append_loss_csv(epoch, self.global_step, "epoch", va, tr)
                cur = va["total_loss"]
            else:
                cur = tr["total_loss"]
            cur = self._broadcast_float(cur)
            if self.main_process:
                log.info(msg)
                print(msg, flush=True)
            self._step_scheduler_after_epoch(cur)
            improved = cur < self.best_validation_loss
            if improved:
                self.best_validation_loss = cur
            should_save = self.checkpoint_path is not None and improved
            if should_save:
                self._sync_dispersion_graph_observation()
                if self.main_process:
                    self.save_checkpoint(self.checkpoint_path, epoch=epoch)
            self._save_rolling_checkpoint(epoch)
            if self._dist_ready():
                dist.barrier()
            if self._reached_max_steps():
                break
        return self.best_validation_loss

    def _validate_and_checkpoint(self, epoch: int, *, label: str) -> dict:
        """Validate one model state and make it eligible for global best selection."""
        va = self._val_pass()
        cur = self._broadcast_float(va["total_loss"])
        improved = cur < self.best_validation_loss
        if improved:
            self.best_validation_loss = cur
            if self.checkpoint_path is not None:
                self._sync_dispersion_graph_observation()
        if self.main_process:
            log.info(
                "epoch %d step %d %s-VAL loss=%.6g Frmse=%.6g Ermse=%.6g Fmae=%.6g Emae=%.6g",
                epoch, self.global_step, label.upper(), cur, va["force_rmse"],
                va["energy_rmse_avg"], va["force_mae"], va["energy_mae_avg"],
            )
            self._append_loss_csv(epoch, self.global_step, label, va)
            if improved:
                if self.checkpoint_path is not None:
                    self.save_checkpoint(self.checkpoint_path, epoch=epoch)
        self._save_rolling_checkpoint(epoch)
        self.best_validation_loss = self._broadcast_float(self.best_validation_loss)
        return va

    # -------------------------------------------------------------- checkpoint
    def _collect_arch_metadata(self):
        """Build the ``model_hyperparameters`` dict that the deploy-side
        ``LAMMPS_MLIAP_MFF.from_checkpoint`` reads to rebuild the architecture.

        Mirrors the FSCETP trainer's ``_collect_e3_arch_metadata`` for the keys that
        matter to the ictd-fix baseline: the ModelConfig snapshot plus the handful of
        live model attributes (num_interaction / avg_num_neighbors / ictd_fix_* etc.)
        whose defaults would otherwise be guessed wrong on reload."""
        base = self._raw_model
        cfg = self.config
        meta = {}
        if cfg is not None:
            hp = {
                "dtype": str(getattr(cfg, "dtype", torch.float64)).replace("torch.", ""),
                "channel_in": int(getattr(cfg, "channel_in", 64)),
                "channel_in2": int(getattr(cfg, "channel_in2", 32)),
                "channel_in3": int(getattr(cfg, "channel_in3", 32)),
                "channel_in4": int(getattr(cfg, "channel_in4", 32)),
                "channel_in5": int(getattr(cfg, "channel_in5", 32)),
                "max_atomvalue": int(getattr(cfg, "max_atomvalue", 10)),
                "embedding_dim": int(getattr(cfg, "embedding_dim", 16)),
                "embed_size": list(getattr(cfg, "embed_size", [128, 128, 128])),
                "output_size": int(getattr(cfg, "output_size", 8)),
                "lmax": int(getattr(cfg, "lmax", 2)),
                "irreps_output_conv_channels": getattr(cfg, "irreps_output_conv_channels", None),
                "function_type": str(getattr(cfg, "function_type", "gaussian")),
                "max_radius": float(getattr(cfg, "max_radius", 5.0)),
                "max_radius_main": float(getattr(cfg, "max_radius_main", getattr(cfg, "max_radius", 5.0))),
                "number_of_basis": int(getattr(cfg, "number_of_basis", 8)),
                "number_of_basis_main": int(getattr(cfg, "number_of_basis_main", 8)),
                "num_layers": int(getattr(cfg, "num_layers", 1)),
                "main_hidden_sizes3": list(getattr(cfg, "main_hidden_sizes3", [64, 32])),
                "emb_number_main_2": list(getattr(cfg, "emb_number_main_2", [64, 64, 64])),
            }
            for attr in (
                "num_interaction", "invariant_channels",
                "ictd_tp_path_policy", "ictd_tp_max_rank_other", "ictd_save_tp_mode",
                "save_contraction_order", "save_multiple_mix_channels",
                "ictd_fix_route", "ictd_fix_product_backend",
                "ictd_fix_use_reduced_cg", "ictd_fix_first_layer_self_connection",
                "ictd_fix_self_connection_rank",
                "ictd_fix_conv_tp_scale_init", "ictd_fix_freeze_conv_tp_weight",
                "ictd_tp_backend",
                "ictd_fix_interaction_init",
                "ictd_fix_edge_lmax",
                "ictd_fix_readout_hidden_channels",
                "ictd_fix_fusion_heads", "ictd_fix_fusion_head_weight_mode",
                "ictd_fix_interaction_attn_heads", "ictd_fix_interaction_scale",
                "ictd_fix_fusion_scale_init", "ictd_fix_gmix_gate_init",
                "ictd_fix_gmix_output_lmax", "avg_num_neighbors",
                "radial_sqrt_num_basis",
                "polynomial_cutoff_p", "long_range_mode", "angular_basis",
                "long_range_hidden_dim", "long_range_boundary", "long_range_neutralize",
                "long_range_filter_hidden_dim", "long_range_kmax", "long_range_mesh_size",
                "long_range_slab_padding_factor", "long_range_include_k0",
                "long_range_source_channels", "long_range_backend",
                "long_range_reciprocal_backend", "long_range_energy_partition",
                "long_range_green_mode", "long_range_assignment",
                "long_range_mesh_fft_full_ewald", "long_range_max_multipole_l",
                "long_range_mesh_fft_reciprocal_only",
                "long_range_smearing_sigma",
                "long_range_multipole_gate_init",
                "long_range_dispersion_mode", "long_range_dispersion", "dispersion_cutoff",
                "dispersion_min_cutoff", "dispersion_switch_width",
                "dispersion_max_num_neighbors", "dispersion_neighbor_method",
                "dispersion_bruteforce_threshold", "dispersion_allow_large_bruteforce_fallback",
                "dispersion_training_graph_rule", "dispersion_deployment_graph_rule",
                "dispersion_train_deploy_graph_compatibility",
                "dispersion_slq_num_probes", "dispersion_slq_lanczos_steps",
                "mbd_operator_backend", "mbd_pme_mesh_size", "mbd_pme_assignment",
                "mbd_pme_k_norm_floor", "mbd_pme_assignment_window_floor",
                "mbd_pme_ewald_alpha_prefactor",
            ):
                if hasattr(base, attr):
                    val = getattr(base, attr)
                    if val is not None:
                        hp[attr] = val
            hp["angular_basis_folded_in_state_dict"] = bool(
                getattr(base, "angular_basis", "ictd") == "e3nn"
                and getattr(base, "_e3nn_folded", False)
            )
            hp["energy_output_scale_enabled"] = bool(
                getattr(base, "energy_output_scale_enabled", False)
            )
            scale_buf = getattr(base, "energy_output_scale", None)
            hp["energy_output_scale"] = (
                float(scale_buf.detach().cpu().item()) if torch.is_tensor(scale_buf) else 1.0
            )
            hp["energy_output_shift_enabled"] = bool(
                getattr(base, "energy_output_shift_enabled", False)
            )
            shift_buf = getattr(base, "energy_output_shift", None)
            hp["energy_output_shift"] = (
                float(shift_buf.detach().cpu().item()) if torch.is_tensor(shift_buf) else 0.0
            )
            # Explicit construction choices win (these are exactly what build_baseline_model
            # passed, so from_checkpoint rebuilds the same architecture).
            hp.update(self.extra_hparams)
            from mace_ictc.utils.legacy_dispersion import (
                dispersion_deployment_graph_rule,
                dispersion_train_deploy_graph_compatibility,
                dispersion_training_graph_rule,
            )

            hp["dispersion_training_graph_rule"] = dispersion_training_graph_rule(
                str(hp.get("long_range_dispersion_mode", "none")),
                mbd_operator_backend=str(hp.get("mbd_operator_backend", "edge_sparse")),
            )
            hp["dispersion_deployment_graph_rule"] = dispersion_deployment_graph_rule(
                str(hp.get("long_range_dispersion_mode", "none")),
                mbd_operator_backend=str(hp.get("mbd_operator_backend", "edge_sparse")),
            )
            hp["dispersion_train_deploy_graph_compatibility"] = dispersion_train_deploy_graph_compatibility(
                str(hp.get("long_range_dispersion_mode", "none")),
                mbd_operator_backend=str(hp.get("mbd_operator_backend", "edge_sparse")),
            )
            meta["model_hyperparameters"] = hp
        return meta

    def _training_metadata(self):
        return {
            "loss": self.loss_type,
            "loss_beta": self.loss_beta,
            "energy_weight": self.a,
            "force_weight": self.b,
            "stress_weight": self.c,
            "force_shift_value": self.force_shift_value,
            "optimizer": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps,
            "amsgrad": self.amsgrad,
            "lr_scheduler": self.lr_scheduler_kind,
            "warmup_batches": self.warmup_batches,
            "warmup_start_ratio": self.warmup_start_ratio,
            "warmup_start_lr": self._warmup_start_lr,
            "lr_factor": self.lr_factor,
            "scheduler_patience": self.scheduler_patience,
            "lr_scheduler_gamma": self.lr_scheduler_gamma,
            "lr_decay_step": self.lr_decay_step,
            "lr_decay_factor": self.lr_decay_factor,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "max_grad_norm": self.max_grad_norm,
            "ema_decay": self.ema_decay,
            "ema_start_step": self.ema_start_step,
            "stage_two_enabled": self.stage_two_enabled,
            "stage_two_active": self._stage_two_active,
            "stage_two_activated_epoch": self._stage_two_activated_epoch,
            "stage_two_activated_step": self._stage_two_activated_step,
            "swa_start_epoch": self.swa_start_epoch,
            "swa_start_step": self.swa_start_step,
            "swa_lr": self.swa_lr,
            "swa_energy_weight": self.stage_two_energy_weight,
            "swa_force_weight": self.stage_two_force_weight,
            "swa_stress_weight": self.stage_two_stress_weight,
            "swa_anneal_epochs": self.swa_anneal_epochs,
            "swa_anneal_strategy": self.swa_anneal_strategy,
            "swa_n_averaged": self._swa_n,
            "checkpoint_state_source": self.checkpoint_state_source,
            "train_makefx_compile": self.train_makefx_compile,
            "operator_plan": self.operator_plan_info,
            "backward_targets": self.backward_targets,
            "makefx_max_slots": self._makefx_max_slots,
            "dispersion_training_graph_observation": self._dispersion_graph_observation_summary(),
        }

    def _default_state_source(self):
        have_ema = isinstance(self._ema_state, dict)
        have_swa = isinstance(self._swa_state, dict)
        requested = self.checkpoint_state_source
        if requested == "auto":
            if have_ema:
                return "ema"
            if have_swa:
                return "swa"
            return "raw"
        if requested == "ema" and not have_ema:
            log.warning("checkpoint_state_source=ema requested but no EMA state exists; using raw")
            return "raw"
        if requested == "swa" and not have_swa:
            log.warning("checkpoint_state_source=swa requested but no SWA state exists; using raw")
            return "raw"
        return requested

    def save_checkpoint(self, path, *, epoch=0):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        default_state_source = self._default_state_source()
        ckpt = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "e3trans_state_dict": self._raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "dtype": str(self.dtype).replace("torch.", ""),
            "max_radius": float(self.max_radius),
            "tensor_product_mode": self.tensor_product_mode,
            "atomic_energy_keys": self.keys.detach().cpu(),
            "atomic_energy_values": self.values.detach().cpu(),
            "a": self.a, "b": self.b, "c": self.c,
            "training_hyperparameters": self._training_metadata(),
            "default_state_source": default_state_source,
        }
        if isinstance(self._ema_state, dict):
            ckpt["e3trans_ema_state_dict"] = self._ema_state
        if isinstance(self._swa_state, dict):
            ckpt["e3trans_swa_state_dict"] = self._swa_state
            ckpt["swa_n_averaged"] = int(self._swa_n)
        ckpt.update(self._collect_arch_metadata())
        torch.save(ckpt, path)

    def _save_recovery_checkpoint(self, epoch: int) -> None:
        """Atomically replace the periodic resume checkpoint after an optimizer step."""
        if (
            self.checkpoint_interval_steps <= 0
            or self.checkpoint_path is None
            or not self.main_process
            or self.global_step % self.checkpoint_interval_steps != 0
        ):
            return
        stem, ext = os.path.splitext(self.checkpoint_path)
        path = f"{stem}.recovery{ext or '.pth'}"
        tmp = f"{path}.tmp"
        self.save_checkpoint(tmp, epoch=epoch)
        os.replace(tmp, path)

    def _save_rolling_checkpoint(self, epoch: int) -> None:
        """Keep the N most-recent VALIDATION checkpoints (``--keep-checkpoints N``), saved at
        EVERY validation (mid-epoch + epoch-end), rolling: delete the oldest beyond N. This is
        independent of the best-on-improvement save to ``checkpoint_path`` (which remains the
        canonical resume/deploy target). Each rolling file is ``<stem>.e<E>s<STEP><ext>``.
        No-op when ``keep_checkpoints <= 0`` (default), so existing runs are unaffected."""
        if self.keep_checkpoints <= 0 or self.checkpoint_path is None:
            return
        # The observation reduction is collective; writing is rank-zero only.
        self._sync_dispersion_graph_observation()
        if not self.main_process:
            return
        stem, ext = os.path.splitext(self.checkpoint_path)
        path = f"{stem}.e{epoch}s{self.global_step}{ext or '.pth'}"
        self.save_checkpoint(path, epoch=epoch)
        self._rolling_ckpts.append(path)
        while len(self._rolling_ckpts) > self.keep_checkpoints:
            old = self._rolling_ckpts.pop(0)
            if old != self.checkpoint_path and os.path.exists(old):
                try:
                    os.remove(old)
                except OSError:
                    pass
