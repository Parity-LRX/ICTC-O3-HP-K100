"""LAMMPS ML-IAP unified / ML-IAP-Kokkos interface for the molecular force field.

Provides LAMMPS_MLIAP_MFF, a subclass of MLIAPUnified that computes per-atom forces
via autograd on a dummy position tensor (dE/d(pos)), avoiding the O(N*M) edge-force
gradient storage of the traditional per-pair approach.

Per-atom forces are written directly into the LAMMPS force buffer (data.f), and
global virial is handled automatically by LAMMPS's virial_fdotr_compute().

支持两种运行模式：
- 标准 ML-IAP unified（CPU）：activate_mliappy + 直接写入 data.f
- ML-IAP-Kokkos（GPU）：activate_mliappy_kokkos + GPU tensor 直接写入

仅以下五种模型支持：e3nn_layers、e3nn_layers_channelwise、cue_layers_channelwise、
pure_cartesian_ictd_layers、pure_cartesian_ictd_layers_full（因其支持 precomputed_edge_vec）。

Usage:
    # Export:
    python -m mace_ictc.cli.export_mliap checkpoint.pth --elements H O

    # LAMMPS input:
    pair_style mliap unified model-mliap.pt 0
    pair_coeff * * H O
"""

from __future__ import annotations

import io
import inspect
import os
from typing import List, Optional, Tuple

import numpy as np
from ictc_o3.electrostatics.mesh import normalize_mesh_size
import torch
import torch.nn as nn
import torch.utils.dlpack as torch_dlpack

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except (ImportError, OSError):
    class MLIAPUnified:
        """Stub when lammps is not installed or shared lib is missing."""
        def __init__(self, interface=None, element_types=None,
                     ndescriptors=None, nparams=None, rcutfac=None):
            self.interface = interface
            self.element_types = element_types
            self.ndescriptors = ndescriptors
            self.nparams = nparams
            self.rcutfac = rcutfac

        def pickle(self, fname):
            import pickle
            with open(fname, "wb") as fp:
                pickle.dump(self, fp)

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
from ictc_o3.electrostatics.cutoff import derive_long_range_far_max_radius_multiplier
from mace_ictc.utils.checkpoint_metadata import get_checkpoint_e3_state_dict
from mace_ictc.utils.checkpoint_metadata import infer_external_tensor_rank_from_state_dict
from mace_ictc.utils.checkpoint_metadata import infer_ictd_save_final_readout_mode_from_state_dict
from mace_ictc.utils.checkpoint_metadata import infer_ictd_save_multiple_fusion_scheme_from_state_dict
from mace_ictc.utils.checkpoint_metadata import infer_ictd_save_multiple_order_from_state_dict
from mace_ictc.utils.checkpoint_metadata import infer_physical_tensor_outputs_from_state_dict
from mace_ictc.utils.checkpoint_metadata import validate_dispersion_deployment_graph_rule
from mace_ictc.utils.checkpoint_metadata import validate_dispersion_train_deploy_graph_compatibility
from mace_ictc.utils.checkpoint_metadata import validate_dispersion_training_graph_rule
from mace_ictc.utils.external_tensor_specs import normalize_external_tensor_specs
from mace_ictc.models.zbl import maybe_wrap_model_with_zbl
from mace_ictc.utils.tensor_utils import map_tensor_values


from mace_ictc.checkpoints.adapters import (
    _GLOBAL_PHYS_LAYOUT,
    _ATOM_PHYS_LAYOUT,
    _layout_total_dim,
    _layout_offsets,
    _GLOBAL_PHYS_TOTAL_DIM,
    _ATOM_PHYS_TOTAL_DIM,
    _GLOBAL_PHYS_OFFSETS,
    _ATOM_PHYS_OFFSETS,
    _PHYS_MASK_WIDTH,
    _resolve_model_external_tensor_rank,
    _resolve_model_external_tensor_specs,
    _resolve_model_external_tensor_total_numel,
    _canonicalize_cartesian_rank2,
    _physical_block_is_cartesian,
    _recover_rank1_cartesian,
    _recover_rank2_cartesian,
    _recover_cartesian_physical_tensors,
    _warm_ictd_projector_caches_for_trace,
    _TorchScriptEdgeVecCore,
    _TorchScriptEdgeVecCoreWithFidelity,
    _TorchScriptEdgeVecAdapter,
    _TorchScriptEdgeVecAdapterWithFidelity,
    _maybe_torchscript_trace_model,
    AtomForcesWrapper,
    EdgeForcesWrapper,
)


class LAMMPS_MP(torch.autograd.Function):
    """Autograd-compatible wrapper for LAMMPS Kokkos ghost communication.

    forward_exchange: copies local features → ghost atoms (across GPUs / PBC).
    reverse_exchange: accumulates ghost gradients back to local atoms.
    """

    @staticmethod
    def forward(ctx, feats: torch.Tensor, data) -> torch.Tensor:
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None



class LAMMPS_MLIAP_MFF(MLIAPUnified):
    """ML-IAP unified interface for the molecular force field.

    Computes per-atom forces via autograd on a dummy position tensor and writes
    them directly into the LAMMPS force buffer.  Global virial is handled by
    LAMMPS's ``virial_fdotr_compute()`` automatically.

    Implements the three required methods of MLIAPUnified:
    - compute_forces(data)
    - compute_descriptors(data)
    - compute_gradients(data)

    Attributes set for LAMMPS:
    - element_types: list of element symbols (e.g. ["H", "O"])
    - rcutfac: cutoff radius
    - ndescriptors / nparams: set to 1 (not used directly)
    """

    def __init__(
        self,
        model: nn.Module,
        element_types: List[str],
        max_radius: float,
        atomic_energy_keys: torch.Tensor,
        atomic_energy_values: torch.Tensor,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(
            interface=None,
            element_types=element_types,
            ndescriptors=1,
            nparams=1,
            rcutfac=max_radius,
        )
        self.device = device
        self.dtype = dtype
        self.wrapper = AtomForcesWrapper(model, atomic_energy_keys, atomic_energy_values)
        self.wrapper = self.wrapper.to(dtype=dtype).to(device)
        self.initialized = False

        # Buffer cache for compute_forces (reuse when ntotal/npairs unchanged)
        self._cache_ntotal: Optional[int] = None
        self._cache_npairs: Optional[int] = None
        self._cache_batch: Optional[torch.Tensor] = None
        self._cache_edge_shifts: Optional[torch.Tensor] = None
        self._cache_cell: Optional[torch.Tensor] = None
        # DLPack/cupy indices often come as int32; converting to int64 is a copy.
        # Cache converted indices by underlying (cupy) pointer to avoid per-step copies.
        self._cache_elems_ptr: Optional[int] = None
        self._cache_elems_i64: Optional[torch.Tensor] = None
        self._cache_pair_i_ptr: Optional[int] = None
        self._cache_pair_j_ptr: Optional[int] = None
        self._cache_pair_i_i64: Optional[torch.Tensor] = None
        self._cache_pair_j_i64: Optional[torch.Tensor] = None

        # Build elem index → atomic number Z lookup table.
        # LAMMPS data.elems is 0-based: index 0 = element_types[0], etc.
        from ase.data import atomic_numbers as ase_Z
        self._elem_to_Z = torch.tensor(
            [ase_Z.get(s, 0) for s in element_types], dtype=torch.long,
        )

    @classmethod
    def from_checkpoint(
        cls,
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
    ) -> "LAMMPS_MLIAP_MFF":
        """Compatibility entrypoint for legacy MLIAP callers and pickles."""
        from mace_ictc.checkpoints import load_checkpoint
        return load_checkpoint(
            checkpoint_path=checkpoint_path,
            element_types=element_types,
            max_radius=max_radius,
            atomic_energy_keys=atomic_energy_keys,
            atomic_energy_values=atomic_energy_values,
            device=device,
            embed_size=embed_size,
            output_size=output_size,
            tensor_product_mode=tensor_product_mode,
            num_interaction=num_interaction,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            avg_num_neighbors=avg_num_neighbors,
            torchscript=torchscript,
            force_naive=force_naive,
            ictd_tp_backend=ictd_tp_backend,
            operator_plan=operator_plan,
            generated_layout=generated_layout,
            _potential_class=cls,
        )

    def _init_device(self, data):
        """Detect device from data tensors (GPU if Kokkos, else CPU)."""
        try:
            using_kokkos = "kokkos" in data.__class__.__module__.lower()
        except Exception:
            using_kokkos = False

        self._using_kokkos = using_kokkos
        self._has_gpu_api = hasattr(data, "update_pair_forces_gpu")
        self._has_exchange = hasattr(data, "forward_exchange")

        if using_kokkos:
            device = torch.as_tensor(data.elems).device
        else:
            device = torch.device("cpu")
        self.device = device
        self.wrapper = self.wrapper.to(device)
        self._elem_to_Z = self._elem_to_Z.to(device)

        # Optional: torch.compile for 2-5x speedup (PyTorch 2.0+)
        if os.environ.get("MLIAP_USE_COMPILE", "").lower() in ("1", "true", "yes"):
            try:
                # Prewarm one-time caches (keeps Python-side setup out of Dynamo tracing)
                if os.environ.get("MLIAP_PREWARM", "1").lower() not in ("0", "false", "no"):
                    try:
                        with torch.no_grad():
                            for m in self.wrapper.model.modules():
                                prewarm = getattr(m, "prewarm_caches", None)
                                if callable(prewarm):
                                    prewarm(device=self.device, dtype=self.dtype)
                        print("[MLIAP] prewarmed model caches for compile", flush=True)
                    except Exception as e:
                        print(f"[MLIAP] cache prewarm skipped: {e}", flush=True)

                # In Kokkos+CUDA environments, CUDA Graph capture can cause warnings
                # and sometimes interfere with Kokkos finalize. Disable CUDA graphs by default.
                if using_kokkos and os.environ.get("MLIAP_DISABLE_CUDAGRAPHS", "").lower() not in ("0", "false", "no"):
                    try:
                        import torch._inductor.config as _icfg  # type: ignore
                        _icfg.triton.cudagraphs = False
                        print("[MLIAP] disabled inductor CUDA graphs (Kokkos safety)", flush=True)
                    except Exception:
                        pass

                mode = os.environ.get("MLIAP_COMPILE_MODE", "reduce-overhead")
                self.wrapper = torch.compile(self.wrapper, mode=mode, dynamic=True)
                print(f"[MLIAP] torch.compile enabled (mode={mode})", flush=True)
            except Exception as e:
                print(f"[MLIAP] torch.compile failed: {e}, using eager", flush=True)

        self.initialized = True

    def _to_torch(self, x, *, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert LAMMPS-provided arrays to torch.Tensor with minimal copies.

        Supports:
        - torch.Tensor (no copy unless dtype/device mismatch)
        - numpy.ndarray (CPU)
        - cupy.ndarray / objects implementing DLPack (GPU, zero-copy when dtype matches)
        """
        if torch.is_tensor(x):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.as_tensor(x)
        else:
            # Prefer DLPack for GPU arrays (e.g., cupy.ndarray)
            if hasattr(x, "__dlpack__"):
                t = torch_dlpack.from_dlpack(x)
            elif hasattr(x, "toDlpack"):
                t = torch_dlpack.from_dlpack(x.toDlpack())
            else:
                # Fallback: try torch.as_tensor (may copy)
                t = torch.as_tensor(x)

        if device is not None and t.device != device:
            t = t.to(device)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t

    def compute_forces(self, data):
        """Compute per-atom forces and write directly into the LAMMPS force buffer.

        Uses ``dE/d(pos)`` (per-atom, O(N)) instead of ``dE/d(edge_vec)``
        (per-pair, O(N*M)), reducing autograd leaf-gradient storage.

        Global virial is handled by LAMMPS C++ side via ``virial_fdotr_compute()``.

        Supports two modes:
        - Standard ML-IAP (CPU): writes to ``data.f`` numpy view
        - ML-IAP-Kokkos (GPU): writes to ``data.f`` GPU tensor
        """
        natoms = data.nlocal
        ntotal = data.ntotal
        npairs = data.npairs

        if not self.initialized:
            self._init_device(data)
            if os.environ.get("MLIAP_DEBUG", "").lower() in ("1", "true", "yes"):
                def _fmt_arr(x):
                    try:
                        if torch.is_tensor(x):
                            return (f"torch.Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, "
                                    f"device={x.device}, contiguous={x.is_contiguous()})")
                        if isinstance(x, np.ndarray):
                            return (f"np.ndarray(shape={x.shape}, dtype={x.dtype}, "
                                    f"c_contig={bool(x.flags['C_CONTIGUOUS'])})")
                        # cupy.ndarray (or other array-likes) common in ML-IAP-Kokkos
                        mod = type(x).__module__
                        name = type(x).__name__
                        if mod.startswith("cupy"):
                            try:
                                dev = getattr(getattr(x, "device", None), "id", None)
                            except Exception:
                                dev = None
                            return f"{mod}.{name}(shape={getattr(x,'shape',None)}, dtype={getattr(x,'dtype',None)}, device={dev})"
                        return f"{type(x)}"
                    except Exception as _e:  # pragma: no cover
                        return f"{type(x)} (fmt_error={_e})"

                model_cls = type(self.wrapper.model).__name__
                model_mod = type(self.wrapper.model).__module__
                nghost = ntotal - natoms
                inflate = ntotal / natoms if natoms > 0 else 0
                pairs_per_local = npairs / natoms if natoms > 0 else 0
                print(
                    f"[MLIAP] model={model_mod}.{model_cls}, kokkos={self._using_kokkos}, "
                    f"gpu_api={self._has_gpu_api}, device={self.device}",
                    flush=True,
                )
                print(
                    f"[MLIAP] nlocal={natoms}, ntotal={ntotal}, nghost={nghost}, npairs={npairs} | "
                    f"ntotal/nlocal={inflate:.2f}x, npairs/nlocal={pairs_per_local:.1f}",
                    flush=True,
                )
                print(f"[MLIAP] data.rij: {_fmt_arr(getattr(data, 'rij', None))}", flush=True)
                print(f"[MLIAP] data.elems: {_fmt_arr(getattr(data, 'elems', None))}", flush=True)
                print(f"[MLIAP] data.pair_i: {_fmt_arr(getattr(data, 'pair_i', None))}", flush=True)
                print(f"[MLIAP] data.pair_j: {_fmt_arr(getattr(data, 'pair_j', None))}", flush=True)
                print(f"[MLIAP] data.f: {_fmt_arr(getattr(data, 'f', None))}", flush=True)
                print(f"[MLIAP] data.eatoms: {_fmt_arr(getattr(data, 'eatoms', None))}", flush=True)

                # DLPack / zero-copy verification (Cupy -> Torch)
                def _ptr_cupy(x):
                    try:
                        # cupy.ndarray
                        return int(x.data.ptr)
                    except Exception:
                        return None

                def _verify_dlpack(name: str, x, *, dtype=None):
                    try:
                        t = self._to_torch(x, dtype=dtype, device=self.device)
                        cupy_ptr = _ptr_cupy(x) if type(x).__module__.startswith("cupy") else None
                        torch_ptr = int(t.data_ptr())
                        extra = ""
                        if cupy_ptr is not None:
                            extra = f", cupy_ptr=0x{cupy_ptr:x}, torch_ptr=0x{torch_ptr:x}, same_ptr={cupy_ptr == torch_ptr}"
                        print(
                            f"[MLIAP] to_torch({name}): shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}{extra}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[MLIAP] to_torch({name}) failed: {e}", flush=True)

                _verify_dlpack("rij", getattr(data, "rij", None), dtype=self.dtype)
                # For int32 indices (cupy): dtype=None shows true zero-copy;
                # dtype=torch.long shows the required cast (copy) for PyTorch indexing.
                _verify_dlpack("elems(raw)", getattr(data, "elems", None), dtype=None)
                _verify_dlpack("elems(i64)", getattr(data, "elems", None), dtype=torch.long)
                _verify_dlpack("pair_i(raw)", getattr(data, "pair_i", None), dtype=None)
                _verify_dlpack("pair_i(i64)", getattr(data, "pair_i", None), dtype=torch.long)
                _verify_dlpack("pair_j(raw)", getattr(data, "pair_j", None), dtype=None)
                _verify_dlpack("pair_j(i64)", getattr(data, "pair_j", None), dtype=torch.long)
                _verify_dlpack("f", getattr(data, "f", None), dtype=None)
                _verify_dlpack("eatoms", getattr(data, "eatoms", None), dtype=None)

        if natoms == 0 or npairs <= 1:
            return

        # --- Build tensors from LAMMPS data ---
        rij = self._to_torch(data.rij, dtype=self.dtype, device=self.device)

        # elems/pair_i/pair_j are often cupy int32; casting to int64 is required for indexing.
        # Cache casts by cupy pointer to avoid per-step copies when neighbor list is unchanged.
        try:
            elems_ptr = int(data.elems.data.ptr)  # cupy
        except Exception:
            elems_ptr = None
        if (
            elems_ptr is not None
            and self._cache_elems_ptr == elems_ptr
            and self._cache_elems_i64 is not None
            and int(self._cache_elems_i64.numel()) == int(ntotal)
        ):
            elem_idx = self._cache_elems_i64
        else:
            elem_raw = self._to_torch(data.elems, dtype=None, device=self.device)
            elem_idx = elem_raw.to(torch.long) if elem_raw.dtype != torch.long else elem_raw
            if elems_ptr is not None:
                self._cache_elems_ptr = elems_ptr
                self._cache_elems_i64 = elem_idx

        lut = self._elem_to_Z.to(device=self.device)
        species = lut[elem_idx]

        try:
            pair_i_ptr = int(data.pair_i.data.ptr)  # cupy
        except Exception:
            pair_i_ptr = None
        try:
            pair_j_ptr = int(data.pair_j.data.ptr)  # cupy
        except Exception:
            pair_j_ptr = None

        if (
            pair_i_ptr is not None
            and self._cache_pair_i_ptr == pair_i_ptr
            and self._cache_pair_i_i64 is not None
            and int(self._cache_pair_i_i64.numel()) == int(npairs)
        ):
            edge_src = self._cache_pair_i_i64
        else:
            pi_raw = self._to_torch(data.pair_i, dtype=None, device=self.device)
            edge_src = pi_raw.to(torch.long) if pi_raw.dtype != torch.long else pi_raw
            if pair_i_ptr is not None:
                self._cache_pair_i_ptr = pair_i_ptr
                self._cache_pair_i_i64 = edge_src

        if (
            pair_j_ptr is not None
            and self._cache_pair_j_ptr == pair_j_ptr
            and self._cache_pair_j_i64 is not None
            and int(self._cache_pair_j_i64.numel()) == int(npairs)
        ):
            edge_dst = self._cache_pair_j_i64
        else:
            pj_raw = self._to_torch(data.pair_j, dtype=None, device=self.device)
            edge_dst = pj_raw.to(torch.long) if pj_raw.dtype != torch.long else pj_raw
            if pair_j_ptr is not None:
                self._cache_pair_j_ptr = pair_j_ptr
                self._cache_pair_j_i64 = edge_dst

        # Robustness: Kokkos buffers can be reused; enforce consistent edge length.
        # All edge-index arrays must match rij length.
        E_rij = int(rij.shape[0])
        E_i = int(edge_src.numel())
        E_j = int(edge_dst.numel())
        Emin = min(E_rij, E_i, E_j)
        if Emin <= 0:
            return
        if Emin != E_rij or Emin != E_i or Emin != E_j:
            if os.environ.get("MLIAP_DEBUG", "").lower() in ("1", "true", "yes"):
                print(
                    f"[MLIAP] WARN: edge length mismatch (rij={E_rij}, pair_i={E_i}, pair_j={E_j}), "
                    f"using Emin={Emin}",
                    flush=True,
                )
            rij = rij[:Emin]
            edge_src = edge_src[:Emin]
            edge_dst = edge_dst[:Emin]
            # keep npairs consistent for downstream buffers
            npairs = Emin

        # Reuse buffers when sizes unchanged (typical in MD)
        if self._cache_ntotal == ntotal and self._cache_batch is not None:
            batch = self._cache_batch.zero_()
        else:
            batch = torch.zeros(ntotal, dtype=torch.long, device=self.device)
            self._cache_batch = batch
            self._cache_ntotal = ntotal
        if self._cache_npairs == npairs and self._cache_edge_shifts is not None:
            edge_shifts = self._cache_edge_shifts.zero_()
        else:
            edge_shifts = torch.zeros(npairs, 3, dtype=self.dtype, device=self.device)
            self._cache_edge_shifts = edge_shifts
            self._cache_npairs = npairs
        if self._cache_cell is None:
            self._cache_cell = torch.eye(3, dtype=self.dtype, device=self.device).unsqueeze(0) * 100.0
        cell = self._cache_cell

        # --- Forward (atom forces via dE/d(pos)) ---
        E_total, atom_energies, atom_forces = self.wrapper(
            rij, species, batch, edge_src, edge_dst,
            edge_shifts, cell, nlocal=natoms,
        )

        # --- Write back to LAMMPS ---
        # NOTE: .item() forces GPU→CPU sync; unavoidable for LAMMPS energy
        data.energy = E_total.item()

        atom_e = atom_energies.squeeze(-1).detach()
        forces = atom_forces.detach()
        if forces.dtype != torch.float64:
            forces = forces.to(torch.float64)

        if self._using_kokkos and self._has_gpu_api:
            self._writeback_kokkos(data, atom_e, forces, natoms, ntotal)
        else:
            self._writeback_cpu(data, atom_e, forces, natoms, ntotal)

    # ------------------------------------------------------------------
    # Write-back helpers
    # ------------------------------------------------------------------

    def _writeback_cpu(self, data, atom_e, forces, natoms, ntotal):
        """CPU path: write per-atom energies and forces via numpy views."""
        ae_np = atom_e.cpu().numpy().astype(np.float64)
        data.eatoms = ae_np

        f_view = data.f
        forces_np = forces.cpu().numpy()
        flat_f = np.asarray(f_view).ravel()
        flat_f[:ntotal * 3] += forces_np[:ntotal].ravel()

    def _writeback_kokkos(self, data, atom_e, forces, natoms, ntotal):
        """Kokkos GPU path: write per-atom energies and forces via GPU tensors."""
        eatoms_t = self._to_torch(data.eatoms, device=self.device)
        eatoms_t.copy_(atom_e[:natoms].to(eatoms_t.dtype))

        f_t = self._to_torch(data.f, device=self.device)
        f_flat = f_t.view(-1)
        f_flat[:ntotal * 3] += forces[:ntotal].to(f_t.dtype).view(-1)

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
