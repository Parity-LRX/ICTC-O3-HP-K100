"""Model/force/TorchScript adapters shared by checkpoint consumers."""
from __future__ import annotations

import io
import inspect
import os
from typing import List, Optional, Tuple
import torch
from torch import nn
from mace_ictc.models.pure_cartesian_ictd_layers import PhysicalTensorICTDRecovery
from mace_ictc.utils.tensor_utils import map_tensor_values


_GLOBAL_PHYS_LAYOUT = (
    ("charge", 1),
    ("dipole", 3),
    ("polarizability", 9),
    ("quadrupole", 9),
)
_ATOM_PHYS_LAYOUT = (
    ("charge_per_atom", 1),
    ("dipole_per_atom", 3),
    ("polarizability_per_atom", 9),
    ("quadrupole_per_atom", 9),
    ("born_effective_charge_per_atom", 9),
)


def _layout_total_dim(layout: tuple[tuple[str, int], ...]) -> int:
    return sum(dim for _, dim in layout)


def _layout_offsets(layout: tuple[tuple[str, int], ...]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    start = 0
    for name, dim in layout:
        out[name] = (start, start + dim)
        start += dim
    return out


_GLOBAL_PHYS_TOTAL_DIM = _layout_total_dim(_GLOBAL_PHYS_LAYOUT)
_ATOM_PHYS_TOTAL_DIM = _layout_total_dim(_ATOM_PHYS_LAYOUT)
_GLOBAL_PHYS_OFFSETS = _layout_offsets(_GLOBAL_PHYS_LAYOUT)
_ATOM_PHYS_OFFSETS = _layout_offsets(_ATOM_PHYS_LAYOUT)
_PHYS_MASK_WIDTH = max(len(_GLOBAL_PHYS_LAYOUT), len(_ATOM_PHYS_LAYOUT))


def _resolve_model_external_tensor_rank(model: nn.Module) -> int | None:
    ext_rank = getattr(model, "external_tensor_rank", None)
    if ext_rank is not None:
        return int(ext_rank)
    conv = getattr(model, "e3_conv_emb", None)
    ext_rank = getattr(conv, "external_tensor_rank", None)
    if ext_rank is not None:
        return int(ext_rank)
    nested = getattr(model, "model", None)
    if nested is not None and nested is not model:
        return _resolve_model_external_tensor_rank(nested)
    return None


def _resolve_model_external_tensor_specs(model: nn.Module) -> list[dict] | None:
    specs = getattr(model, "external_tensor_specs", None)
    if specs is not None:
        return list(specs)
    conv = getattr(model, "e3_conv_emb", None)
    specs = getattr(conv, "external_tensor_specs", None)
    if specs is not None:
        return list(specs)
    nested = getattr(model, "model", None)
    if nested is not None and nested is not model:
        return _resolve_model_external_tensor_specs(nested)
    return None


def _resolve_model_external_tensor_total_numel(model: nn.Module) -> int:
    total = getattr(model, "external_tensor_total_numel", None)
    if total is not None:
        return int(total)
    conv = getattr(model, "e3_conv_emb", None)
    total = getattr(conv, "external_tensor_total_numel", None)
    if total is not None:
        return int(total)
    specs = _resolve_model_external_tensor_specs(model)
    if specs:
        return int(sum(int(spec.get("numel", 0)) for spec in specs))
    ext_rank = _resolve_model_external_tensor_rank(model)
    if ext_rank is None:
        return 0
    return 3 ** int(ext_rank) if int(ext_rank) > 0 else 1


def _canonicalize_cartesian_rank2(x: torch.Tensor) -> torch.Tensor:
    # Keep the user-facing LAMMPS output simple: always expose a full 3x3 Cartesian tensor.
    return x.reshape(*x.shape[:-2], 9)


def _physical_block_is_cartesian(block: torch.Tensor, rank: int) -> bool:
    if block is None:
        return False
    if rank == 0:
        return block.dim() >= 2 and block.shape[-1] == 1
    if rank == 1:
        return block.dim() >= 3 and block.shape[-1] == 3
    if rank == 2:
        return block.dim() >= 4 and block.shape[-2:] == (3, 3)
    return False


def _recover_rank1_cartesian(blocks: dict[int, torch.Tensor]) -> torch.Tensor:
    blk = blocks.get(1)
    if blk is None:
        raise ValueError("Missing l=1 block for rank-1 Cartesian physical tensor")
    if not _physical_block_is_cartesian(blk, 1):
        raise ValueError(f"rank-1 Cartesian block must have shape (..., C, 3), got {tuple(blk.shape)}")
    if blk.shape[-2] != 1:
        raise ValueError("rank-1 Cartesian physical tensor export currently requires channels_out=1")
    return blk[..., 0, :]


def _recover_rank2_cartesian(blocks: dict[int, torch.Tensor], *, include_trace: bool) -> torch.Tensor:
    mat = None
    if 2 in blocks:
        blk = blocks[2]
        if not _physical_block_is_cartesian(blk, 2):
            raise ValueError(f"rank-2 Cartesian block must have shape (..., C, 3, 3), got {tuple(blk.shape)}")
        if blk.shape[-3] != 1:
            raise ValueError("rank-2 Cartesian physical tensor export currently requires channels_out=1")
        mat = blk[..., 0, :, :]
    if mat is None:
        if include_trace:
            raise ValueError("Missing l=2 block for rank-2 Cartesian physical tensor")
        return torch.zeros(0)
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    if include_trace and 0 in blocks:
        trace_blk = blocks[0]
        if not _physical_block_is_cartesian(trace_blk, 0):
            raise ValueError(f"rank-0 Cartesian block must have shape (..., C, 1), got {tuple(trace_blk.shape)}")
        if trace_blk.shape[-2] != 1:
            raise ValueError("rank-0 Cartesian physical tensor export currently requires channels_out=1")
        trace = trace_blk[..., 0, 0]
        eye = torch.eye(3, device=mat.device, dtype=mat.dtype)
        mat = mat + trace.unsqueeze(-1).unsqueeze(-1) * eye
    return mat


def _recover_cartesian_physical_tensors(
    physical_out: dict[str, dict[int, torch.Tensor]] | None,
    *,
    num_graphs: int,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global_phys = torch.zeros(num_graphs, _GLOBAL_PHYS_TOTAL_DIM, device=device, dtype=dtype)
    atom_phys = torch.zeros(num_nodes, _ATOM_PHYS_TOTAL_DIM, device=device, dtype=dtype)
    global_mask = torch.zeros(_PHYS_MASK_WIDTH, device=device, dtype=dtype)
    atom_mask = torch.zeros(_PHYS_MASK_WIDTH, device=device, dtype=dtype)
    if not physical_out:
        return global_phys, atom_phys, global_mask, atom_mask

    recover_rank1 = PhysicalTensorICTDRecovery(rank=1, channels_in=1, lmax_in=1, include_trace_chain=False).to(device)
    recover_rank2_full = PhysicalTensorICTDRecovery(
        rank=2,
        channels_in=1,
        lmax_in=2,
        include_trace_chain=True,
        rank2_mode="symmetric",
    ).to(device)
    recover_rank2_l2 = PhysicalTensorICTDRecovery(rank=2, channels_in=1, lmax_in=2, include_trace_chain=False).to(device)
    recover_rank2_bec = PhysicalTensorICTDRecovery(
        rank=2,
        channels_in=1,
        lmax_in=2,
        include_trace_chain=True,
        rank2_mode="full",
    ).to(device)

    global_mask_index = {name: idx for idx, (name, _) in enumerate(_GLOBAL_PHYS_LAYOUT)}
    atom_mask_index = {name: idx for idx, (name, _) in enumerate(_ATOM_PHYS_LAYOUT)}

    for name, blocks in physical_out.items():
        if not isinstance(blocks, dict):
            continue
        if name in _GLOBAL_PHYS_OFFSETS:
            start, end = _GLOBAL_PHYS_OFFSETS[name]
            if name == "charge":
                blk = blocks.get(0)
                if blk is None:
                    continue
                if _physical_block_is_cartesian(blk, 0):
                    if blk.shape[-2] != 1:
                        raise ValueError(f"{name} channels_out>1 is not supported for LibTorch export")
                    global_phys[:, start:end] = blk[..., 0, 0].reshape(blk.shape[0], 1)
                else:
                    if blk.shape[-2] != 1:
                        raise ValueError(f"{name} channels_out>1 is not supported for LibTorch export")
                    global_phys[:, start:end] = blk[..., 0, :].reshape(blk.shape[0], 1)
            elif name == "dipole":
                rec = _recover_rank1_cartesian(blocks) if _physical_block_is_cartesian(blocks.get(1), 1) else recover_rank1(blocks)
                global_phys[:, start:end] = rec.reshape(rec.shape[0], 3)
            elif name == "polarizability":
                rec = (
                    _recover_rank2_cartesian(blocks, include_trace=True)
                    if _physical_block_is_cartesian(blocks.get(2), 2)
                    else recover_rank2_full(blocks)
                )
                global_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            elif name == "quadrupole":
                l2_only = {2: blocks[2]} if 2 in blocks else {}
                if not l2_only:
                    continue
                rec = (
                    _recover_rank2_cartesian(l2_only, include_trace=False)
                    if _physical_block_is_cartesian(l2_only.get(2), 2)
                    else recover_rank2_l2(l2_only)
                )
                global_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            global_mask[global_mask_index[name]] = 1.0
        elif name in _ATOM_PHYS_OFFSETS:
            start, end = _ATOM_PHYS_OFFSETS[name]
            if name == "charge_per_atom":
                blk = blocks.get(0)
                if blk is None:
                    continue
                if _physical_block_is_cartesian(blk, 0):
                    if blk.shape[-2] != 1:
                        raise ValueError(f"{name} channels_out>1 is not supported for LibTorch export")
                    atom_phys[:, start:end] = blk[..., 0, 0].reshape(blk.shape[0], 1)
                else:
                    if blk.shape[-2] != 1:
                        raise ValueError(f"{name} channels_out>1 is not supported for LibTorch export")
                    atom_phys[:, start:end] = blk[..., 0, :].reshape(blk.shape[0], 1)
            elif name == "dipole_per_atom":
                rec = _recover_rank1_cartesian(blocks) if _physical_block_is_cartesian(blocks.get(1), 1) else recover_rank1(blocks)
                atom_phys[:, start:end] = rec.reshape(rec.shape[0], 3)
            elif name == "polarizability_per_atom":
                rec = (
                    _recover_rank2_cartesian(blocks, include_trace=True)
                    if _physical_block_is_cartesian(blocks.get(2), 2)
                    else recover_rank2_full(blocks)
                )
                atom_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            elif name == "quadrupole_per_atom":
                l2_only = {2: blocks[2]} if 2 in blocks else {}
                if not l2_only:
                    continue
                rec = (
                    _recover_rank2_cartesian(l2_only, include_trace=False)
                    if _physical_block_is_cartesian(l2_only.get(2), 2)
                    else recover_rank2_l2(l2_only)
                )
                atom_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            elif name == "born_effective_charge_per_atom":
                if _physical_block_is_cartesian(blocks.get(2), 2):
                    rec = _recover_rank2_cartesian(blocks, include_trace=True)
                    if 1 in blocks and _physical_block_is_cartesian(blocks.get(1), 1):
                        anti = blocks[1][..., 0, :]
                        rec = rec.clone()
                        rec[:, 1, 2] += anti[:, 0]
                        rec[:, 2, 1] -= anti[:, 0]
                        rec[:, 2, 0] += anti[:, 1]
                        rec[:, 0, 2] -= anti[:, 1]
                        rec[:, 0, 1] += anti[:, 2]
                        rec[:, 1, 0] -= anti[:, 2]
                else:
                    rec = recover_rank2_bec(blocks)
                atom_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            atom_mask[atom_mask_index[name]] = 1.0

    return global_phys, atom_phys, global_mask, atom_mask


def _warm_ictd_projector_caches_for_trace(model: nn.Module) -> None:
    """Build ICTD CG projector caches before TorchScript tracing.

    If the cache is cold, tracing records the Python projector construction
    graph itself.  That construction includes temporary tensor reshapes that
    are unrelated to the input graph but can replay incorrectly when LAMMPS
    later calls the traced core with a different atom/edge count.  Warming the
    cache makes the projector tensors enter the traced graph as constants.
    """
    for module in model.modules():
        getter = getattr(module, "_get_proj_group_list", None)
        if not callable(getter):
            continue
        try:
            ref = next(module.parameters(), None)
        except Exception:
            ref = None
        if ref is None:
            try:
                ref = next(module.buffers(), None)
            except Exception:
                ref = None
        device = ref.device if ref is not None else torch.device("cpu")
        dtype = ref.dtype if ref is not None and ref.is_floating_point() else torch.get_default_dtype()
        try:
            getter(device=device, dtype=dtype)
        except Exception:
            # Cache warming is an export robustness optimization; leave the
            # original model error path intact if an unusual module rejects it.
            continue


# ---------------------------------------------------------------------------
# Multi-GPU message passing via LAMMPS Kokkos forward/reverse exchange
# ---------------------------------------------------------------------------



class _TorchScriptEdgeVecCore(nn.Module):
    """Core wrapper to make precomputed_edge_vec traceable (positional arg).

    LAMMPS LibTorch 接口始终返回 per-atom energy；若模型配置了 physical tensor
    heads，则额外导出固定 schema 的体系级/逐原子笛卡尔物理张量，供 C++ 侧缓存。
    """

    def __init__(self, model: nn.Module, *, export_reciprocal_source: bool = False):
        super().__init__()
        _warm_ictd_projector_caches_for_trace(model)
        self.model = model
        self.external_tensor_rank = _resolve_model_external_tensor_rank(model)
        self.external_tensor_specs = _resolve_model_external_tensor_specs(model)
        self.external_tensor_total_numel = _resolve_model_external_tensor_total_numel(model)
        self.has_physical_tensor_heads = (
            hasattr(model, "physical_tensor_heads") and getattr(model, "physical_tensor_heads", None) is not None
        )
        self.export_reciprocal_source = bool(export_reciprocal_source)

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        edge_vec: torch.Tensor,
        dispersion_edge_src: torch.Tensor,
        dispersion_edge_dst: torch.Tensor,
        dispersion_edge_shifts: torch.Tensor,
        dispersion_edge_vec: torch.Tensor,
        external_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            kwargs = {
                "precomputed_edge_vec": edge_vec,
                "dispersion_edge_src": dispersion_edge_src,
                "dispersion_edge_dst": dispersion_edge_dst,
                "dispersion_edge_shifts": dispersion_edge_shifts,
                "precomputed_dispersion_edge_vec": dispersion_edge_vec,
                "return_physical_tensors": self.has_physical_tensor_heads,
                "return_reciprocal_source": self.export_reciprocal_source,
            }
            if self.external_tensor_total_numel > 0:
                kwargs["external_tensor"] = external_tensor
            out = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)
        except TypeError:
            kwargs = {
                "precomputed_edge_vec": edge_vec,
                "dispersion_edge_src": dispersion_edge_src,
                "dispersion_edge_dst": dispersion_edge_dst,
                "dispersion_edge_shifts": dispersion_edge_shifts,
                "precomputed_dispersion_edge_vec": dispersion_edge_vec,
                "return_physical_tensors": self.has_physical_tensor_heads,
                "return_reciprocal_source": self.export_reciprocal_source,
            }
            if self.external_tensor_total_numel > 0:
                kwargs["external_tensor"] = external_tensor
            out = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)

        reciprocal_source = torch.empty((pos.shape[0], 0), device=pos.device, dtype=pos.dtype)
        if isinstance(out, tuple):
            atom_energy = out[0]
            if self.has_physical_tensor_heads:
                physical_out = out[1] if len(out) > 1 else None
                if self.export_reciprocal_source and len(out) > 2:
                    reciprocal_source = out[2]
            else:
                physical_out = None
                if self.export_reciprocal_source and len(out) > 1:
                    reciprocal_source = out[1]
        else:
            atom_energy = out
            physical_out = None
        global_phys, atom_phys, global_mask, atom_mask = _recover_cartesian_physical_tensors(
            physical_out,
            num_graphs=cell.size(0),
            num_nodes=pos.shape[0],
            device=pos.device,
            dtype=atom_energy.dtype,
        )
        return atom_energy, global_phys, atom_phys, global_mask, atom_mask, reciprocal_source


class _TorchScriptEdgeVecCoreWithFidelity(nn.Module):
    """Core wrapper with a runtime graph-level fidelity id positional input."""

    def __init__(self, model: nn.Module, *, export_reciprocal_source: bool = False):
        super().__init__()
        _warm_ictd_projector_caches_for_trace(model)
        self.model = model
        self.external_tensor_rank = _resolve_model_external_tensor_rank(model)
        self.external_tensor_specs = _resolve_model_external_tensor_specs(model)
        self.external_tensor_total_numel = _resolve_model_external_tensor_total_numel(model)
        self.has_physical_tensor_heads = (
            hasattr(model, "physical_tensor_heads") and getattr(model, "physical_tensor_heads", None) is not None
        )
        self.export_reciprocal_source = bool(export_reciprocal_source)
        self.num_fidelity_levels = int(getattr(model, "num_fidelity_levels", 0) or 0)

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        edge_vec: torch.Tensor,
        dispersion_edge_src: torch.Tensor,
        dispersion_edge_dst: torch.Tensor,
        dispersion_edge_shifts: torch.Tensor,
        dispersion_edge_vec: torch.Tensor,
        external_tensor: torch.Tensor,
        fidelity_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = {
            "precomputed_edge_vec": edge_vec,
            "dispersion_edge_src": dispersion_edge_src,
            "dispersion_edge_dst": dispersion_edge_dst,
            "dispersion_edge_shifts": dispersion_edge_shifts,
            "precomputed_dispersion_edge_vec": dispersion_edge_vec,
            "return_physical_tensors": self.has_physical_tensor_heads,
            "return_reciprocal_source": self.export_reciprocal_source,
            "fidelity_ids": fidelity_ids.to(device=pos.device, dtype=torch.long).view(-1),
        }
        if self.external_tensor_total_numel > 0:
            kwargs["external_tensor"] = external_tensor
        out = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)

        reciprocal_source = torch.empty((pos.shape[0], 0), device=pos.device, dtype=pos.dtype)
        if isinstance(out, tuple):
            atom_energy = out[0]
            if self.has_physical_tensor_heads:
                physical_out = out[1] if len(out) > 1 else None
                if self.export_reciprocal_source and len(out) > 2:
                    reciprocal_source = out[2]
            else:
                physical_out = None
                if self.export_reciprocal_source and len(out) > 1:
                    reciprocal_source = out[1]
        else:
            atom_energy = out
            physical_out = None
        global_phys, atom_phys, global_mask, atom_mask = _recover_cartesian_physical_tensors(
            physical_out,
            num_graphs=cell.size(0),
            num_nodes=pos.shape[0],
            device=pos.device,
            dtype=atom_energy.dtype,
        )
        return atom_energy, global_phys, atom_phys, global_mask, atom_mask, reciprocal_source


class _TorchScriptEdgeVecAdapter(nn.Module):
    """Adapter that preserves the original forward signature used by AtomForcesWrapper."""

    def __init__(self, core: torch.jit.ScriptModule):
        super().__init__()
        self.core = core
        self._refresh_schema_flags()

    def _refresh_schema_flags(self) -> None:
        self.core_takes_dispersion_edges_arg = False
        try:
            schema = self.core.forward.schema
            nargs = len(schema.arguments)
            if nargs > 0 and schema.arguments[0].name == "self":
                nargs -= 1
            self.core_takes_dispersion_edges_arg = nargs >= 13
        except Exception:
            self.core_takes_dispersion_edges_arg = False

    def __getstate__(self):
        # Make this module picklable via torch.save by serializing the ScriptModule to bytes.
        buf = io.BytesIO()
        torch.jit.save(self.core, buf)
        return {"core_bytes": buf.getvalue()}

    def __setstate__(self, state):
        # Restore ScriptModule from bytes.
        # IMPORTANT: TorchScript graphs may contain constant tensors that do NOT move with `.to()`.
        # So we should load constants onto the right device up-front.
        nn.Module.__init__(self)
        pref = os.environ.get("MLIAP_TORCHSCRIPT_MAP_LOCATION", "").strip().lower()
        if pref in ("cpu", "cuda"):
            map_loc = pref
        else:
            map_loc = "cuda" if torch.cuda.is_available() else "cpu"
        core = torch.jit.load(io.BytesIO(state["core_bytes"]), map_location=map_loc)
        self.core = core
        self._refresh_schema_flags()

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        *,
        precomputed_edge_vec: Optional[torch.Tensor] = None,
        dispersion_edge_src: Optional[torch.Tensor] = None,
        dispersion_edge_dst: Optional[torch.Tensor] = None,
        dispersion_edge_shifts: Optional[torch.Tensor] = None,
        precomputed_dispersion_edge_vec: Optional[torch.Tensor] = None,
        external_tensor: Optional[torch.Tensor] = None,
        sync_after_scatter=None,
    ) -> torch.Tensor:
        if precomputed_edge_vec is None:
            raise ValueError("TorchScript model requires precomputed_edge_vec")
        # sync_after_scatter is ignored in TorchScript mode.
        if external_tensor is None:
            external_tensor = torch.empty(0, dtype=pos.dtype, device=pos.device)
        if self.core_takes_dispersion_edges_arg:
            if dispersion_edge_src is None:
                dispersion_edge_src = edge_src
            if dispersion_edge_dst is None:
                dispersion_edge_dst = edge_dst
            if dispersion_edge_shifts is None:
                dispersion_edge_shifts = edge_shifts
            if precomputed_dispersion_edge_vec is None:
                precomputed_dispersion_edge_vec = precomputed_edge_vec
            out = self.core(
                pos,
                A,
                batch,
                edge_src,
                edge_dst,
                edge_shifts,
                cell,
                precomputed_edge_vec,
                dispersion_edge_src,
                dispersion_edge_dst,
                dispersion_edge_shifts,
                precomputed_dispersion_edge_vec,
                external_tensor,
            )
        else:
            out = self.core(
                pos, A, batch, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec, external_tensor
            )
        return out[0] if isinstance(out, tuple) else out


class _TorchScriptEdgeVecAdapterWithFidelity(nn.Module):
    """Adapter for TorchScript cores that require runtime fidelity_ids."""

    def __init__(self, core: torch.jit.ScriptModule):
        super().__init__()
        self.core = core
        self._refresh_schema_flags()

    def _refresh_schema_flags(self) -> None:
        self.core_takes_dispersion_edges_arg = False
        try:
            schema = self.core.forward.schema
            nargs = len(schema.arguments)
            if nargs > 0 and schema.arguments[0].name == "self":
                nargs -= 1
            self.core_takes_dispersion_edges_arg = nargs >= 14
        except Exception:
            self.core_takes_dispersion_edges_arg = False

    def __getstate__(self):
        buf = io.BytesIO()
        torch.jit.save(self.core, buf)
        return {"core_bytes": buf.getvalue()}

    def __setstate__(self, state):
        nn.Module.__init__(self)
        pref = os.environ.get("MLIAP_TORCHSCRIPT_MAP_LOCATION", "").strip().lower()
        if pref in ("cpu", "cuda"):
            map_loc = pref
        else:
            map_loc = "cuda" if torch.cuda.is_available() else "cpu"
        self.core = torch.jit.load(io.BytesIO(state["core_bytes"]), map_location=map_loc)
        self._refresh_schema_flags()

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        *,
        precomputed_edge_vec: Optional[torch.Tensor] = None,
        dispersion_edge_src: Optional[torch.Tensor] = None,
        dispersion_edge_dst: Optional[torch.Tensor] = None,
        dispersion_edge_shifts: Optional[torch.Tensor] = None,
        precomputed_dispersion_edge_vec: Optional[torch.Tensor] = None,
        external_tensor: Optional[torch.Tensor] = None,
        fidelity_ids: Optional[torch.Tensor] = None,
        sync_after_scatter=None,
    ) -> torch.Tensor:
        if precomputed_edge_vec is None:
            raise ValueError("TorchScript model requires precomputed_edge_vec")
        if fidelity_ids is None:
            raise ValueError("TorchScript model requires fidelity_ids")
        if external_tensor is None:
            external_tensor = torch.empty(0, dtype=pos.dtype, device=pos.device)
        if self.core_takes_dispersion_edges_arg:
            if dispersion_edge_src is None:
                dispersion_edge_src = edge_src
            if dispersion_edge_dst is None:
                dispersion_edge_dst = edge_dst
            if dispersion_edge_shifts is None:
                dispersion_edge_shifts = edge_shifts
            if precomputed_dispersion_edge_vec is None:
                precomputed_dispersion_edge_vec = precomputed_edge_vec
            out = self.core(
                pos,
                A,
                batch,
                edge_src,
                edge_dst,
                edge_shifts,
                cell,
                precomputed_edge_vec,
                dispersion_edge_src,
                dispersion_edge_dst,
                dispersion_edge_shifts,
                precomputed_dispersion_edge_vec,
                external_tensor,
                fidelity_ids.to(device=pos.device, dtype=torch.long).view(-1),
            )
        else:
            out = self.core(
                pos,
                A,
                batch,
                edge_src,
                edge_dst,
                edge_shifts,
                cell,
                precomputed_edge_vec,
                external_tensor,
                fidelity_ids.to(device=pos.device, dtype=torch.long).view(-1),
            )
        return out[0] if isinstance(out, tuple) else out


def _maybe_torchscript_trace_model(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    enable: bool,
    export_reciprocal_source: bool = False,
) -> nn.Module:
    """Optionally trace a model to TorchScript for faster Python dispatch."""
    if not enable:
        return model

    # Only trace in eval mode; gradients w.r.t. inputs are still supported.
    model.eval()

    # Trace a core wrapper that takes edge_vec as positional arg.
    num_fidelity_levels = int(getattr(model, "num_fidelity_levels", 0) or 0)
    fixed_fidelity_id = getattr(model, "fixed_fidelity_id", None)
    runtime_fidelity = num_fidelity_levels > 0 and fixed_fidelity_id is None
    core_cls = _TorchScriptEdgeVecCoreWithFidelity if runtime_fidelity else _TorchScriptEdgeVecCore
    core = core_cls(model, export_reciprocal_source=export_reciprocal_source).to(device=device)

    ext_total_numel = _resolve_model_external_tensor_total_numel(model)

    # Example inputs (dynamic shapes should still work for most ops).
    N = 32
    E = 256
    pos = torch.zeros(N, 3, device=device, dtype=dtype)
    A = torch.ones(N, device=device, dtype=torch.long)
    batch = torch.zeros(N, device=device, dtype=torch.long)
    edge_src = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_dst = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_shifts = torch.zeros(E, 3, device=device, dtype=dtype)
    cell = (torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 100.0)
    edge_vec = torch.randn(E, 3, device=device, dtype=dtype)
    dispersion_edge_src = edge_src.clone()
    dispersion_edge_dst = edge_dst.clone()
    dispersion_edge_shifts = edge_shifts.clone()
    dispersion_edge_vec = edge_vec.clone()
    if ext_total_numel <= 0:
        external_tensor = torch.empty(0, device=device, dtype=dtype)
    else:
        external_tensor = torch.zeros(ext_total_numel, device=device, dtype=dtype)
    fidelity_ids = torch.zeros(1, device=device, dtype=torch.long)

    try:
        # Prewarm one-time caches before tracing to keep Python-side setup out of the trace.
        try:
            with torch.no_grad():
                for m in core.modules():
                    prewarm = getattr(m, "prewarm_caches", None)
                    if callable(prewarm):
                        prewarm(device=device, dtype=dtype)
                # One eager run to lock in branches and fill caches.
                if runtime_fidelity:
                    _ = core(
                        pos,
                        A,
                        batch,
                        edge_src,
                        edge_dst,
                        edge_shifts,
                        cell,
                        edge_vec,
                        dispersion_edge_src,
                        dispersion_edge_dst,
                        dispersion_edge_shifts,
                        dispersion_edge_vec,
                        external_tensor,
                        fidelity_ids,
                    )
                else:
                    _ = core(
                        pos,
                        A,
                        batch,
                        edge_src,
                        edge_dst,
                        edge_shifts,
                        cell,
                        edge_vec,
                        dispersion_edge_src,
                        dispersion_edge_dst,
                        dispersion_edge_shifts,
                        dispersion_edge_vec,
                        external_tensor,
                    )
        except Exception:
            pass

        if runtime_fidelity:
            trace_inputs = (
                pos,
                A,
                batch,
                edge_src,
                edge_dst,
                edge_shifts,
                cell,
                edge_vec,
                dispersion_edge_src,
                dispersion_edge_dst,
                dispersion_edge_shifts,
                dispersion_edge_vec,
                external_tensor,
                fidelity_ids,
            )
        else:
            trace_inputs = (
                pos,
                A,
                batch,
                edge_src,
                edge_dst,
                edge_shifts,
                cell,
                edge_vec,
                dispersion_edge_src,
                dispersion_edge_dst,
                dispersion_edge_shifts,
                dispersion_edge_vec,
                external_tensor,
            )
        core_ts = torch.jit.trace(core, trace_inputs, check_trace=False, strict=False)
        try:
            core_ts = torch.jit.freeze(core_ts.eval())
        except Exception:
            core_ts = core_ts.eval()
        return _TorchScriptEdgeVecAdapterWithFidelity(core_ts) if runtime_fidelity else _TorchScriptEdgeVecAdapter(core_ts)
    except Exception as e:
        raise RuntimeError(f"TorchScript trace failed: {e}")


class AtomForcesWrapper(nn.Module):
    """Wrapper that computes per-atom energies and per-atom forces via autograd on pos.

    Instead of differentiating through edge_vec (O(npairs) leaf gradient), this
    wrapper uses a dummy ``pos`` tensor as the autograd leaf and constructs
    ``edge_vec = pos[dst] - pos[src] + rij`` so that the gradient accumulates
    into per-atom forces (O(natoms) leaf gradient).
    """

    def __init__(self, model: nn.Module, atomic_energy_keys: torch.Tensor,
                 atomic_energy_values: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_energy_keys", atomic_energy_keys)
        self.register_buffer("atomic_energy_values", atomic_energy_values)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _build_forward_kwargs(
        self,
        *,
        edge_vec: torch.Tensor,
        external_tensor: Optional[torch.Tensor],
        fidelity_ids: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        params = inspect.signature(self.model.forward).parameters
        kwargs: dict[str, torch.Tensor] = {}
        if "precomputed_edge_vec" in params:
            kwargs["precomputed_edge_vec"] = edge_vec
        if external_tensor is not None and "external_tensor" in params:
            kwargs["external_tensor"] = external_tensor
        if fidelity_ids is not None and "fidelity_ids" in params:
            kwargs["fidelity_ids"] = fidelity_ids
        return kwargs

    def forward(
        self,
        rij: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        nlocal: int,
        external_tensor: Optional[torch.Tensor] = None,
        fidelity_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total energy, per-atom energies and per-atom forces.

        Args:
            rij: (E, 3) pair distance vectors from LAMMPS (detached)
            A: (N,) atomic numbers
            batch: (N,) batch indices (all zeros for single structure)
            edge_src/edge_dst: (E,) edge indices
            edge_shifts: (E, 3) PBC shift integers
            cell: (1, 3, 3) cell matrix
            nlocal: number of local (owned) atoms

        Returns:
            (total_energy, atom_energies[:nlocal], atom_forces) where
            atom_forces = -dE/d(pos), shape (N, 3)
        """
        ntotal = A.size(0)
        pos = torch.zeros(ntotal, 3, dtype=rij.dtype, device=rij.device,
                          requires_grad=True)

        edge_vec = pos[edge_dst] - pos[edge_src] + rij.detach()

        kwargs = self._build_forward_kwargs(
            edge_vec=edge_vec,
            external_tensor=external_tensor,
            fidelity_ids=fidelity_ids,
        )
        atom_energies = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)

        mapped_A = map_tensor_values(
            A.to(dtype=self.atomic_energy_values.dtype),
            self.atomic_energy_keys,
            self.atomic_energy_values,
        )
        E_offset = mapped_A[:nlocal].sum()
        E_total = atom_energies[:nlocal].sum() + E_offset

        neg_forces = torch.autograd.grad(E_total, pos, create_graph=False)[0]
        atom_forces = -neg_forces

        return E_total, atom_energies[:nlocal].detach(), atom_forces.detach()


class EdgeForcesWrapper(nn.Module):
    """Legacy wrapper: per-pair forces via autograd on edge_vec (O(npairs) gradient).

    Kept for backward compatibility.  Prefer :class:`AtomForcesWrapper`.
    """

    def __init__(self, model: nn.Module, atomic_energy_keys: torch.Tensor,
                 atomic_energy_values: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_energy_keys", atomic_energy_keys)
        self.register_buffer("atomic_energy_values", atomic_energy_values)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _build_forward_kwargs(
        self,
        *,
        edge_vec: torch.Tensor,
        external_tensor: Optional[torch.Tensor],
        fidelity_ids: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        params = inspect.signature(self.model.forward).parameters
        kwargs: dict[str, torch.Tensor] = {}
        if "precomputed_edge_vec" in params:
            kwargs["precomputed_edge_vec"] = edge_vec
        if external_tensor is not None and "external_tensor" in params:
            kwargs["external_tensor"] = external_tensor
        if fidelity_ids is not None and "fidelity_ids" in params:
            kwargs["fidelity_ids"] = fidelity_ids
        return kwargs

    def forward(
        self,
        edge_vec: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        nlocal: int,
        external_tensor: Optional[torch.Tensor] = None,
        fidelity_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total energy, per-atom energies and per-pair forces."""
        pos = torch.zeros(A.size(0), 3, dtype=edge_vec.dtype, device=edge_vec.device)

        kwargs = self._build_forward_kwargs(
            edge_vec=edge_vec,
            external_tensor=external_tensor,
            fidelity_ids=fidelity_ids,
        )
        atom_energies = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)

        mapped_A = map_tensor_values(
            A.to(dtype=self.atomic_energy_values.dtype),
            self.atomic_energy_keys,
            self.atomic_energy_values,
        )
        E_offset = mapped_A[:nlocal].sum()
        E_total = atom_energies[:nlocal].sum() + E_offset

        pair_forces = torch.autograd.grad(E_total, edge_vec, create_graph=False)[0]

        return E_total, atom_energies[:nlocal].detach(), pair_forces.detach()


