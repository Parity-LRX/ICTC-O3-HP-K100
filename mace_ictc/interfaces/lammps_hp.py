# SPDX-License-Identifier: MIT
"""Native K100 MFF execution and differentiable, layerwise RCCL transport.

The small TorchScript archive calls this installed Python runtime. It is a
dispatch envelope, not a traced copy of the network or an AOTInductor package.
Each force evaluation uses the current LAMMPS owner map; topology is never cached.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import shutil
import uuid

import torch
import torch.distributed as dist

_RUNTIMES: dict[tuple[int, int], "_Runtime"] = {}
_OWN_GROUP = False


def _distributed(rank: int, size: int, device: torch.device) -> None:
    global _OWN_GROUP
    if size == 1:
        return
    if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) != 1:
        raise ValueError("MFF native RCCL currently supports one node")
    launcher_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", str(size))))
    if launcher_size != size:
        raise ValueError("MFF native RCCL requires the full launcher communicator; LAMMPS partitions are unsupported")
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(30000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000))
        dist.init_process_group("nccl", rank=rank, world_size=size,
            timeout=datetime.timedelta(seconds=120), device_id=device)
        _OWN_GROUP = True
    if dist.get_rank() != rank or dist.get_world_size() != size or dist.get_backend() != "nccl":
        raise ValueError("Existing process group does not match the MFF RCCL communicator")


def shutdown() -> None:
    """Release the RCCL group created by this runtime before MPI finalization."""
    global _OWN_GROUP
    if _OWN_GROUP and dist.is_initialized():
        dist.destroy_process_group()
    _OWN_GROUP = False


class _Plan:
    """Exact current ghost owners, including repeated periodic images."""
    def __init__(self, owners: torch.Tensor, owned: int, total: int, rank: int, size: int,
                 device: torch.device):
        _distributed(rank, size, device)
        if owners.shape != (total, 2) or owners.dtype != torch.int64:
            raise ValueError("Expected an int64 [owned+ghost,2] LAMMPS owner map")
        self.owned, self.total = owned, total
        # Requests are small integers. Keep their assembly on the host; only
        # index and FP32 feature payloads use RCCL. Never match coordinates.
        raw = owners[owned:].cpu()
        if raw.numel() and (bool((raw[:, 0] < 0).any()) or bool((raw[:, 0] >= size).any())
                            or bool((raw[:, 1] < 0).any())):
            raise ValueError("Invalid LAMMPS ghost owner map")
        requests = [raw[raw[:, 0] == peer, 1] for peer in range(size)]
        self.recv_counts = [rows.numel() for rows in requests]
        counts = torch.tensor(self.recv_counts, device=device, dtype=torch.int64)
        incoming = torch.empty_like(counts)
        dist.all_to_all_single(incoming, counts)
        self.send_counts = incoming.cpu().tolist()
        request_rows = torch.cat(requests).to(device)
        self.send_rows = torch.empty(sum(self.send_counts), device=device, dtype=torch.int64)
        dist.all_to_all_single(self.send_rows, request_rows,
            output_split_sizes=self.send_counts, input_split_sizes=self.recv_counts)
        if self.send_rows.numel() and (bool((self.send_rows < 0).any()) or bool((self.send_rows >= owned).any())):
            raise ValueError("LAMMPS ghost request refers to a non-owned row")
        self.recv_rows = torch.cat([torch.nonzero(raw[:, 0] == peer).flatten() + owned
                                   for peer in range(size)]).to(device)

    def forward(self, features):
        if features.shape[0] != self.total or features.dtype != torch.float32:
            raise ValueError("MFF feature exchange requires FP32 [owned+ghost,features]")
        send = features.index_select(0, self.send_rows).contiguous()
        recv = features.new_empty((sum(self.recv_counts), features.shape[1]))
        dist.all_to_all_single(recv, send, output_split_sizes=self.recv_counts,
                              input_split_sizes=self.send_counts)
        out = torch.empty_like(features)
        out[:self.owned].copy_(features[:self.owned])
        out.index_copy_(0, self.recv_rows, recv)
        return out

    def reverse(self, gradient):
        send = gradient.index_select(0, self.recv_rows).contiguous()
        recv = gradient.new_empty((sum(self.send_counts), gradient.shape[1]))
        dist.all_to_all_single(recv, send, output_split_sizes=self.send_counts,
                              input_split_sizes=self.recv_counts)
        out = torch.zeros_like(gradient)
        out[:self.owned].copy_(gradient[:self.owned])
        out.index_add_(0, self.send_rows, recv)
        return out


class _Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, plan):
        ctx.plan = plan
        return plan.forward(features)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.plan.reverse(gradient), None


def _chunk(model, desired, device):
    # Bound the largest path-preserving message by actual TP width, so changing
    # channels, angular degree or allowed paths does not inherit a C64-only cap.
    widths = [int(module.channels) * sum(2 * path[2] + 1 for path in module.paths)
              for module in model.modules() if hasattr(module, "paths") and hasattr(module, "channels")]
    width = max(widths, default=1)
    budget = min(9 * 2**30, int(torch.cuda.get_device_properties(device).total_memory * .15))
    return max(256, min(desired, (budget // (4 * width) // 256) * 256))


def _load(checkpoint, metadata, device):
    from mace_ictc.interfaces.foundation_hp import FoundationHP
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    e0_keys, e0_values = metadata.get("e0_keys"), metadata.get("e0_values")
    if isinstance(saved, FoundationHP):
        model = saved.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if metadata.get("head"):
            model.select_head(metadata["head"])
        desired = 262144 if metadata["inference_mode"] == "performance" else 32768
        chunk = _chunk(model, desired, device)
        plan = model.configure_inference(chunk_edges=chunk, backward_edges=32768,
            radial_chunk=32768, product_basis="native")
        return model, None, plan
    if not isinstance(saved, dict) or saved.get("tensor_product_mode") != "pure-cartesian-ictd-fix":
        raise ValueError("Native MFF expects a pure-cartesian-ictd-fix checkpoint or converted FoundationHP model")
    from mace_ictc.checkpoints import load_checkpoint
    potential = load_checkpoint(str(checkpoint), metadata["elements"],
        device=str(device), operator_plan="checkpoint", torchscript=False,
        atomic_energy_keys=e0_keys, atomic_energy_values=e0_values,
        avg_num_neighbors=metadata.get("avg_num_neighbors"))
    model = potential.wrapper.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    from mace_ictc.models.pure_cartesian_ictd_fix import PureCartesianICTDFix
    if not isinstance(model, PureCartesianICTDFix):
        raise ValueError("Native MFF requires a PureCartesianICTDFix model")
    if (model.dispersion is not None
            or model.ictd_fix_interaction_attn_heads != 0
            or getattr(model, "physical_tensor_heads", None)
            or getattr(model, "external_tensor_rank", None) is not None
            or getattr(model, "num_fidelity_levels", 0)):
        raise ValueError("Native MFF supports scalar energy models without dispersion or extra input/output heads")
    if model.long_range_module is not None:
        from .mff_electrostatics import NativeReciprocal
        NativeReciprocal(model.long_range_module)
    for module in model.modules():
        if hasattr(module, "internal_compute_dtype"):
            module.internal_compute_dtype = torch.float32
    model.edge_compute_dtype = torch.float32
    model.preserve_edge_order = True
    model.skip_input_validation = True
    desired = 1048576 if metadata["inference_mode"] == "performance" else 32768
    chunk = _chunk(model, desired, device)
    from mace_ictc.hp import configure_model
    plan = configure_model(model, training=False, policy="generated", layout="fused_pack",
        message_schedule="roles", species_schedule="indexed", graph_reduction="csr",
        topology_backend="native", csr_accumulator_limit=48 if model.channels >= 256 else 24,
        csr_slice_search=True, geometry_schedule="streamed", message_chunk_edges=chunk,
        inference_message_layout="degree", gate_cache_bytes=0, inference_gate_adjoint_rows=16384,
        inference_gemm_policy="dtk2604", inference_message_adjoint_edges=32768,
        inference_basis_fusion=True, species_linear_schedule="grouped", electrostatics_backend="hip")
    if metadata["inference_mode"] == "capacity":
        # Frozen radial prefixes can recompute their SiLU activations during
        # coordinate backpropagation instead of retaining every edge/hidden
        # activation for every interaction. Keep this local to native MFF;
        # no feature or reciprocal collective is replayed by the checkpoint.
        for module in model.modules():
            if hasattr(module, "_radial_checkpoint"):
                module._radial_checkpoint = True
                module._radial_chunk_rows = 32768
        plan["radial_checkpoint"] = True
        plan["radial_chunk_rows"] = 32768
    if e0_keys is None:
        e0_keys = potential.wrapper.atomic_energy_keys.tolist()
        e0_values = potential.wrapper.atomic_energy_values.tolist()
    e0 = dict(zip(e0_keys, e0_values)) if metadata["embed_e0"] else {}
    return model, e0, dict(plan, chunk_edges=chunk)


class _Runtime:
    def __init__(self, checkpoint, metadata, device):
        self.model, e0, self.plan = _load(checkpoint, metadata, device)
        self.foundation = e0 is None
        from .mff_electrostatics import NativeReciprocal
        module = getattr(self.model, "long_range_module", None)
        self.reciprocal = NativeReciprocal(module) if module is not None else None
        if self.reciprocal is not None:
            self.plan["electrostatics"] = self.reciprocal.description
        numbers = self.model.native.atomic_numbers.tolist() if self.foundation else self.model.atomic_numbers
        self.numbers = torch.tensor(numbers, dtype=torch.int64, device=device)
        self.e0 = torch.tensor([0. if e0 is None else e0.get(z, 0.) for z in numbers],
                               dtype=torch.float32, device=device)
        self.metadata = metadata
        self.first = True
        self.first_exchange = True
        self.sorted_checked = False
        self.calls = 0
        self.exchanges = 0
        self.graph = {}
        self.capacity_reclaim = False

    def energy(self, pos, atomic, batch, src, dst, shifts, cell, vectors, owned, owners, rank, size):
        if not self.foundation and self.metadata["inference_mode"] == "capacity":
            total = torch.cuda.get_device_properties(pos.device).total_memory
            self.capacity_reclaim |= torch.cuda.max_memory_allocated(pos.device) > .75 * total
            if self.capacity_reclaim:
                # Neighbor counts change between steps. Reclaim completed
                # Python wrappers and idle allocator blocks before another
                # large force graph, rather than carrying incompatible cached
                # block sizes into its peak. Live tensors are unaffected.
                import gc
                torch.cuda.synchronize(pos.device)
                gc.collect()
                torch.cuda.empty_cache()
            self.plan["capacity_reclaim_active"] = self.capacity_reclaim
        if not self.sorted_checked:
            if not bool(torch.all(dst[1:] >= dst[:-1])):
                raise ValueError("Native MFF requires destination-sorted edges")
            match = atomic[:, None] == self.numbers[None, :]
            if not bool(match.any(-1).all()):
                raise ValueError("LAMMPS atom type is absent from the native MFF checkpoint")
            self.sorted_checked = True
        species = torch.searchsorted(self.numbers, atomic) if bool(torch.all(self.numbers[1:] > self.numbers[:-1])) else (
            (atomic[:, None] == self.numbers[None, :]).long().argmax(-1))
        plan = _Plan(owners, owned, pos.shape[0], rank, size, pos.device) if size > 1 else None
        self.calls += 1
        self.graph = dict(owned=owned, nodes=pos.shape[0], edges=src.numel(), ghosts=pos.shape[0]-owned)

        communication_anchor = pos.sum() * 0.

        def exchange(layer, state):
            nonlocal communication_anchor
            if not self.foundation and layer == self.model.num_interaction - 1:
                return state
            if self.first_exchange:
                torch.cuda.empty_cache()
                self.first_exchange = False
            self.exchanges += 1
            self.graph["feature_columns"] = state.shape[1]
            # A disconnected local graph must still visit every reverse
            # collective. Connect a constant state to coordinates if needed,
            # and retain a zero cotangent path from the owned energy below.
            state = state + communication_anchor
            exchanged = _Exchange.apply(state.contiguous(), plan)
            communication_anchor = exchanged.sum() * 0.
            return exchanged

        if self.foundation:
            value = self.model.forward_vectors(vectors, species, src, dst,
                owned=owned, exchange=exchange if plan is not None else None)
            result = value["node_energy"] if self.metadata["embed_e0"] else value["interaction_energy"]
            if owned != pos.shape[0]:
                result = torch.cat((result, result.new_zeros(pos.shape[0] - owned)))
        else:
            options = {} if plan is None else dict(distributed_atom_count=owned,
                layer_state_callback=exchange, compact_local_projection=True, compact_local_messages=True)
            if self.reciprocal is not None:
                def reciprocal(position, graph, box, q, dipole, quadrupole):
                    # Order reverse spectral reductions before reverse feature
                    # exchange even on domains whose learned sources are constant.
                    return self.reciprocal(position, graph, box, q + communication_anchor,
                                           dipole, quadrupole, owned=owned, size=size)
                options["long_range_energy_callback"] = reciprocal
            value = self.model(pos, atomic, batch, src, dst, shifts, cell,
                               precomputed_edge_vec=vectors, **options)
            result = (value[0] if isinstance(value, tuple) else value).reshape(-1) + self.e0[species]
        if self.first:
            torch.cuda.empty_cache()
            self.first = False
        # Explicit zero dependency also handles a valid graph with no edges.
        # The zero chain fixes reverse-collective order even if an isolated
        # domain's physical energy does not depend on a communicated state.
        return result + communication_anchor


def register_core(path: str, device: str) -> None:
    """Load a native MFF artifact on the LAMMPS-selected K100 device."""
    target = torch.device(device)
    if target.type != "cuda" or not torch.version.hip:
        raise ValueError("Native MFF requires K100 with a HIP-enabled PyTorch build")
    torch.cuda.set_device(target)
    metadata = json.loads(Path(path + ".json").read_text())
    key = (int(metadata["model_id"]), torch.cuda.current_device())
    if key not in _RUNTIMES:
        checkpoint = Path(path).resolve().parent / metadata["checkpoint"]
        _RUNTIMES[key] = _Runtime(checkpoint, metadata, target)


def _energy(pos, atomic, batch, src, dst, shifts, cell, vectors, model_id, owned, owners, rank, size):
    runtime = _RUNTIMES[(model_id, pos.device.index)]
    with torch.enable_grad():
        return runtime.energy(pos, atomic, batch, src, dst, shifts, cell, vectors, owned, owners, rank, size)


def _release(anchor, model_id):
    if torch.cuda.memory_reserved(anchor.device) > .85 * torch.cuda.get_device_properties(anchor.device).total_memory:
        torch.cuda.empty_cache()
    return anchor


_LIBRARY = torch.library.Library("mff_ictc_hp", "DEF")
_LIBRARY.define("energy(Tensor pos, Tensor atomic, Tensor batch, Tensor src, Tensor dst, Tensor shifts, Tensor cell, Tensor vectors, int model_id, int owned, Tensor owners, int rank, int size) -> Tensor")
_LIBRARY.impl("energy", _energy, "CompositeImplicitAutograd")
_LIBRARY.define("release(Tensor anchor, int model_id) -> Tensor")
_LIBRARY.impl("release", _release, "CompositeImplicitAutograd")


class NativeCore(torch.nn.Module):
    __constants__ = ["model_id"]

    def __init__(self, model_id: int):
        super().__init__()
        self.model_id = model_id

    def forward(self, pos, atomic, batch, src, dst, shifts, cell, vectors):
        return torch.ops.mff_ictc_hp.energy(pos, atomic, batch, src, dst, shifts, cell,
            vectors, self.model_id, pos.shape[0], atomic.new_empty((0, 2)), 0, 1)

    @torch.jit.export
    def forward_distributed(self, pos, atomic, batch, src, dst, shifts, cell, vectors,
                            owned: int, owners: torch.Tensor, rank: int, size: int):
        return torch.ops.mff_ictc_hp.energy(pos, atomic, batch, src, dst, shifts, cell,
            vectors, self.model_id, owned, owners, rank, size)

    @torch.jit.export
    def warmup_on_real_graph(self) -> bool:
        return True

    @torch.jit.export
    def after_force(self, anchor: torch.Tensor):
        return torch.ops.mff_ictc_hp.release(anchor, self.model_id)


def export_native(*, checkpoint, elements, out_pt, inference_mode="performance", head=None,
                  embed_e0=True, e0_csv=None, avg_num_neighbors=None):
    """Write a relocatable native core, checkpoint, and metadata sidecars."""
    from ase.data import atomic_numbers
    from mace_ictc.interfaces.foundation_hp import FoundationHP
    if inference_mode not in ("performance", "capacity"):
        raise ValueError("inference_mode must be performance or capacity")
    if not torch.cuda.is_available() or not torch.version.hip:
        raise ValueError("Export native MFF on the target K100 with HIP PyTorch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(saved, FoundationHP):
        if head is not None:
            saved.select_head(head)
        numbers = saved.native.atomic_numbers.tolist()
        depth, cutoff = saved.num_interaction, saved.max_radius
        mode = "foundation-hp"
        head = saved.head
        if e0_csv:
            raise ValueError("FoundationHP retains its own per-head E0; --e0-csv is unsupported")
    elif isinstance(saved, dict) and saved.get("tensor_product_mode") == "pure-cartesian-ictd-fix":
        arch = saved.get("model_hyperparameters", {})
        numbers = saved.get("atomic_numbers", arch.get("atomic_numbers", saved.get("atomic_energy_keys")))
        if torch.is_tensor(numbers):
            numbers = numbers.tolist()
        if numbers is None:
            raise ValueError("Native MFF requires checkpoint atomic_numbers metadata")
        depth = int(saved.get("num_interaction", arch.get("num_interaction", 2)))
        cutoff = float(saved.get("max_radius", arch.get("max_radius", 5.)))
        mode = "pure-cartesian-ictd-fix"
        if head is not None:
            raise ValueError("--head applies to FoundationHP checkpoints")
    else:
        raise ValueError("Use mff-convert-mace or mff-convert-foundation before native MFF export")
    if not elements or len(set(elements)) != len(elements) or any(atomic_numbers.get(s) not in numbers for s in elements):
        raise ValueError("--elements must list distinct element symbols present in the checkpoint")
    target = Path(out_pt).resolve()
    if target.suffix == ".pt2":
        raise ValueError("K100 native MFF uses .pt, not .pt2")
    target.parent.mkdir(parents=True, exist_ok=True)
    copied = target.with_name(target.name + ".checkpoint.pth")
    if Path(checkpoint).resolve() != copied:
        shutil.copy2(checkpoint, copied)
    metadata = dict(backend="ictc-hp", model_id=uuid.uuid4().int & ((1 << 63) - 1),
        checkpoint=copied.name, elements=list(elements), atomic_numbers=list(numbers),
        message_passing_depth=depth, max_radius=cutoff, tensor_product_mode=mode,
        dtype="float32", inference_mode=inference_mode, head=head,
        feature_exchange="rccl", embed_e0=bool(embed_e0), avg_num_neighbors=avg_num_neighbors)
    if e0_csv:
        from mace_ictc.utils.config import ModelConfig
        cfg = ModelConfig(dtype=torch.float32)
        cfg.load_atomic_energies_from_file(e0_csv)
        metadata.update(e0_keys=cfg.atomic_energy_keys.tolist(), e0_values=cfg.atomic_energy_values.tolist())
    # Validate reconstruction and native operator coverage before publishing the archive.
    checked = _Runtime(copied, metadata, torch.device("cuda", torch.cuda.current_device()))
    if checked.reciprocal is not None:
        metadata["electrostatics"] = checked.reciprocal.description
    torch.jit.save(torch.jit.script(NativeCore(metadata["model_id"])), str(target))
    Path(str(target) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    Path(str(target) + ".meta").write_text(
        f"backend ictc-hp\nmessage_passing_depth {depth}\nfeature_exchange rccl\n")
    print(f"Wrote native K100 MFF core: {target} ({inference_mode}, {depth} interactions)")
    return metadata
