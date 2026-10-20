#!/usr/bin/env python3
"""Official AOTInductor (.pt2) export CLI for FSCETP LAMMPS inference -- ``mff-export-aoti``.

Exports a trained checkpoint to an AOTInductor ``.pt2`` package that the
``pair_style mff/torch`` engine loads directly: the force is traced INTO the
graph, so there is no C++ autograd at runtime (the .pt2 returns ``(energy,
force)`` from a single ``.run()``). N-dynamic by DEFAULT -- ONE ``.pt2`` serves
any atom count -- via ``torch.export`` with the atom axis as a ``Dim``
(``--static-n`` opts out to the legacy baked-N + padding path).

Pipeline: make_fx flattens forward + 1st-order force-autograd (reusing
``makefx_compile.trace_and_compile_force(do_compile=False)``) -> ``torch.export.export``
(N and E dynamic) -> ``aoti_compile_and_package`` -> load back and verify numerics +
rotation-equivariance + VARY-N, then time vs eager. Writes ``<out>`` plus a
``<out>.meta`` sidecar (``dynamic 1 / nmax 0``) that the LAMMPS engine reads.

Usage:
  # export a trained checkpoint (N-dynamic, absolute-energy .pt2):
  mff-export-aoti --checkpoint model.pth --elements H,O,F,K \
      --atoms 400 --device cuda --embed-e0 --out core.pt2
  # (equivalently: python -m mace_ictc.cli.export_aoti_core ...)
  # then in LAMMPS:  pair_coeff * * core.pt2 H O F K   (.pt2 -> AOTI auto-detected)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from mace_ictc.training.makefx_compile import trace_and_compile_force

# Default species (H, C, N, O) for the synthetic no-checkpoint path only; a real --checkpoint export
# overrides this with the checkpoint's own atomic_numbers.
SPECIES = (1, 6, 7, 8)


def _long_range_deploy_metadata(
    model,
    *,
    export_reciprocal_source: bool,
    use_explicit_dispersion_edges: bool,
) -> dict:
    """Metadata consumed by mff/torch for reciprocal and MBD deployment paths."""
    from mace_ictc.models.dispersion import (
        dispersion_deployment_graph_rule,
        dispersion_train_deploy_graph_compatibility,
        dispersion_training_graph_rule,
    )

    lrm = getattr(model, "long_range_module", None)
    es_attr = getattr(lrm, "energy_scale", None) if lrm is not None else None
    s_chan = int(getattr(getattr(model, "multipole_readout", None), "source_channels", 1) or 1)
    dispersion_mode = str(getattr(model, "long_range_dispersion_mode", "none"))
    mbd_operator_backend = str(getattr(model, "mbd_operator_backend", "edge_sparse"))
    mbd_source_enabled = bool(getattr(model, "long_range_mbd_source_enabled", False))
    _disp = getattr(model, "dispersion", None)
    _mbd_beta = float(_disp.mbd_beta()) if (mbd_source_enabled and _disp is not None) else 1.0
    _mbd_cs = float(_disp.mbd_coupling_scale()) if (mbd_source_enabled and _disp is not None) else 1.0
    _mbd_es = float(_disp.mbd_energy_scale()) if (mbd_source_enabled and _disp is not None) else 1.0
    _disp_min_cutoff = float(getattr(model, "dispersion_min_cutoff", getattr(_disp, "min_cutoff", 0.0)))
    _disp_switch_width = float(getattr(model, "dispersion_switch_width", getattr(_disp, "switch_width", 0.5)))
    return {
        "export_reciprocal_source": bool(export_reciprocal_source),
        "reciprocal_source_channels": s_chan,
        "reciprocal_source_boundary": "periodic",
        "reciprocal_source_slab_padding_factor": 2,
        "long_range_runtime_backend": str(getattr(model, "long_range_runtime_backend", "mesh_fft")),
        "long_range_mesh_size": int(getattr(model, "long_range_mesh_size", 16)),
        "long_range_max_multipole_l": int(getattr(model, "long_range_max_multipole_l", 0)),
        "long_range_source_kind": str(getattr(model, "long_range_runtime_source_kind", "latent_multipole")),
        "long_range_source_channels": s_chan,
        "long_range_source_layout": str(
            getattr(model, "long_range_runtime_source_layout", "packed_q_dipole_quad")
        ),
        "long_range_boundary": str(getattr(model, "long_range_boundary", "periodic")),
        "long_range_energy_partition": str(getattr(model, "long_range_energy_partition", "uniform")),
        "long_range_neutralize": bool(getattr(model, "long_range_neutralize", True)),
        "long_range_green_mode": str(getattr(model, "long_range_green_mode", "poisson")),
        # Mesh assignment (cic/pcs) the multipole reciprocal solver must match the in-model PME; the
        # C++ cuFFT solver reads this and dispatches the matching spread/deconvolution stencil.
        "long_range_assignment": str(getattr(model, "long_range_assignment", "pcs")),
        "long_range_mesh_fft_full_ewald": bool(getattr(model, "long_range_mesh_fft_full_ewald", False)),
        "long_range_dispersion_mode": str(getattr(model, "long_range_dispersion_mode", "none")),
        "long_range_dispersion": bool(getattr(model, "long_range_dispersion", False)),
        "dispersion_cutoff": float(getattr(model, "dispersion_cutoff", 0.0)),
        "dispersion_min_cutoff": _disp_min_cutoff,
        "dispersion_switch_width": _disp_switch_width,
        "dispersion_max_num_neighbors": getattr(model, "dispersion_max_num_neighbors", None),
        "dispersion_neighbor_method": str(getattr(model, "dispersion_neighbor_method", "auto")),
        "dispersion_bruteforce_threshold": int(getattr(model, "dispersion_bruteforce_threshold", 1024)),
        "dispersion_allow_large_bruteforce_fallback": bool(
            getattr(model, "dispersion_allow_large_bruteforce_fallback", False)
        ),
        "aoti_dispersion_edges": bool(use_explicit_dispersion_edges),
        "dispersion_training_graph_rule": dispersion_training_graph_rule(
            dispersion_mode,
            mbd_operator_backend=mbd_operator_backend,
        ),
        "dispersion_deployment_graph_rule": dispersion_deployment_graph_rule(
            dispersion_mode,
            mbd_operator_backend=mbd_operator_backend,
        ),
        "dispersion_train_deploy_graph_compatibility": dispersion_train_deploy_graph_compatibility(
            dispersion_mode,
            mbd_operator_backend=mbd_operator_backend,
        ),
        "dispersion_slq_num_probes": int(getattr(model, "dispersion_slq_num_probes", 8)),
        "dispersion_slq_lanczos_steps": int(getattr(model, "dispersion_slq_lanczos_steps", 16)),
        "mbd_operator_backend": mbd_operator_backend,
        "mbd_pme_mesh_size": int(getattr(model, "mbd_pme_mesh_size", 16)),
        "mbd_pme_assignment": str(getattr(model, "mbd_pme_assignment", "pcs")),
        "mbd_pme_k_norm_floor": float(getattr(model, "mbd_pme_k_norm_floor", 1.0e-6)),
        "mbd_pme_assignment_window_floor": float(getattr(model, "mbd_pme_assignment_window_floor", 1.0e-6)),
        "mbd_pme_ewald_alpha_prefactor": float(getattr(model, "mbd_pme_ewald_alpha_prefactor", 5.0)),
        "long_range_ewald_alpha_prefactor": float(
            getattr(getattr(lrm, "kernel", None), "ewald_alpha_prefactor", 5.0)
        ),
        "long_range_energy_scale": (float(es_attr.detach().cpu().item()) if es_attr is not None else 1.0),
        "long_range_theta": 0.5,
        "long_range_leaf_size": 32,
        "long_range_multipole_order": 0,
        "long_range_screening": 0.0,
        "long_range_softening": 1.0e-6,
        # MBD source packing: the deploy source is [electrostatic | omega, alpha]; the C++ MBD solver
        # reads source[:, offset:offset+channels] and uses these learned damping params.
        "long_range_mbd_source_enabled": mbd_source_enabled,
        "long_range_mbd_source_offset": int(getattr(model, "long_range_mbd_source_offset", 0)),
        "long_range_mbd_source_channels": int(getattr(model, "long_range_mbd_source_channels", 2)),
        "long_range_mbd_beta": _mbd_beta,
        "long_range_mbd_coupling_scale": _mbd_cs,
        "long_range_mbd_energy_scale": _mbd_es,
    }


def make_fixed_graph(*, num_nodes, avg_degree, dtype, device, seed: int = 42):
    """Deterministic single-molecule example graph (no self-loops) for tracing + numerics/equivariance
    checks. Big non-periodic box (edge_shifts=0); the real LAMMPS graph is supplied at runtime, this is
    only the torch.export trace/validation sample."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.randn(num_nodes, 3, generator=g, dtype=torch.float64) * 2.5
    species = torch.tensor(SPECIES, dtype=torch.long)
    A = species[torch.randint(0, len(SPECIES), (num_nodes,), generator=g)]
    num_edges = num_nodes * avg_degree
    edge_src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    loop = edge_src == edge_dst
    edge_dst[loop] = (edge_dst[loop] + 1) % num_nodes  # break self-loops deterministically
    edge_shifts = torch.zeros(num_edges, 3, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 100.0  # big box, no PBC wrap
    batch = torch.zeros(num_nodes, dtype=torch.long)
    return (pos.to(device=device, dtype=dtype), A.to(device=device), batch.to(device=device),
            edge_src.to(device=device), edge_dst.to(device=device),
            edge_shifts.to(device=device, dtype=dtype), cell.to(device=device, dtype=dtype))


class _E0Wrap(torch.nn.Module):
    """Add E0(Z) to the bare model's per-atom energies so the exported .pt2 returns ABSOLUTE energies
    (a true drop-in for an E0-embedded TorchScript core). E0 is a per-atom constant -> forces are
    unchanged (its gradient w.r.t. positions is zero)."""

    def __init__(self, model, e0_lut):
        super().__init__()
        self.model = model
        self.register_buffer("e0_lut", e0_lut)

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
        dispersion_edge_src=None,
        dispersion_edge_dst=None,
        dispersion_edge_shifts=None,
        return_reciprocal_source: bool = False,
    ):
        if return_reciprocal_source:
            out = self.model(
                pos,
                A,
                batch,
                edge_src,
                edge_dst,
                edge_shifts,
                cell,
                dispersion_edge_src=dispersion_edge_src,
                dispersion_edge_dst=dispersion_edge_dst,
                dispersion_edge_shifts=dispersion_edge_shifts,
                return_reciprocal_source=True,
            )
            e_atom, rs = out[0], out[1]
            e0 = self.e0_lut[A].to(e_atom.dtype).reshape(e_atom.shape)
            return e_atom + e0, rs
        out = self.model(
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            dispersion_edge_src=dispersion_edge_src,
            dispersion_edge_dst=dispersion_edge_dst,
            dispersion_edge_shifts=dispersion_edge_shifts,
        )
        e_atom = out[0] if isinstance(out, tuple) else out
        e0 = self.e0_lut[A].to(e_atom.dtype).reshape(e_atom.shape)
        return e_atom + e0


def force_compute_fn_factory(
    model,
    *,
    training: bool,
    emit_reciprocal_source: bool = False,
    use_dispersion_edges: bool = False,
    owned_energy_mask: bool = False,
):
    """compute_fn returning (E_atom, force=-dE/dpos[, reciprocal_source]) -- force traced into the
    graph. With emit_reciprocal_source the model defers the reciprocal energy and emits the packed
    [q|mu|Q] source as a 3rd output (the C++ reciprocal solver does the sum); the force is then the
    short-range force only (the C++ adds the reciprocal force)."""
    def _finish(p, out, owned=None):
        if emit_reciprocal_source:
            e_atom, rs = out[0], out[1]
        else:
            e_atom = out[0] if isinstance(out, tuple) else out
        total=e_atom.sum() if owned is None else (e_atom.reshape(-1)*owned).sum()
        grad = torch.autograd.grad(total, p, create_graph=training)[0]
        if emit_reciprocal_source:
            return e_atom, -grad, rs
        return e_atom, -grad

    if owned_energy_mask:
        def compute_owned(pos,A,batch,edge_src,edge_dst,edge_shifts,cell,*extra):
            p=pos.detach().requires_grad_(True)
            if use_dispersion_edges:
                ds,dd,dsh,owned=extra
                out=model(p,A,batch,edge_src,edge_dst,edge_shifts,cell,dispersion_edge_src=ds,
                    dispersion_edge_dst=dd,dispersion_edge_shifts=dsh,return_reciprocal_source=emit_reciprocal_source)
            else:
                owned,=extra
                out=model(p,A,batch,edge_src,edge_dst,edge_shifts,cell,return_reciprocal_source=emit_reciprocal_source)
            return _finish(p,out,owned)
        return compute_owned

    if use_dispersion_edges:
        def compute_fn(
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            dispersion_edge_src,
            dispersion_edge_dst,
            dispersion_edge_shifts,
        ):
            p = pos.detach().requires_grad_(True)
            out = model(
                p,
                A,
                batch,
                edge_src,
                edge_dst,
                edge_shifts,
                cell,
                dispersion_edge_src=dispersion_edge_src,
                dispersion_edge_dst=dispersion_edge_dst,
                dispersion_edge_shifts=dispersion_edge_shifts,
                return_reciprocal_source=emit_reciprocal_source,
            )
            return _finish(p, out)

        return compute_fn

    def compute_fn(pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        p = pos.detach().requires_grad_(True)
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell, return_reciprocal_source=emit_reciprocal_source)
        return _finish(p, out)
    return compute_fn


def _ef(out):
    """First two outputs (energy, force) from a 2- or 3-output (multipole reciprocal_source) result."""
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    return out, None


def _inner_mace_contraction(module):
    inner = getattr(module, "symmetric_contractions", module)
    if hasattr(inner, "symmetric_contractions") and not hasattr(inner, "contractions"):
        inner = inner.symmetric_contractions
    return inner


def _model_uses_cueq_product(model) -> bool:
    return any(type(module).__name__.startswith("Cueq") for module in model.modules())


def _model_supports_e3nn_basis_fold(model) -> bool:
    return any(
        module is not model and callable(getattr(module, "enable_e3nn_basis", None))
        for module in model.modules()
    )


def _configure_angular_basis_for_export(model, requested: str) -> str:
    """Apply the requested export angular basis without asking users to know backend limits.

    ``angular_basis='e3nn'`` is only a real speed/graph optimization when the product
    backend can fold its symmetric contraction into the e3nn basis.  Bridge-U is
    already a bridge backend and intentionally has no e3nn fold hook, so keep the
    model in ICTC basis instead of failing later during the first forward/export.
    """
    model=getattr(model,"model",model)  # E0 is a wrapper around the angular model.
    if not hasattr(model, "angular_basis"):
        if requested != "checkpoint":
            raise TypeError("--angular-basis requires a PureCartesianICTDFix-style model with angular_basis")
        return "unavailable"

    current = str(getattr(model, "angular_basis"))
    target = current if requested == "checkpoint" else str(requested)
    if target not in {"ictd", "e3nn"}:
        raise ValueError(f"angular_basis must be 'ictd' or 'e3nn', got {target!r}")

    if target == "e3nn" and not _model_supports_e3nn_basis_fold(model):
        backend = getattr(model, "ictd_fix_product_backend", "?")
        print(
            "[aoti] WARNING: angular_basis='e3nn' requested but the selected product "
            f"backend {backend!r} does not expose an e3nn fold (bridge-U has no e3nn-fold path). "
            "Keeping angular_basis='ictd'. Use --cueq-product for the e3nn-folded product path."
        )
        target = "ictd"

    if current != target:
        model.angular_basis = target
        if hasattr(model, "_e3nn_folded"):
            model._e3nn_folded = False
    return target


def _torch_export_with_strict_fallback(gm, example_inputs, *, dynamic_shapes, prefer_strict: bool):
    """Export with strict first when possible, then non-strict with mandatory downstream checks.

    ``strict=False`` is a fallback for exporter limitations (for example symbolic
    integer returns in optimized graphs).  It is not treated as proof of correctness:
    the caller still compiles, loads, and numerically compares the produced .pt2.
    """
    import traceback

    first_trace = None
    attempts = [bool(prefer_strict)]
    if prefer_strict:
        attempts.append(False)

    for strict in attempts:
        try:
            exported = torch.export.export(
                gm,
                tuple(example_inputs),
                dynamic_shapes=dynamic_shapes,
                strict=bool(strict),
            )
            if prefer_strict and not strict:
                print(
                    "[aoti] torch.export.export OK after strict=False fallback "
                    "(dynamic numerics will still be checked)"
                )
            return exported, bool(strict)
        except Exception as ex:
            if strict and prefer_strict:
                first_trace = traceback.format_exc()
                print(f"[aoti] torch.export strict=True FAILED: {type(ex).__name__}: {ex}")
                print("[aoti] retrying torch.export with strict=False")
                continue
            if first_trace is not None:
                print("[aoti] original strict=True traceback:")
                print(first_trace.rstrip())
            traceback.print_exc()
            raise

    raise RuntimeError("unreachable torch.export strict fallback state")


def _replace_products_with_cueq(model, *, device: torch.device) -> None:
    from mace_ictc.models.pure_cartesian_ictd_fix import CueqMACEProductBasisBlockSO3

    if not hasattr(model, "products"):
        raise TypeError("--cueq-product expects a model with a .products ModuleList")
    new_products = torch.nn.ModuleList()
    for product in model.products:
        if type(product).__name__.startswith("Cueq"):
            if hasattr(product, "refresh_cueq_weights"):
                product.refresh_cueq_weights()
            new_products.append(product)
            continue
        try:
            ref = next(product.parameters())
        except StopIteration:
            ref = next(model.parameters())
        fast = CueqMACEProductBasisBlockSO3(
            num_elements=int(model.num_elements),
            channels=int(product.channels),
            lmax=int(product.lmax),
            target_lmax=int(product.target_lmax),
            correlation=int(_inner_mace_contraction(product.symmetric_contractions).contractions[0].correlation),
            use_reduced_cg=bool(getattr(product, "use_reduced_cg", False)),
        ).to(device=device, dtype=ref.dtype)
        fast.linear.load_state_dict(product.linear.state_dict())
        fast_inner = fast.symmetric_contractions.symmetric_contractions
        old_inner = _inner_mace_contraction(product.symmetric_contractions)
        _copy_contraction_learnable_weights_only(old_inner, fast_inner)
        fast.refresh_cueq_weights()
        new_products.append(fast.eval())
    model.products = new_products
    if hasattr(model, "ictd_fix_product_backend"):
        model.ictd_fix_product_backend = "cueq"
    if hasattr(model, "ictd_fix_effective_product_backends"):
        model.ictd_fix_effective_product_backends = ["cueq"] * len(model.products)


def _copy_contraction_learnable_weights_only(src_sc, dst_sc) -> None:
    """Copy MACE symmetric-contraction learnable weights without copying fixed U buffers.

    Bridge-U folds the ICTC<->e3nn basis change into its inner ``U_matrix_*`` buffers
    at construction time.  cuEq product contractions, however, must keep their own
    e3nn/O3 U tensors.  A full ``load_state_dict`` would silently copy bridge-U's
    folded U buffers into the cuEq backend.  Only the learned per-element weights
    are model parameters and should move across backends.
    """
    src_contractions = src_sc.contractions
    dst_contractions = dst_sc.contractions
    if len(src_contractions) != len(dst_contractions):
        raise ValueError(
            f"Cannot copy contraction weights: source has {len(src_contractions)} contractions, "
            f"destination has {len(dst_contractions)}"
        )
    with torch.no_grad():
        for src, dst in zip(src_contractions, dst_contractions):
            dst.weights_max.copy_(src.weights_max.to(dtype=dst.weights_max.dtype, device=dst.weights_max.device))
            if len(src.weights) != len(dst.weights):
                raise ValueError(
                    f"Cannot copy contraction weights: source has {len(src.weights)} lower-order weights, "
                    f"destination has {len(dst.weights)}"
                )
            for src_w, dst_w in zip(src.weights, dst.weights):
                dst_w.copy_(src_w.to(dtype=dst_w.dtype, device=dst_w.device))


def _enable_fused_selector_message_linears(model) -> int:
    enabled = 0
    for module in model.modules():
        fn = getattr(module, "enable_eval_fused_selector_message", None)
        if callable(fn) and fn():
            enabled += 1
    return enabled


def _set_parameter(module: torch.nn.Module, name: str, value: torch.Tensor, *, like: torch.nn.Parameter) -> None:
    param = torch.nn.Parameter(value.detach().clone().contiguous(), requires_grad=like.requires_grad)
    if isinstance(module, torch.nn.ParameterDict):
        module[name] = param
    else:
        setattr(module, name, param)


def _prune_model_elements(model: torch.nn.Module, selected_z: list[int]) -> list[int]:
    """Restrict element-conditioned weights to the exported element set.

    Pretrained MACE checkpoints can contain many species while a LAMMPS export may
    target a smaller subset.  This preserves exact numerics for ``selected_z`` and
    reduces element-conditioned tensors before tracing/AOTI compilation.
    """
    old_z = [int(z) for z in getattr(model, "atomic_numbers")]
    selected_z = [int(z) for z in selected_z]
    old_index = {z: i for i, z in enumerate(old_z)}
    missing = [z for z in selected_z if z not in old_index]
    if missing:
        raise ValueError(f"Cannot prune to elements absent from checkpoint atomic_numbers: {missing}")
    keep = torch.tensor([old_index[z] for z in selected_z], dtype=torch.long)

    ref_param = next(model.parameters())
    keep_dev = keep.to(device=ref_param.device)

    if hasattr(model, "node_embedding"):
        old_embedding = model.node_embedding
        new_embedding = torch.nn.Linear(len(selected_z), old_embedding.out_features, bias=False).to(
            device=old_embedding.weight.device,
            dtype=old_embedding.weight.dtype,
        )
        with torch.no_grad():
            new_embedding.weight.copy_(old_embedding.weight.index_select(1, keep_dev))
        model.node_embedding = new_embedding

    for module in model.modules():
        if type(module).__name__ == "ElementConditionedLinearSO3":
            module.num_elements = len(selected_z)
            for key, param in list(module.weights.items()):
                _set_parameter(module.weights, key, param.index_select(0, keep_dev), like=param)
            bias = getattr(module, "bias", None)
            if bias is not None:
                for key, param in list(bias.items()):
                    _set_parameter(bias, key, param.index_select(0, keep_dev), like=param)
        if type(module).__name__ == "MaceSymmetricContraction":
            for contraction in module.contractions:
                weights_max = getattr(contraction, "weights_max", None)
                if isinstance(weights_max, torch.nn.Parameter):
                    _set_parameter(contraction, "weights_max", weights_max.index_select(0, keep_dev), like=weights_max)
                weights = getattr(contraction, "weights", None)
                if isinstance(weights, torch.nn.ParameterList):
                    for idx, param in enumerate(list(weights)):
                        weights[idx] = torch.nn.Parameter(
                            param.index_select(0, keep_dev).detach().clone().contiguous(),
                            requires_grad=param.requires_grad,
                        )
                if getattr(contraction, "_use_scalar_corr3_fast", False):
                    contraction.refresh_scalar_corr3_fast_buffers()

    sc0 = getattr(model, "mace_first_layer_sc0", None)
    if sc0 is not None:
        model.mace_first_layer_sc0 = sc0.index_select(0, keep_dev).detach().clone().contiguous()

    map_size = max(int(getattr(model, "atomic_number_to_index").numel()), max(selected_z) + 1)
    mapping = torch.full((map_size,), -1, dtype=torch.long, device=ref_param.device)
    for new_idx, z in enumerate(selected_z):
        mapping[int(z)] = int(new_idx)
    model.atomic_numbers = tuple(selected_z)
    model.num_elements = len(selected_z)
    model.atomic_number_to_index = mapping
    return selected_z


def _aoti_compile(exported, out_path):
    """torch 2.x AOTInductor compile+package; tolerate API moves across versions."""
    try:
        from torch._inductor import aoti_compile_and_package
        return aoti_compile_and_package(exported, package_path=out_path)
    except TypeError:
        # older signature took the package path positionally / via options
        from torch._inductor import aoti_compile_and_package
        return aoti_compile_and_package(exported, out_path)


def _aoti_load(path, device):
    """Load an AOTI .pt2 back into a callable; tolerate API moves."""
    last = None
    for loader in ("aoti_load_package",):
        try:
            from torch._inductor import aoti_load_package  # type: ignore
            return aoti_load_package(path)
        except Exception as e:  # pragma: no cover
            last = e
    # fallback: torch._export.aot_load (older) returns a runner callable
    try:
        from torch._export import aot_load  # type: ignore
        return aot_load(path, device)
    except Exception as e:
        raise RuntimeError(f"no working AOTI loader (last: {last} / {e})")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _random_rotation(dtype, device, seed=12345):
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(m)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(device=device, dtype=dtype)


def _time(fn, device, iters, warmup):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1e3 / iters


def _synthetic_long_range_extra_from_args(args: argparse.Namespace) -> dict:
    """Build long-range kwargs for the synthetic benchmark/export path."""
    extra = {}
    if args.long_range_mode_arg:
        extra.update(
            long_range_mode=args.long_range_mode_arg,
            long_range_reciprocal_backend=args.long_range_reciprocal_backend_arg,
            long_range_boundary="periodic",
            long_range_mesh_size=args.lr_mesh_size_arg,
            long_range_kmax=args.long_range_kmax_arg,
            long_range_assignment=args.long_range_assignment_arg,
            long_range_mesh_fft_full_ewald=args.long_range_full_ewald_arg,
            long_range_max_multipole_l=args.long_range_multipole_l_arg,
        )
    if args.dispersion_mode_arg:
        extra.update(
            long_range_dispersion_mode=args.dispersion_mode_arg,
            dispersion_cutoff=args.dispersion_cutoff_arg,
            mbd_pme_mesh_size=args.lr_mesh_size_arg,
        )
        if args.mbd_operator_backend_arg:
            extra.update(mbd_operator_backend=args.mbd_operator_backend_arg)
        if getattr(args, "mbd_anisotropic_arg", False):
            extra.update(mbd_anisotropic_polarizability=True)
    return extra


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", default="baseline", choices=["baseline"])
    p.add_argument("--atoms", type=int, default=256)
    p.add_argument("--degree", type=int, default=24)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--num-interaction", type=int, default=2)
    p.add_argument("--attn-heads", dest="attn_heads", type=int, default=1)
    p.add_argument("--contraction-order", dest="contraction_order", type=int, default=2)
    p.add_argument("--product-backend", default="ictd-bridge-u")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--allow-tf32", action="store_true",
                   help="deprecated: TF32 is not allowed; passing this flag raises an error.")
    p.add_argument("--checkpoint", required=True,
                   help="path to a trained .pth checkpoint to export (via LAMMPS_MLIAP_MFF.from_checkpoint). "
                        "When set, the REAL trained weights/dtype/species are used instead of a random model, "
                        "so the produced .pt2 is the one LAMMPS will actually run.")
    p.add_argument("--operator-plan", choices=("auto", "checkpoint", "generated"), default="auto",
                   help="Default auto enables qualified independent CUDA FP32, positive channels, L<=5, correlation 1/2/3 generated "
                        "operators. checkpoint preserves the original route; generated requires coverage.")
    p.add_argument("--reference-buffer-policy", choices=("auto", "device"), default="auto",
                   help="auto keeps canonical high-L/cor3 U on CPU after factorization; device retains conventional "
                        "device buffers during preparation. Only used factorized constants are exported.")
    p.add_argument("--owned-energy-mask",action="store_true",
                   help="Export an extra per-node energy ownership input for exact MPI local-energy forces; generated mode enables this automatically.")
    p.add_argument("--generated-layout", choices=("packed", "fused_pack"), default="packed",
                   help="Generated HP layout packaged into the export; changing it requires a new package.")
    p.add_argument("--elements", default="H,C,N,O",
                   help="comma-separated element symbols for the checkpoint (from_checkpoint requires element_types; "
                        "the validation graph's species fall back to these if the model can't report its own)")
    p.add_argument("--prune-to-elements", action="store_true",
                   help="for checkpoint export, prune element-conditioned model weights to --elements before "
                        "tracing. This preserves numerics for those elements and can reduce AOTI element-index "
                        "work, but the exported model will reject other atomic numbers.")
    p.add_argument("--angular-basis", choices=("checkpoint", "ictd", "e3nn"), default="checkpoint",
                   help="override the checkpoint model's angular_basis before tracing. The default keeps the "
                        "checkpoint setting. 'e3nn' lets compatible product backends (notably --cueq-product) "
                        "consume e3nn-basis features directly and can remove ICTC<->e3nn bridge work.")
    p.add_argument("--avg-num-neighbors", dest="avg_num_neighbors", type=float, default=None,
                   help="message-normalization constant the weights were trained under (model divides messages "
                        "by it). For ictd-fix it is auto-computed from the training data and is NOT saved in the "
                        "checkpoint, so pass the TRAINING value (logged as 'Computed average number of neighbors') "
                        "or the deployed energies/forces are wrong (from_checkpoint else falls back to 14.38).")
    p.add_argument("--embed-e0", dest="embed_e0", action="store_true",
                   help="add E0(Z) atomic reference energies into the exported per-atom energy, so the .pt2 returns "
                        "ABSOLUTE energy (a drop-in for an E0-embedded TorchScript core). Forces are unaffected.")
    p.add_argument("--long-range-mode", dest="long_range_mode_arg", default=None,
                   help="synthetic combined export: long-range electrostatics mode (e.g. reciprocal-spectral-v1)")
    p.add_argument("--long-range-multipole-l", dest="long_range_multipole_l_arg", type=int, default=0,
                   help="synthetic combined export: electrostatic multipole order l (0=monopole)")
    p.add_argument("--long-range-reciprocal-backend", dest="long_range_reciprocal_backend_arg",
                   choices=("direct_kspace", "mesh_fft"), default="mesh_fft",
                   help="synthetic combined export: electrostatic reciprocal backend")
    p.add_argument("--long-range-kmax", dest="long_range_kmax_arg", type=int, default=2,
                   help="synthetic combined export: direct-kspace reciprocal cutoff")
    p.add_argument("--long-range-assignment", dest="long_range_assignment_arg",
                   choices=("cic", "pcs"), default="pcs",
                   help="synthetic combined export: mesh-FFT assignment stencil")
    p.add_argument("--long-range-full-ewald", dest="long_range_full_ewald_arg", action="store_true",
                   default=True,
                   help="synthetic combined export: use full-Ewald screened mesh-FFT weights")
    p.add_argument("--no-long-range-full-ewald", dest="long_range_full_ewald_arg", action="store_false",
                   help="synthetic combined export: disable full-Ewald screening")
    p.add_argument("--dispersion-mode", dest="dispersion_mode_arg", default=None,
                   help="synthetic combined export: long-range dispersion mode (e.g. mbd-slq)")
    p.add_argument("--dispersion-cutoff", dest="dispersion_cutoff_arg", type=float, default=8.0,
                   help="synthetic combined export: dispersion cutoff (A). ~8 gives ~5%% MBD-energy "
                        "convergence (the energy decays ~1/r^6); 4 was ~48%% under-converged.")
    p.add_argument("--lr-mesh-size", dest="lr_mesh_size_arg", type=int, default=32,
                   help="synthetic combined export: long-range/MBD PME mesh size")
    p.add_argument("--mbd-operator-backend", dest="mbd_operator_backend_arg", default=None,
                   choices=["edge_sparse", "pme_fft"],
                   help="synthetic combined export: MBD-SLQ operator backend (edge_sparse=direct cutoff sum; "
                        "pme_fft=reciprocal-only PME). Deploy runs the matching C++ MBD operator.")
    p.add_argument("--mbd-anisotropic", dest="mbd_anisotropic_arg", action="store_true",
                   help="synthetic combined export: anisotropic (l=2 tensor) polarizability -> [N,8] MBD source.")
    p.add_argument("--out", default="/tmp/fscetp_aoti.pt2")
    p.add_argument("--fallback", default=None,
                   help="path to an N-flexible TorchScript core (.pt) the LAMMPS engine should fall back to "
                        "when ntotal exceeds this .pt2's baked N (a ghost-count spike). Written into <out>.meta.")
    p.add_argument("--no-reciprocal-source", dest="no_reciprocal_source", action="store_true",
                   help="DISPERSION-ONLY AOTI: suppress the packed reciprocal_source 3rd output so the model "
                        "computes the MBD/SLQ dispersion energy in-graph (deployable AOTI; avoids the engine's "
                        "dispersion-edges + reciprocal-output fallback guard).")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dynamic", action="store_true",
                   help="export with dynamic_shapes (Dim for atom/edge counts) -> one .pt2 for all "
                        "system sizes (needed for LAMMPS' per-frame-varying N)")
    p.add_argument("--vary-degree", dest="vary_degree", type=int, default=0,
                   help="with --dynamic: also call the .pt2 at this avg degree (!= --degree) -> a "
                        "different EDGE count at the SAME atom count, proving one .pt2 handles varying "
                        "neighbor count (the thing that actually varies in LAMMPS NVE/NVT, where N is fixed)")
    p.add_argument("--n-dynamic", dest="n_dynamic", action="store_true", default=True,
                   help="(DEFAULT) make the atom count N a torch.export Dim -> ONE .pt2 for ANY N (no "
                        "padding, no N_max, no fallback). Works by disabling e3nn's jit-scripting of "
                        "o3.Linear (e3nn.set_optimization_defaults(jit_script_fx=False)), which specialized "
                        "N via torch.size(x)[0] typed as a TorchScript int. Numerically identical (script "
                        "vs fx GraphModule = same compute, same weights). Implies --dynamic.")
    p.add_argument("--static-n", dest="n_dynamic", action="store_false",
                   help="opt OUT of N-dynamic: bake N (legacy pad-to-N_max + dual-core fallback path). "
                        "Use only for a fixed-N .pt2 (~5%% faster kernels at one size, but needs padding "
                        "+ a TorchScript fallback for ntotal spikes in LAMMPS).")
    p.add_argument("--vary-atoms", dest="vary_atoms", type=int, default=0,
                   help="with --n-dynamic: also call the .pt2 at this atom count (!= --atoms) -> proves "
                        "one .pt2 handles a different N. Uses --degree for its neighbor count.")
    p.add_argument("--inductor-max-autotune", action="store_true",
                   help="enable Inductor max-autotune for AOTI compilation. This can improve the exported "
                        ".pt2 runtime, but substantially increases compile time; numerics are still checked "
                        "before the command exits.")
    p.add_argument("--assume-cutoff-edges", action="store_true",
                   help="assume every supplied edge has already been filtered to r <= cutoff and skip the "
                        "model-side edge mask. LAMMPS mff/torch filters neighbor-list skin edges before "
                        "calling the core, so this is safe for that deployment path and can reduce large "
                        "edge-tensor elementwise work. Do not use with raw neighbor lists that include skin.")
    p.add_argument("--preserve-edge-order", action="store_true",
                   help="skip the model-side argsort(edge_dst) and keep the caller's edge order. This is "
                        "mathematically the same scatter-sum graph but can change fp32 reduction order at "
                        "round-off level. Use only when the caller already supplies a stable/deployment "
                        "edge order or when fp32 round-off differences are acceptable.")
    p.add_argument("--cueq-product", action="store_true",
                   help="replace exact MACE product contractions with the experimental cuEquivariance "
                        "backend before tracing. This preserves fp32-level numerics and can improve large-N "
                        "throughput, but the resulting .pt2 requires cuEquivariance custom op registration "
                        "before AOTI loading.")
    p.add_argument("--fuse-selector-message-linear", action="store_true",
                   help="for eval/AOTI export, precompose interaction message_linear + fixed "
                        "avg-neighbor scaling + element selector (+ per-l output scale when present) "
                        "into one element-conditioned linear map. This changes only fp32 accumulation "
                        "order and is skipped automatically when the interaction uses attention or "
                        "nonlinear message normalization.")
    p.add_argument("--no-equiv", dest="no_equiv", action="store_true",
                   help="skip the rotation-equivariance gate on the loaded .pt2")
    args = p.parse_args()
    if args.operator_plan=="generated":
        args.owned_energy_mask=True
    if args.inductor_max_autotune:
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
        os.environ["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
        try:
            import torch._inductor.config as inductor_config
            inductor_config.max_autotune = True
            inductor_config.coordinate_descent_tuning = True
        except Exception as ex:
            print(f"[aoti] WARNING: failed to set Inductor autotune config directly: {type(ex).__name__}: {ex}")
        print("[aoti] Inductor max-autotune enabled (slower compile, benchmark before production use)")
    if args.n_dynamic:
        args.dynamic = True  # N-dynamic implies E-dynamic
        # MUST be set BEFORE any e3nn module is constructed (the model build below). jit_script_fx=True
        # compiles o3.Linear to a RecursiveScriptModule whose `torch.size(x)[0]` is a TorchScript int ->
        # int(SymInt) guard bakes N. As an fx GraphModule the size stays a symbolic SymInt -> N dynamic.
        import e3nn
        e3nn.set_optimization_defaults(jit_script_fx=False)
        print("[aoti] --n-dynamic: e3nn jit_script_fx=False (o3.Linear stays fx GraphModule -> N symbolic)")

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.allow_tf32:
        raise ValueError("TF32 is not allowed; use full float32 precision")
    # Hard-off TF32: keep full float32 matmul precision. TF32 would drop matmul
    # to roughly 1e-3 on some models, so unchanged numerics remains the default.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    if args.checkpoint:
        # Real deployment path: export the actual trained checkpoint. from_checkpoint rebuilds the
        # architecture from the checkpoint's hyperparameters and loads the (EMA) weights; the bare
        # energy model lives at obj.wrapper.model (a PureCartesianICTDFix, ZBL-wrapped only if the
        # checkpoint configured ZBL -- same 7-arg forward either way). The checkpoint's dtype/species
        # OVERRIDE the CLI so the .pt2 matches what LAMMPS will run.
        from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
        element_types = [s.strip() for s in args.elements.split(",") if s.strip()]
        obj = LAMMPS_MLIAP_MFF.from_checkpoint(
            checkpoint_path=args.checkpoint, element_types=element_types, device=args.device,
            avg_num_neighbors=args.avg_num_neighbors,
            operator_plan="checkpoint",  # Configure once, after export transforms/freezing.
        )
        model = obj.wrapper.model
        model.skip_input_validation = True  # set on the BARE model (the A.max().item() guard lives here,
                                            # not on the E0 wrapper) so make_fx tracing stays data-independent
        model.assume_edges_within_radius = bool(args.assume_cutoff_edges)
        model.preserve_edge_order = bool(args.preserve_edge_order)
        dtype = next(model.parameters()).dtype  # honor the trained dtype (likely float64)
        species_z = [int(z) for z in (getattr(model, "atomic_numbers", None) or SPECIES) if int(z) > 0] or list(SPECIES)
        if args.prune_to_elements:
            from ase.data import atomic_numbers as ase_atomic_numbers

            selected_z = [int(ase_atomic_numbers.get(symbol, 0)) for symbol in element_types]
            if any(z <= 0 for z in selected_z):
                bad = [symbol for symbol, z in zip(element_types, selected_z) if int(z) <= 0]
                raise ValueError(f"Cannot prune to unknown element symbols: {bad}")
            species_z = _prune_model_elements(model, selected_z)
            print(f"[aoti] pruned element-conditioned weights to species={species_z}")
        print(f"[aoti] loaded checkpoint {args.checkpoint}  trained_dtype={dtype}  species={species_z}  "
              f"avg_num_neighbors={getattr(model, 'avg_num_neighbors', None)}")
        if args.embed_e0:
            from mace_ictc.cli.export_libtorch_core import _e0_lut_from_keys_values
            aek = obj.wrapper.atomic_energy_keys.detach().cpu()
            aev = obj.wrapper.atomic_energy_values.detach().cpu()
            lut = _e0_lut_from_keys_values(aek, aev, dtype=dtype, device=device)
            model = _E0Wrap(model, lut).to(device=device)
            print(f"[aoti] embedded E0(Z) for Z={aek.tolist()} -> absolute-energy .pt2")
    model.eval()
    model.skip_input_validation = True
    if args.cueq_product:
        _replace_products_with_cueq(model, device=device)
        print("[aoti] cuEquivariance product backend enabled for export (experimental)")
    export_angular_basis = _configure_angular_basis_for_export(model, args.angular_basis)
    if args.angular_basis != "checkpoint" or export_angular_basis == "ictd":
        print(f"[aoti] angular_basis for export -> {export_angular_basis}")
    if args.fuse_selector_message_linear:
        n_fused = _enable_fused_selector_message_linears(model)
        print(f"[aoti] fused selector/message interaction linears enabled for {n_fused} block(s)")
    for param in model.parameters():
        param.requires_grad_(False)
    print("[aoti] model parameters frozen for force-only inference export")
    from mace_ictc.interfaces.deployment_triton import configure_mace_deployment_plan
    if args.operator_plan == "generated" and args.cueq_product:
        raise ValueError("Generated independent deployment cannot mix cuEq")
    plan_info = configure_mace_deployment_plan(model, policy=args.operator_plan, generated_layout=args.generated_layout,
                                               reference_buffer_policy=args.reference_buffer_policy)
    if plan_info["selected"] == "generated":
        args.owned_energy_mask = True
    print(f"[aoti] operator plan: {plan_info}")
    if args.assume_cutoff_edges:
        print("[aoti] assuming cutoff-filtered edges: model-side edge_mask skipped")
    if args.preserve_edge_order:
        print("[aoti] preserving caller edge order: model-side argsort(edge_dst) skipped")
    def _apply_species(g):
        # When the checkpoint's element set differs from the harness default, re-draw the species
        # vector A from it (deterministic seed, so positions/edges -- and thus the traced shapes --
        # are unchanged; only the atomic numbers feeding the embedding change). Using a Z the model
        # was NOT trained on would index its embedding out of range -> CUDA device-side assert, so
        # EVERY graph fed to the model (incl. the VARY-E proof) must go through this.
        if species_z == list(SPECIES):
            return g
        n = int(g[0].shape[0])
        gA = torch.Generator(device="cpu").manual_seed(42)
        sp = torch.tensor(species_z, dtype=torch.long)
        A_new = sp[torch.randint(0, len(species_z), (n,), generator=gA)].to(device)
        return (g[0], A_new) + tuple(g[2:])

    graph = _apply_species(make_fixed_graph(num_nodes=args.atoms, avg_degree=args.degree, dtype=dtype, device=device))

    def _with_dispersion_edges(g):
        from mace_ictc.models.dispersion import dispersion_neighbor_list, dispersion_mode_uses_canonical_edges

        disp_src, disp_dst, disp_shifts = dispersion_neighbor_list(
            g[0],
            g[2],
            g[6],
            float(getattr(_bare, "dispersion_cutoff", 0.0)),
            pbc=bool(getattr(_bare, "dispersion_pbc", True)),
            canonical_undirected=dispersion_mode_uses_canonical_edges(dispersion_mode),
            method=str(getattr(_bare, "dispersion_neighbor_method", "auto")),
            bruteforce_threshold=int(getattr(_bare, "dispersion_bruteforce_threshold", 1024)),
            max_num_neighbors=getattr(_bare, "dispersion_max_num_neighbors", None),
            allow_large_bruteforce_fallback=bool(
                getattr(_bare, "dispersion_allow_large_bruteforce_fallback", False)
            ),
        )
        if disp_src.numel() < 1:
            raise RuntimeError(
                "AOTI MBD/SLQ-MBD export example graph produced no dispersion edges; "
                "increase --atoms or --degree, or use a larger checkpoint dispersion_cutoff."
            )
        return (g[0],) + tuple(g[1:]) + (disp_src, disp_dst, disp_shifts.to(dtype=g[5].dtype))

    # Multipole long-range models emit a packed reciprocal_source as a 3rd graph output (E, force,
    # reciprocal_source); the C++ reciprocal solver does the sum at deploy time. Detect off the bare
    # model (the E0 wrapper forwards the flag).
    _bare = model.model if isinstance(model, _E0Wrap) else model
    emit_rs = bool(getattr(_bare, "long_range_exports_reciprocal_source", False))
    if getattr(args, "no_reciprocal_source", False) and emit_rs:
        emit_rs = False
        print("[aoti] --no-reciprocal-source: dispersion-only export (in-graph MBD energy, no reciprocal_source)")
    if emit_rs:
        print("[aoti] multipole long-range: exporting (E, force, reciprocal_source) 3-tuple")
    use_explicit_dispersion_edges = False
    dispersion_mode = str(getattr(_bare, "long_range_dispersion_mode", "none"))
    mbd_operator_backend = str(getattr(_bare, "mbd_operator_backend", "edge_sparse"))
    from mace_ictc.models.dispersion import dispersion_mode_needs_deployment_edges

    if (
        getattr(_bare, "dispersion", None) is not None
        and dispersion_mode == "mbd-slq"
        and mbd_operator_backend == "pme_fft"
    ):
        # pme_fft is reciprocal-only (no real-space dispersion edges): the model emits (omega, alpha) and the
        # C++ MBD solver runs the matching reciprocal PME operator (use_fft) with the same mesh/assignment/
        # alpha-prefactor (carried in the metadata). It still needs the AOTI-friendly SLQ quadrature, since
        # eigh on the Lanczos tridiagonal is not a deployable AOTI path.
        term = getattr(getattr(_bare, "dispersion", None), "term", None)
        if term is not None and hasattr(term, "quadrature"):
            term.quadrature = "newton-schulz"
            if hasattr(term, "probe_mode"):
                term.probe_mode = "atom-rademacher"
            if hasattr(term, "sqrt_iterations"):
                term.sqrt_iterations = max(int(getattr(term, "sqrt_iterations", 8)), 8)
            print("[aoti] dispersion pme_fft: SLQ atom-rademacher + newton-schulz quadrature for AOTI")
    # Pairwise C6 dispersion can ride on the deployment edge list. MBD/SLQ-MBD needs a second
    # explicit dispersion edge list so its cutoff can differ from the short-range MACE cutoff.
    if getattr(_bare, "dispersion", None) is not None and float(getattr(_bare, "dispersion_cutoff", 0.0) or 0.0) > 0.0:
        if dispersion_mode_needs_deployment_edges(
            dispersion_mode,
            mbd_operator_backend=mbd_operator_backend,
        ):
            if dispersion_mode == "mbd":
                raise NotImplementedError(
                    "AOTI export still does not support dense MBD: torch.linalg.eigh on the dense "
                    "3N x 3N oscillator matrix is not a deployable AOTI path. Use mbd-slq for AOTI."
                )
            use_explicit_dispersion_edges = True
            term = getattr(getattr(_bare, "dispersion", None), "term", None)
            if term is not None and hasattr(term, "quadrature"):
                term.quadrature = "newton-schulz"
                if hasattr(term, "probe_mode"):
                    term.probe_mode = "atom-rademacher"
                if hasattr(term, "sqrt_iterations"):
                    term.sqrt_iterations = max(int(getattr(term, "sqrt_iterations", 8)), 8)
                print("[aoti] dispersion: using SLQ atom-rademacher probes + newton-schulz quadrature for AOTI")
            print(f"[aoti] dispersion: exporting explicit dispersion edge inputs for "
                  f"{_bare.long_range_dispersion_mode} (cutoff={_bare.dispersion_cutoff})")
        else:
            print(f"[aoti] dispersion: export uses the edge list (dispersion_cutoff "
              f"{_bare.dispersion_cutoff} -> 0; set the LAMMPS pair cutoff to the dispersion range)")
            _bare.dispersion_cutoff = 0.0

    example_inputs = _with_dispersion_edges(graph) if use_explicit_dispersion_edges else (graph[0],) + tuple(graph[1:])
    example_disp_edges = int(example_inputs[-3].numel()) if use_explicit_dispersion_edges else 0
    if args.owned_energy_mask:example_inputs=tuple(example_inputs)+(torch.ones(graph[0].shape[0],device=device,dtype=dtype),)

    print(f"[aoti] device={device} dtype={dtype} atoms={args.atoms} edges={args.atoms*args.degree} "
          + (f"disp_edges={example_disp_edges} " if use_explicit_dispersion_edges else "")
          + f"route={args.route} attn_heads={args.attn_heads} torch={torch.__version__}")

    # ---- eager reference ----
    eager_fn = force_compute_fn_factory(
        model,
        training=False,
        emit_reciprocal_source=emit_rs,
        use_dispersion_edges=use_explicit_dispersion_edges,
        owned_energy_mask=args.owned_energy_mask,
    )
    e_ref, f_ref = _ef(eager_fn(*example_inputs))
    e_ref = e_ref.detach(); f_ref = f_ref.detach()
    print(f"[aoti] eager energy_sum={e_ref.sum().item():.6e}  force_absmax={f_ref.abs().max().item():.6e}")

    # ---- make_fx flatten (force into the graph) ----
    try:
        gm = trace_and_compile_force(
            model, example_inputs, training=False,
            compute_fn=force_compute_fn_factory(
                model,
                training=False,
                emit_reciprocal_source=emit_rs,
                use_dispersion_edges=use_explicit_dispersion_edges,
                owned_energy_mask=args.owned_energy_mask,
            ),
            do_compile=False,
        )
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[aoti] make_fx FLATTEN FAILED: {type(ex).__name__}: {ex}")
        return 1
    e_gm, f_gm = _ef(gm(*example_inputs))
    dE_gm = (e_gm - e_ref).abs().max().item(); dF_gm = (f_gm - f_ref).abs().max().item()
    print(f"[aoti] flat gm vs eager: dE={dE_gm:.3e} dF={dF_gm:.3e}  (should be ~0, bit-identical)")

    # ---- torch.export (optionally with dynamic_shapes for varying atom/edge counts) ----
    dyn = None
    if args.dynamic:
        from torch.export import Dim
        # inputs order: (pos[N,3], A[N], batch[N], edge_src[E], edge_dst[E], edge_shifts[E,3],
        #                cell[M,3,3][, dispersion_edge_src[D], dispersion_edge_dst[D],
        #                dispersion_edge_shifts[D,3]])
        Edim = Dim("n_edges", min=2, max=2**31-1)
        Ddim = Dim("n_dispersion_edges", min=1, max=2**31-1)
        if args.n_dynamic:
            # N-DYNAMIC: with e3nn jit_script_fx=False (set above) the make_fx flatten no longer bakes the
            # atom count -- N is a symbol -- so make N a Dim too. ONE .pt2 for ANY atom count AND any edge
            # count (no padding, no N_max, no fallback). num_mol (cell dim 0) stays 1 (LAMMPS is 1 graph).
            Ndim = Dim("n_atoms", min=2, max=2**31-1)
            dyn = ({0: Ndim}, {0: Ndim}, {0: Ndim}, {0: Edim}, {0: Edim}, {0: Edim}, None)
            if use_explicit_dispersion_edges:
                dyn = dyn + ({0: Ddim}, {0: Ddim}, {0: Ddim})
        else:
            # E-only dynamic (legacy padding path): make_fx BAKES N (the view(*x.shape[:-1], C, 2l+1) in the
            # ICTC layers via e3nn's scripted o3.Linear), so N stays static (one .pt2 per system size, padded
            # in LAMMPS). The EDGE count E stays symbolic (scatter/index_select) -- right for fixed-N NVE/NVT.
            dyn = (None, None, None, {0: Edim}, {0: Edim}, {0: Edim}, None)
            if use_explicit_dispersion_edges:
                dyn = dyn + ({0: Ddim}, {0: Ddim}, {0: Ddim})
    if args.owned_energy_mask and dyn is not None:dyn=tuple(dyn)+({0:Ndim} if args.n_dynamic else None,)
    prefer_export_strict = not _model_uses_cueq_product(model)
    try:
        exported, export_strict = _torch_export_with_strict_fallback(
            gm,
            example_inputs,
            dynamic_shapes=dyn,
            prefer_strict=prefer_export_strict,
        )
        print(f"[aoti] torch.export.export OK (dynamic={args.dynamic}, strict={export_strict})")
    except Exception as ex:
        print(f"[aoti] torch.export FAILED: {type(ex).__name__}: {ex}")
        return 1

    # ---- AOTInductor compile + package ----
    t0 = time.perf_counter()
    try:
        pt2 = _aoti_compile(exported, args.out)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[aoti] aoti_compile_and_package FAILED: {type(ex).__name__}: {ex}")
        return 1
    print(f"[aoti] compiled .pt2 -> {pt2}  ({time.perf_counter()-t0:.1f}s)")

    # Sidecar metadata the LAMMPS engine reads: baked atom count N (-> pad ntotal up to it), a valid
    # padding species, and an optional N-flexible TorchScript fallback for ntotal > N spikes.
    meta_path = str(args.out) + ".meta"
    pad_z = int(species_z[0]) if species_z else 1
    with open(meta_path, "w") as mf:
        mf.write(f"operator_plan {plan_info['selected']}\n")
        mf.write(f"generated_layout {args.generated_layout}\n")
        mf.write(f"message_passing_depth {len(_bare.interactions)}\n")
        if args.owned_energy_mask:mf.write("owned_energy_mask 1\n")
        if args.n_dynamic:
            # N-dynamic: the .pt2 accepts ANY atom count -> LAMMPS needs no padding/N_max.
            # A fallback can still be useful for unsupported AOTI/runtime combinations.
            mf.write("dynamic 1\n")
            mf.write("nmax 0\n")
            if use_explicit_dispersion_edges:
                mf.write("dispersion_edges 1\n")
            if _model_uses_cueq_product(model):
                mf.write("requires_torch_ops cuequivariance_ops_torch\n")
            if args.fallback:
                mf.write(f"fallback {args.fallback}\n")
        else:
            mf.write(f"nmax {args.atoms}\n")
            mf.write(f"pad_z {pad_z}\n")
            if use_explicit_dispersion_edges:
                mf.write("dispersion_edges 1\n")
            if _model_uses_cueq_product(model):
                mf.write("requires_torch_ops cuequivariance_ops_torch\n")
            if args.fallback:
                mf.write(f"fallback {args.fallback}\n")
    print(f"[aoti] wrote {meta_path}  ("
          + ("N-DYNAMIC (no padding)" if args.n_dynamic else f"nmax={args.atoms} pad_z={pad_z} fallback={args.fallback}") + ")")

    # Long-range/dispersion deploy metadata sidecar: engine reads "<core>.pt2.json" for the
    # reciprocal solver config and for MBD dispersion deployment metadata.  Pure MBD-SLQ AOTI exports
    # may not emit reciprocal_source, but still need this sidecar so C++ validates dispersion_cutoff
    # and the single-image runtime graph guard against the trained MBD graph rule.
    if emit_rs or use_explicit_dispersion_edges:
        import json as _json
        lr_meta = _long_range_deploy_metadata(
            _bare,
            export_reciprocal_source=emit_rs,
            use_explicit_dispersion_edges=use_explicit_dispersion_edges,
        )
        json_path = str(args.out) + ".json"
        with open(json_path, "w") as jf:
            _json.dump(lr_meta, jf, indent=2)
        print(f"[aoti] wrote {json_path} (multipole long-range deploy metadata, l={lr_meta['long_range_max_multipole_l']})")

    # ---- load back + verify numerics ----
    try:
        loaded = _aoti_load(pt2, device)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[aoti] aoti load FAILED: {type(ex).__name__}: {ex}")
        return 1
    out = loaded(*example_inputs)
    e_a, f_a = _ef(out)
    dE = (e_a - e_ref).abs().max().item()
    dF = (f_a - f_ref).abs().max().item() if f_a is not None else float("nan")
    escale = e_ref.abs().max().item() + 1e-30; fscale = f_ref.abs().max().item() + 1e-30
    tol = 1e-9 if dtype == torch.float64 else 3e-3
    ok = (dE / escale <= tol) and (dF / fscale <= tol)
    print(f"[aoti] AOTI vs eager: dE={dE:.3e} (rel {dE/escale:.2e})  dF={dF:.3e} (rel {dF/fscale:.2e})")
    print(f"[aoti] NUMERICAL-MATCH {'PASS' if ok else 'FAIL'} (rel tol {tol:.1e})")

    # ---- equivariance gate on the loaded .pt2 (HARD constraint): rotate positions;
    #      energy must be invariant and force covariant: f(x R^T) == f(x) R^T. ----
    if not args.no_equiv:
        R = _random_rotation(dtype, device)
        rest = tuple(example_inputs[1:])
        e0, f0 = _ef(loaded(example_inputs[0], *rest))
        e1, f1 = _ef(loaded(example_inputs[0] @ R.T, *rest))
        e_inv = (e1.sum() - e0.sum()).abs().item()
        f_cov = (f1 - f0 @ R.T).abs().max().item()
        e_sc = e0.abs().sum().item() + 1e-30
        f_sc = f0.abs().max().item() + 1e-30
        eqtol = 1e-9 if dtype == torch.float64 else 1e-3
        eq_ok = (e_inv / e_sc <= eqtol) and (f_cov / f_sc <= eqtol)
        print(f"[aoti] EQUIVARIANCE |E(Rx)-E(x)|={e_inv:.3e} (rel {e_inv/e_sc:.2e})  "
              f"|F(Rx)-F(x)R^T|={f_cov:.3e} (rel {f_cov/f_sc:.2e})  "
              f"{'PASS' if eq_ok else 'FAIL'} (rel tol {eqtol:.1e})")
        ok = ok and eq_ok
        del e0, f0, e1, f1

    # ---- dynamic-E proof: call the SAME .pt2 at a different EDGE count (same N) ----
    if args.dynamic and args.vary_degree > 0 and args.vary_degree != args.degree:
        g2 = _apply_species(make_fixed_graph(num_nodes=args.atoms, avg_degree=args.vary_degree, dtype=dtype, device=device))
        inp2 = _with_dispersion_edges(g2) if use_explicit_dispersion_edges else (g2[0],) + tuple(g2[1:])
        if args.owned_energy_mask:inp2=tuple(inp2)+(torch.ones(g2[0].shape[0],device=device,dtype=dtype),)
        e2e, f2e = _ef(eager_fn(*inp2)); e2e = e2e.detach(); f2e = f2e.detach()
        try:
            e2a, f2a = _ef(loaded(*inp2))
            d2e = (e2a - e2e).abs().max().item(); d2f = (f2a - f2e).abs().max().item()
            v_ok = (d2e / (e2e.abs().max().item() + 1e-30) <= tol) and (d2f / (f2e.abs().max().item() + 1e-30) <= tol)
            print(f"[aoti] VARY-E: exported@{args.atoms*args.degree} called@{args.atoms*args.vary_degree} edges "
                  f"(same {args.atoms} atoms) -> dE={d2e:.3e} dF={d2f:.3e}  "
                  f"{'PASS (one .pt2 handles varying neighbor count)' if v_ok else 'FAIL'}")
            ok = ok and v_ok
        except Exception as ex:
            print(f"[aoti] VARY-E call FAILED: {type(ex).__name__}: {ex}")
            ok = False
        del e2e, f2e

    # ---- dynamic-N proof: call the SAME .pt2 at a DIFFERENT ATOM COUNT N (the whole point) ----
    if args.n_dynamic and args.vary_atoms > 0 and args.vary_atoms != args.atoms:
        gN = _apply_species(make_fixed_graph(num_nodes=args.vary_atoms, avg_degree=args.degree, dtype=dtype, device=device))
        inpN = _with_dispersion_edges(gN) if use_explicit_dispersion_edges else (gN[0],) + tuple(gN[1:])
        if args.owned_energy_mask:inpN=tuple(inpN)+(torch.ones(gN[0].shape[0],device=device,dtype=dtype),)
        eNe, fNe = _ef(eager_fn(*inpN)); eNe = eNe.detach(); fNe = fNe.detach()
        try:
            eNa, fNa = _ef(loaded(*inpN))
            dNe = (eNa - eNe).abs().max().item(); dNf = (fNa - fNe).abs().max().item()
            n_ok = (dNe / (eNe.abs().max().item() + 1e-30) <= tol) and (dNf / (fNe.abs().max().item() + 1e-30) <= tol)
            print(f"[aoti] VARY-N: exported@{args.atoms} atoms called@{args.vary_atoms} atoms -> "
                  f"dE={dNe:.3e} dF={dNf:.3e}  "
                  f"{'PASS (ONE .pt2 handles any atom count)' if n_ok else 'FAIL'}")
            ok = ok and n_ok
            # equivariance at the new N too (HARD constraint must hold at every N)
            if not args.no_equiv:
                Rn = _random_rotation(dtype, device)
                e0n, f0n = _ef(loaded(inpN[0], *tuple(inpN[1:])))
                e1n, f1n = _ef(loaded(inpN[0] @ Rn.T, *tuple(inpN[1:])))
                ei = (e1n.sum() - e0n.sum()).abs().item() / (e0n.abs().sum().item() + 1e-30)
                fc = (f1n - f0n @ Rn.T).abs().max().item() / (f0n.abs().max().item() + 1e-30)
                eqtol = 1e-9 if dtype == torch.float64 else 1e-3
                print(f"[aoti] VARY-N EQUIVARIANCE @{args.vary_atoms} atoms: E-inv rel {ei:.2e}  F-cov rel {fc:.2e}  "
                      f"{'PASS' if (ei <= eqtol and fc <= eqtol) else 'FAIL'}")
                ok = ok and (ei <= eqtol and fc <= eqtol)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"[aoti] VARY-N call FAILED: {type(ex).__name__}: {ex}")
            ok = False
        del eNe, fNe

    # Free the trace/export/compile intermediates before timing. Otherwise on large
    # float64 systems the script holds eager refs + flat gm + ExportedProgram +
    # compiled package + loaded model all at once, and the eager timing run then
    # OOMs (it is not an AOTI limitation -- the loaded .pt2 alone is small).
    del exported, gm, e_gm, f_gm, e_a, f_a
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---- speed: eager forward+force vs AOTI ----
    t_eager = _time(lambda: eager_fn(*example_inputs), device, args.iters, args.warmup)
    t_aoti = _time(lambda: loaded(*example_inputs), device, args.iters, args.warmup)
    print(f"[aoti] eager fwd+force : {t_eager:8.3f} ms")
    print(f"[aoti] AOTI .pt2       : {t_aoti:8.3f} ms")
    print(f"[aoti] SPEEDUP         : {t_eager / max(t_aoti,1e-9):6.2f}x")
    if device.type == "cuda":
        print(f"[aoti] peak CUDA mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
