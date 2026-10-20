"""Differentiable contraction plans for the standalone MACE-ICTC core.

Only constant basis layout is planned. Live features and learned weights must
never be cached. These algebraic plans retain ordinary PyTorch higher derivatives.
"""
from __future__ import annotations

import torch


def mace_backbone_core(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap deployment/DDP wrappers without selecting an unrelated backbone."""
    from .pure_cartesian_ictd_fix import PureCartesianICTDFix
    while not isinstance(model, PureCartesianICTDFix):
        child = getattr(model, "module", None)
        if child is None:
            child = getattr(model, "model", None)
        if not isinstance(child, torch.nn.Module) or child is model:
            break
        model = child
    return model


def generated_plan_unavailable(model: torch.nn.Module, *, training: bool = False,
                               capture: bool = False) -> list[str]:
    """Check qualified coverage; never change precision or parameter trainability.

    Interaction depth is deliberately unrestricted: each layer owns its plan.
    The default policy is conservative about unqualified architecture variants.
    """
    import importlib.util
    from .pure_cartesian_ictd_fix import PureCartesianICTDFix, ICTDBridgeUSymmetricContractionSO3
    core = mace_backbone_core(model)
    if not isinstance(core, PureCartesianICTDFix):
        return ["model is not the MACE-ICTC backbone"]
    reasons = []
    sample = next(core.parameters())
    if not sample.is_cuda or sample.dtype != torch.float32:
        reasons.append("requires CUDA FP32 (precision is not changed automatically)")
    if core.edge_compute_dtype != torch.float32:
        reasons.append("internal compute dtype must also be FP32")
    if core.channels < 1 or core.lmax < 0 or core.ictd_fix_edge_lmax < core.lmax:
        reasons.append("requires positive channels and 0 <= hidden L <= edge L")
    if getattr(core, "angular_basis", "ictd") != "ictd":
        reasons.append("requires the Cartesian ICTC angular basis")
    if any(type(m).__module__.startswith("cuequivariance") or getattr(m, "tp_cueq", None) is not None
           for m in core.modules()):
        reasons.append("independent generated operators cannot mix cuEq")
    products = [m for m in core.modules() if isinstance(m, ICTDBridgeUSymmetricContractionSO3)]
    if len(products) != len(core.interactions) or not products:
        reasons.append("requires bridge-U products in every interaction")
    elif any(m.use_reduced_cg for m in products):
        reasons.append("independent generated coverage requires native full CG")
    elif any(any(c.correlation not in (1,2,3) for c in m.symmetric_contractions.contractions)
             for m in products):
        reasons.append("generated symmetric contraction currently supports correlation 1, 2, 3")
    if any(int(getattr(m, "interaction_attn_heads", 0)) for m in core.interactions):
        reasons.append("interaction attention is not qualified")
    if getattr(core, "ictd_fix_route", "baseline") != "baseline":
        reasons.append("default generated coverage requires the baseline MACE route")
    if importlib.util.find_spec("triton") is None:
        reasons.append("Triton is unavailable")
    if training:
        if capture:
            reasons.append("make_fx training capture uses the reference route")
        if any(layer.tp.weight.requires_grad for layer in core.interactions):
            reasons.append("TP base path weights are trainable; they are not frozen automatically")
    elif any(p.requires_grad for p in core.parameters()):
        reasons.append("inference model parameters must be frozen")
    return reasons


def configure_mace_runtime_plan(model: torch.nn.Module, *, policy: str = "auto",
                                training: bool = False, capture: bool = False,
                                gate_cache_bytes: int = 0,
                                generated_layout: str = "packed",
                                wide_channel_policy: str = "auto",
                                high_l_policy: str = "auto",
                                reference_buffer_policy: str = "auto") -> dict:
    """Default to qualified generated eager operators, with a reference escape.

    Zero gate cache is intentional for the default, including deep models.
    Callers may explicitly budget an inference cache per interaction for speed.
    """
    from .fp32_policy import strict_fp32
    strict_fp32()
    if policy not in ("auto", "reference", "generated"):
        raise ValueError("operator plan must be auto, reference or generated")
    if generated_layout not in ("packed","canonical","fused_pack"):
        raise ValueError("Unsupported generated message layout")
    if wide_channel_policy not in ("auto","off"):
        raise ValueError("wide_channel_policy must be auto or off")
    if high_l_policy not in ("auto","off"):
        raise ValueError("high_l_policy must be auto or off")
    if reference_buffer_policy not in ('auto','device'):
        raise ValueError('reference_buffer_policy must be auto or device')
    core = mace_backbone_core(model)
    from .reference_u_residency import host_reference_unavailable,restore_model_reference_u,restore_reference_u
    if policy == "reference":
        from .pure_cartesian_ictd_fix import PureCartesianICTDFix
        counts = set_mace_backbone_plan(core) if isinstance(core, PureCartesianICTDFix) else {}
        restore_model_reference_u(core)
        return {"requested": policy, "selected": "reference", "reasons": [], "counts": counts}
    reasons = generated_plan_unavailable(core, training=training, capture=capture)
    if reasons:
        if policy == "generated":
            raise ValueError("Generated operator plan unavailable: " + "; ".join(reasons))
        from .pure_cartesian_ictd_fix import PureCartesianICTDFix
        counts=set_mace_backbone_plan(core) if isinstance(core,PureCartesianICTDFix) else {}
        restore_model_reference_u(core)
        return {"requested": policy, "selected": "reference", "reasons": reasons, "counts": counts}
    if training and gate_cache_bytes:
        raise ValueError("Training uses live radial gates; inference gate cache must be zero")
    wide=(wide_channel_policy=="auto" and core.channels>=96
          and max(core.lmax,core.ictd_fix_edge_lmax)<=3)
    counts = set_mace_backbone_plan(
        core, bond_first=True, angular_backend="triton_full", path_assembly="concat",
        edge_fusion="triton_remat", activation_fusion=True, compact_contraction=True,
        factorized_circuit=True, streamed_gates=True, generated_message=True,
        generated_second=True, fused_harmonics=True, fold_frozen_path_weights=True,
        shared_channel_gemm=not training, edge_chunk_size=65536 if training else 262144,
        shared_species_gemm=wide and core.num_elements<=4,
        generated_csr_forward="auto" if wide else False,
        gate_cache_bytes=gate_cache_bytes, generated_layout=generated_layout,
    )
    from .pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3
    high=(high_l_policy=="auto" and 4<=core.lmax<=5 and core.ictd_fix_edge_lmax<=5)
    cubic=any(c.correlation==3 for module in core.modules()
              if isinstance(module,ICTDBridgeUSymmetricContractionSO3)
              for c in module.symmetric_contractions.contractions)
    high_l_device_capability=(torch.cuda.get_device_capability(next(core.parameters()).device)
                              if high else None)
    counts.update(high_l_sparse_sc=0,high_l_partitioned_sc=0,high_l_geometry=0)
    reference_buffer_reason=host_reference_unavailable(model)
    host_u=(reference_buffer_policy=='auto' and high and reference_buffer_reason is None)
    counts['host_reference_sc']=0
    for layer in core.interactions:
        layer.tp._generated_geometry_streaming=high and core.channels>=64
        tiled=(high and core.channels>=64 and
               ((training and (core.lmax<5 or cubic)) or
                (not training and core.lmax==5 and
                 not (cubic and high_l_device_capability==(8,9)))))
        layer.tp._generated_channel_tile=32 if tiled else 0
        counts["high_l_geometry"]+=int(layer.tp._generated_geometry_streaming)
    for module in core.modules():
        if isinstance(module,ICTDBridgeUSymmetricContractionSO3):
            module._require_factorized_plan=policy=="generated"
            orders={c.correlation for c in module.symmetric_contractions.contractions}
            module._host_reference_u=host_u and orders=={3}
            counts['host_reference_sc']+=int(module._host_reference_u)
            if not module._host_reference_u:restore_reference_u(module)
            if high and orders=={3}:
                module._factorized_circuit=False
                module._adaptive_sparse_tiling=True
                counts["factorized_circuit"]-=1
                counts["high_l_sparse_sc"]+=1
            if high and orders=={2} and module.target_lmax>0 and (training or (module.lmax==5 and core.channels%32)):
                module._partitioned_circuit=True
                counts["high_l_partitioned_sc"]+=1
    return {"requested": policy, "selected": "generated", "reasons": [], "counts": counts,
            "wide_channel_policy":wide_channel_policy,"wide_channels_active":wide,
            "high_l_policy":high_l_policy,"high_l_active":high,
            "high_l_device_capability":high_l_device_capability,
            "reference_buffer_policy":reference_buffer_policy,
            "reference_buffer_reason":reference_buffer_reason,
            "arithmetic":"IEEE FP32; TF32 disabled"}


def bond_first_projection(a: torch.Tensor, b: torch.Tensor, projector: torch.Tensor,
                          angular_backend: str = "torch") -> torch.Tensor:
    """Contract a single-channel geometry leg before broadcasting over channels.

    a: [..., channels, m1], b: [..., 1, m2], projector: [m2, m1*k].
    Unlike broadcasting a batched matmul over channels, this bmm has no repeated
    geometry map and never forms [..., channels, m1*m2].
    """
    magnetic = torch.matmul(b.squeeze(-2), projector).reshape(
        *b.shape[:-2], a.shape[-1], projector.shape[-1] // a.shape[-1]
    )
    from .angular_operator import angular_product
    return angular_product(a, magnetic, angular_backend)


def set_mace_backbone_plan(model: torch.nn.Module, *, bond_first: bool = False,
                           angular_backend: str = "torch", path_assembly: str = "reference",
                           edge_fusion: str = "reference", species_bank: bool = False,
                           activation_fusion: bool = False, banked_contraction: bool = False,
                           compact_contraction: bool = False,
                           factorized_contraction: bool = False,
                           factorized_circuit: bool = False,
                           edge_chunk_size: int = 65536,
                           streamed_gates: bool = False, path_fusion: bool = False,
                           path_tile_edges: int = 1, path_warps: int = 4,
                           generated_message: bool = False,
                           fold_frozen_path_weights: bool = False,
                           gate_cache_bytes: int = 0,
                           fused_harmonics: bool = False,
                           shared_channel_gemm: bool = False,
                           shared_species_gemm: bool = False,
                           generated_csr_forward: bool | str = False,
                           generated_second: bool = False,
                           generated_layout: str = "packed") -> dict[str, int]:
    """Select neutral backbone plans without importing or configuring CHORUS.

    Generated Cartesian messages accelerate C64 FP32 forward/first adjoints
    with frozen TP weights, including live radial gates in force training.
    Unsupported contexts retain the established route. The
    gate cache is an explicit per-interaction byte budget; zero rematerializes
    all gates. All parameters remain live and checkpoint-compatible.
    """
    if angular_backend not in ("torch", "triton", "triton_full"):
        raise ValueError("angular_backend must be torch, triton or triton_full")
    if path_assembly not in ("reference", "concat"):
        raise ValueError("path_assembly must be reference or concat")
    if edge_fusion not in ("reference", "triton", "cuda", "cuda_projection", "triton_projection", "triton_csr", "triton_remat"):
        raise ValueError("Unsupported edge_fusion plan")
    if not isinstance(edge_chunk_size,int) or isinstance(edge_chunk_size,bool) or edge_chunk_size<1:
        raise ValueError("edge_chunk_size must be a positive integer")
    if streamed_gates and edge_fusion!="triton_remat":
        raise ValueError("streamed_gates requires triton_remat geometry tiles")
    if (generated_message or fold_frozen_path_weights or gate_cache_bytes) and not streamed_gates:
        raise ValueError("generated messages, folded gates and gate cache require streamed_gates")
    if generated_second and not generated_message:raise ValueError("generated_second requires generated_message")
    if generated_csr_forward not in (False,True,"auto"):
        raise ValueError("generated_csr_forward must be a boolean or auto")
    if generated_csr_forward and not generated_message:
        raise ValueError("CSR forward requires generated_message")
    if generated_layout not in ("packed","canonical","fused_pack"):
        raise ValueError("generated_layout must be packed, canonical or fused_pack")
    if generated_layout!="packed" and not generated_message:
        raise ValueError("fused generated layout requires generated_message")
    if not isinstance(gate_cache_bytes,int) or isinstance(gate_cache_bytes,bool) or gate_cache_bytes<0:
        raise ValueError("gate_cache_bytes must be a nonnegative integer")
    if path_fusion and edge_fusion!="triton_remat":
        raise ValueError("path_fusion requires triton_remat geometry tiles")
    if path_tile_edges not in (1,2,4,8) or isinstance(path_tile_edges,bool):
        raise ValueError("path_tile_edges must be 1, 2, 4 or 8")
    if path_warps not in (4,8) or isinstance(path_warps,bool):raise ValueError("path_warps must be 4 or 8")
    factorized_contraction=bool(factorized_contraction or factorized_circuit)
    active=(bond_first or angular_backend!="torch" or path_assembly!="reference"
            or edge_fusion!="reference" or species_bank or activation_fusion
            or banked_contraction or compact_contraction or factorized_contraction or fused_harmonics or shared_channel_gemm or shared_species_gemm)
    if active:
        for module in model.modules():
            if (type(module).__module__.startswith("cuequivariance")
                    or type(module).__name__.startswith("Cueq")
                    or getattr(module,"tp_cueq",None) is not None):
                raise ValueError("Independent ICTC plans cannot mix cuEq operators; select a native ICTC backend")
    from .ictd_irreps import HarmonicPathWeightedTensorProduct, EdgeWeightedPathPreservingTensorProduct, EquivariantChannelLinearSO3, EquivariantChannelLinearSO3Rect
    from .pure_cartesian_ictd_fix import ICTDResidualInteractionBlock, _Normalize2MomSiLU, ICTDBridgeUSymmetricContractionSO3,ElementConditionedLinearSO3
    from ._mace_symmetric_contraction import _Contraction
    if hasattr(model,"angular_basis") and hasattr(model,"ictd_fix_edge_lmax"):
        model._fused_direction_harmonics=bool(fused_harmonics)
    count = {"bond_first": 0, "path_assembly": 0, "edge_fusion": 0, "species_bank": 0,
             "activation_fusion":0, "banked_contraction":0, "compact_contraction":0,
             "factorized_contraction":0,"factorized_circuit":0,"streamed_gates":0,
             "generated_message":0,"fold_frozen_path_weights":0,"shared_channel_gemm":0,"shared_species_gemm":0,"generated_second":0,"generated_csr_forward":0}
    for module in model.modules():
        if isinstance(module,ElementConditionedLinearSO3):
            module._grouped_species_gemm=False
            module._shared_species_gemm=bool(shared_species_gemm)
            count["shared_species_gemm"]+=int(bool(shared_species_gemm))
        if isinstance(module,(EquivariantChannelLinearSO3,EquivariantChannelLinearSO3Rect)):
            for adapter in module.adapters.values():adapter._shared_gemm=bool(shared_channel_gemm)
            count["shared_channel_gemm"]+=int(bool(shared_channel_gemm))
        if isinstance(module, HarmonicPathWeightedTensorProduct):
            module._use_bond_first_projection = bool(bond_first)
            module._angular_backend = angular_backend
            count["bond_first"] += 1
        if isinstance(module, EdgeWeightedPathPreservingTensorProduct):
            module._path_assembly = path_assembly
            module._message_chunk_edges=edge_chunk_size
            module._fused_path_forward=bool(path_fusion);module._fused_path_adjoint=bool(path_fusion)
            module._path_tile_edges=path_tile_edges;module._path_warps=path_warps
            module._generated_forward=bool(generated_message);module._generated_adjoint=bool(generated_message)
            module._generated_second=bool(generated_second)
            module._generated_layout=generated_layout
            module._generated_blocks=0;module._generated_schedule=0;module._generated_reg_limit=0
            module._generated_channel_tile=0
            module._generated_geometry_streaming=False
            module._csr_accumulator_limit=48
            module._generated_csr_forward=generated_csr_forward
            count["generated_csr_forward"]+=int(bool(generated_csr_forward))
            module._fold_path_weights=bool(fold_frozen_path_weights);module._gate_cache_bytes=gate_cache_bytes
            count["path_assembly"] += 1
            count["generated_message"]+=int(bool(generated_message))
            count["generated_second"]+=int(bool(generated_second))
            count["fold_frozen_path_weights"]+=int(bool(fold_frozen_path_weights))
        if isinstance(module, ICTDResidualInteractionBlock):
            module._edge_fusion = edge_fusion
            module._streamed_gates=bool(streamed_gates)
            count["edge_fusion"] += 1
            count["streamed_gates"]+=int(bool(streamed_gates))
        if isinstance(module,_Contraction):
            module._use_species_bank=bool(species_bank)
            module._use_banked_contraction=bool(banked_contraction)
            count["species_bank"]+=1
            count["banked_contraction"]+=1
        if isinstance(module,_Normalize2MomSiLU):
            module._fuse_activation=bool(activation_fusion)
            count["activation_fusion"]+=1
        if isinstance(module,ICTDBridgeUSymmetricContractionSO3):
            module._host_reference_u=False
            module._species_circuit=False
            module._partitioned_circuit=False
            module._compact_circuit_adjoints=False
            module._staged_circuit=False
            module._bounded_pair_circuit=False
            module._adaptive_sparse_tiling=False
            module._factorized_circuit=bool(factorized_circuit)
            count["factorized_circuit"]+=1
            module._factorized_contraction=bool(factorized_contraction)
            count["factorized_contraction"]+=1
            module._compact_contraction=bool(compact_contraction)
            count["compact_contraction"]+=1
    return count
