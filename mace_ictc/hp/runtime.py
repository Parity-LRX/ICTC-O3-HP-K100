# SPDX-License-Identifier: MIT
"""Whole-network entrypoints; precision and trainability stay caller-controlled."""
import torch


def create_model(*, channels=64, lmax=3, edge_lmax=None, num_interactions=2,
                 correlation=3, use_reduced_cg=False,
                 atomic_numbers=(1, 6, 8), cutoff=5.0, avg_num_neighbors=50.0,
                 device="cuda", dtype=torch.float32, electrostatics=None):
    """Create a parameterized Cartesian bridge-U model, without a runtime plan.

    New TP base coefficients are fixed by construction; radial/linear/SC weights
    are trainable. This factory creates initialized weights, not a trained
    potential. Existing checkpoints should be loaded through from_checkpoint.
    electrostatics optionally configures the original learned long-range head;
    its new-factory default is reciprocal-only mesh electrostatics with a
    fixed Gaussian length smearing_sigma=1.0 in coordinate units.
    """
    if not isinstance(num_interactions, int) or isinstance(num_interactions, bool) or num_interactions < 1:
        raise ValueError("num_interactions must be a positive integer")
    from .operators import _degree, _positive_integer
    _positive_integer(channels, "channels")
    _positive_integer(correlation, "correlation")
    if correlation not in (1,2,3):
        raise ValueError("HP correlation currently supports 1, 2, 3; cor4/5 experiments are separate")
    if not isinstance(use_reduced_cg, bool):
        raise TypeError("use_reduced_cg must be a boolean")
    if use_reduced_cg:
        raise ValueError("The independent HP factory uses native full CG; cuEq reduced-basis generation is not enabled")
    _degree(lmax, "lmax")
    edge_lmax = lmax if edge_lmax is None else edge_lmax
    _degree(edge_lmax, "edge_lmax")
    if edge_lmax < lmax:
        raise ValueError("edge_lmax must be at least lmax")
    from collections.abc import Mapping
    long_range_options = {}
    if electrostatics is not None:
        if not isinstance(electrostatics, Mapping):
            raise TypeError("electrostatics must be a configuration mapping or None")
        allowed = {'mode', 'boundary', 'reciprocal_backend', 'kmax', 'mesh_size',
                   'source_channels', 'hidden_dim', 'filter_hidden_dim', 'neutralize',
                   'include_k0', 'energy_partition', 'green_mode', 'assignment',
                   'slab_padding_factor', 'mesh_fft_full_ewald', 'mesh_fft_reciprocal_only',
                   'max_multipole_l', 'multipole_gate_init', 'smearing_sigma'}
        unknown = set(electrostatics) - allowed
        if unknown:
            raise ValueError(f"Unknown electrostatics options: {sorted(unknown)}")
        options = dict(mode='reciprocal-spectral-v1', boundary='periodic',
                       reciprocal_backend='mesh_fft', mesh_size=32, assignment='pcs',
                       mesh_fft_full_ewald=True, mesh_fft_reciprocal_only=True)
        options.update(electrostatics)
        if 'smearing_sigma' not in options:
            options['smearing_sigma'] = (1.0 if options['reciprocal_backend'] == 'mesh_fft'
                                        and options['mesh_fft_full_ewald'] else None)
        long_range_options = {'long_range_' + key: value for key, value in options.items()}
    from mace_ictc.models.factory import build_baseline_model
    from mace_ictc.utils.config import ModelConfig
    import e3nn
    previous_dtype = torch.get_default_dtype()
    previous_optimization = e3nn.get_optimization_defaults()
    try:
        # Scripted e3nn size(x)[0] becomes an int and bakes N during make_fx.
        # Build symbolic-friendly FX submodules, restoring caller defaults below.
        e3nn.set_optimization_defaults(jit_script_fx=False)
        config = ModelConfig(dtype=dtype, internal_compute_dtype=dtype, channel_in=channels,
            irreps_output_conv_channels=channels, lmax=lmax, max_radius=cutoff,
            max_radius_main=cutoff, function_type="bessel")
        return build_baseline_model(
            config, avg_num_neighbors=avg_num_neighbors, num_interaction=num_interactions,
            route="baseline", product_backend="ictd-bridge-u", correlation=correlation,
            use_reduced_cg=use_reduced_cg, freeze_conv_tp_weight=True, radial_sqrt_num_basis=False,
            edge_lmax=edge_lmax, attn_heads=0, atomic_numbers=list(atomic_numbers),
            ictd_save_tp_mode="fully-connected", invariant_channels=32,
            device=torch.device(device), dtype=config.dtype,
            **long_range_options,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
        e3nn.set_optimization_defaults(**previous_optimization)


def energy_and_forces(model, positions, atomic_numbers, batch, edge_src, edge_dst,
                      edge_shifts, cell, *, training=False, owned_mask=None):
    """Evaluate a prebuilt graph through the ordinary model forward.

    Returns energy [B], forces [N,3], atomic_energy [N]. Shifts are lattice
    coordinates and cell is [B,3,3]. No neighbor construction, E0 addition or
    optimizer is performed. training=True retains force-loss mixed derivatives;
    inference returns detached results. Owned masks select energy contributions,
    while force derivatives are taken for all supplied local/ghost coordinates.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("Full forces require autograd; use no_grad instead of inference_mode")
    if positions.ndim != 2 or positions.shape[1] != 3 or cell.ndim != 3 or cell.shape[1:] != (3, 3):
        raise ValueError("Expected positions [N,3] and cell [B,3,3]")
    n = positions.shape[0]
    if atomic_numbers.shape != (n,) or batch.shape != (n,):
        raise ValueError("Atomic numbers and graph batch must have shape [N]")
    if owned_mask is not None and (owned_mask.shape != (n,) or owned_mask.requires_grad):
        raise ValueError("owned_mask must be a fixed [N] tensor")
    with torch.enable_grad():
        pos = positions if positions.requires_grad else positions.detach().requires_grad_(True)
        output = model(pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
        atomic = (output[0] if isinstance(output, tuple) else output).reshape(n)
        selected = atomic if owned_mask is None else atomic * owned_mask.to(atomic)
        energy = selected.new_zeros(cell.shape[0]).index_add(0, batch, selected)
        # Keep the coordinate dependency for zero-edge/empty graphs as well.
        force = -torch.autograd.grad(energy.sum() + pos.sum()*0.0, pos,
                                     create_graph=training, retain_graph=training)[0]
        result = {"energy": energy, "forces": force, "atomic_energy": atomic}
        return result if training else {key: value.detach() for key, value in result.items()}


def _layout(value, *, optional=False):
    if not (optional and value is None) and value not in ("packed", "fused_pack"):
        raise ValueError("HP layout must be 'packed' or 'fused_pack'")


def check_model(model: torch.nn.Module, *, training=False, capture=False) -> dict:
    """Read-only coverage check; no freezing, casting, planning or compilation."""
    from mace_ictc.models.operator_plans import generated_plan_unavailable
    reasons = generated_plan_unavailable(model, training=training, capture=capture)
    return {"supported": not reasons, "reasons": reasons}


def configure_model(model: torch.nn.Module, *, training=False, policy="auto",
                    layout="packed", gate_cache_bytes=0, capture=False,
                    wide_channel_policy="auto", high_l_policy="auto",
                    reference_buffer_policy="auto",message_schedule="native",
                    graph_reduction="auto",topology_backend="native",
                    csr_accumulator_limit=None,csr_slice_search=False,
                    geometry_schedule="auto",message_chunk_edges=None,
                    species_schedule="grouped",
                    training_gate_schedule="materialized",mixed_channel_tile=None,
                    mixed_geometry_schedule="fused",training_gate_gemm_policy="torch",
                    training_adjoint_channel_tile=None,training_message_linear_layout="node_component",
                    radial_checkpoint=False,inference_message_layout="native",
                    inference_adjoint_channel_tile="auto",inference_gate_adjoint_rows=0,
                    inference_gemm_policy="torch",inference_message_adjoint_edges=0,
                    inference_basis_fusion=False,species_linear_schedule="auto",
                    electrostatics_backend="auto") -> dict:
    """Configure every interaction for eager training or full-force inference.

    auto selects qualified independent HP operators or reports why the existing
    route is retained. generated requires coverage; reference restores the
    reference recipe. No dtype/trainability is changed. TF32 is disabled globally.
    Wide channels default to shared species GEMMs and dense-graph CSR forward;
    off restores the original HP schedules. The inference gate-cache budget is
    bytes per interaction; training requires zero.
    training_gate_schedule defaults to materialized. cached retains forward
    gates but pages first/mixed cotangents; recompute also rebuilds gates. Both
    are opt-in eager L<=3 schedules with live radial parameters. Materialized
    gates use generated coordinate Hessians; paged coordinate Hessians and
    higher derivatives retain the formula.
    A tuple/list may specify one gate schedule per interaction. radial_checkpoint
    optionally recomputes radial MLP activations with non-reentrant checkpointing.
    mixed_channel_tile optionally changes only the generated mixed-adjoint
    channel tiling (0, 32 or 64); None preserves the established schedule.
    mixed_geometry_schedule="split" moves the harmonic mixed adjoint to a
    separate edge/channel kernel. training_gate_gemm_policy="k100_tiled"
    bounds gradient GEMM rows for the qualified K100 radial projection.
    training_adjoint_channel_tile independently bounds the first TP adjoint
    channel tile (0, 32 or 64) during materialized eager training.
    training_message_linear_layout defaults to "node_component", packing node/angular rows
    for a shared FP32 channel GEMM in message linears. Live weights and their
    mixed adjoints use ordinary differentiable linear operations. "einsum"
    restores the old training layout; inference and reference plans use it automatically.
    inference_message_layout="degree" keeps TP outputs in degree-major angular
    rows through the following linear and coordinate adjoint. It is opt-in,
    eager L<=3 inference with layout="fused_pack"; frozen gate/TP weights use
    the fast route. No weights are frozen by this configuration call.
    inference_adjoint_channel_tile (0, 32, 64 or "auto") tiles only frozen-gate first
    adjoints that also need input-feature gradients. The default "auto" uses
    bounded A100 dispatch for inference and is disabled during training. It
    leaves forward and higher derivatives unchanged; 0 forces the old schedule.
    inference_gate_adjoint_rows optionally bounds rows per frozen-gate gradient
    GEMM independently of TP message chunks. Zero keeps the original GEMM;
    positive values use views and the same output storage, with no new cache.
    inference_gemm_policy="dtk2604" opts into measured FP32 gate GEMMs on the
    qualified K100/DTK stack. Other shapes/stacks retain PyTorch. It requires
    eager inference; differentiable higher adjoints keep the original formula.
    inference_message_adjoint_edges optionally bounds the frozen TP backward
    chunk independently of forward/cache chunks. Cached gates are sliced as
    views; missing gates are recomputed for each smaller chunk. Zero preserves
    the forward chunk size. This does not change the cache budget automatically.
    inference_basis_fusion fuses fixed basis transforms into native K100 frozen
    indexed low-L SC circuit loads/stores and first feature adjoints. Frozen
    weight projections are reused with tensor-version invalidation. Trainable
    banks and higher derivatives retain the differentiable polynomial route.
    Small force workloads retain packed HIP basis transforms and fresh weight
    projections, dispatched by node/channel work relative to compute-unit count.
    species_linear_schedule="grouped" groups integer-indexed element-conditioned
    linear maps (including self connections) by species and uses one GEMM per
    occupied species, avoiding [N,C_out,C_in] weights. It supports arbitrary
    channels, angular degree, species count and floating dtype, including live
    parameters/higher derivatives. It is eager only and copies group sizes to
    the host on each call; no graph/halo membership is cached. auto preserves
    the existing wide-channel policy; gather forces the original weight gather.
    High-L auto uses bounded sparse cubic SC and mode-specific TP schedules for
    L4/5. high_l_policy=off restores the preceding high-L HP schedules.
    reference_buffer_policy=auto keeps dense high-L cubic U on the host after
    a validated GPU plan exists; device retains conventional device buffers.
    Parameter devices/dtypes stay unchanged. Parallel wrappers/groups keep
    device buffers, and reference/export paths restore them before use.
    electrostatics_backend selects auto, hip or reference mesh operations
    independently of backbone planning, without changing the functional.
    """
    from ictc_o3.backends.k100.fp32_policy import strict_fp32
    if electrostatics_backend not in ('auto','hip','reference'):
        raise ValueError('electrostatics_backend must be auto, hip or reference')
    if capture and electrostatics_backend == 'hip':
        raise ValueError('Native HIP electrostatics currently requires eager execution')
    strict_fp32()
    _layout(layout)
    if message_schedule=='compact':
        raise ValueError('The experimental STF-coordinate schedule is retired; use native or roles in the original ICTC basis')
    if message_schedule not in ('native','roles'):
        raise ValueError('message_schedule must be native or roles')
    if graph_reduction not in ('auto','atomic','csr'):
        raise ValueError('graph_reduction must be auto, atomic or csr')
    if topology_backend not in ('native','device'):
        raise ValueError('topology_backend must be native or device')
    if csr_accumulator_limit is not None and (not isinstance(csr_accumulator_limit,int) or isinstance(csr_accumulator_limit,bool) or csr_accumulator_limit<1):
        raise ValueError('csr_accumulator_limit must be a positive integer or None')
    if not isinstance(csr_slice_search,bool):raise ValueError('csr_slice_search must be a bool')
    if geometry_schedule not in ('auto','native','streamed'):
        raise ValueError('geometry_schedule must be auto, native or streamed')
    if species_schedule not in ('grouped','indexed'):
        raise ValueError('species_schedule must be grouped or indexed')
    if species_linear_schedule not in ('auto','gather','grouped'):
        raise ValueError('species_linear_schedule must be auto, gather or grouped')
    if species_linear_schedule=='grouped' and capture and policy!='reference':
        raise ValueError('Grouped species linears require eager execution')
    if inference_adjoint_channel_tile != 'auto' and (isinstance(inference_adjoint_channel_tile,bool)
            or not isinstance(inference_adjoint_channel_tile,int)
            or inference_adjoint_channel_tile not in (0,32,64)):
        raise ValueError('inference_adjoint_channel_tile must be 0, 32, 64 or auto')
    if inference_adjoint_channel_tile not in (0,'auto') and training:
        raise ValueError('inference_adjoint_channel_tile requires training=False')
    if (isinstance(inference_gate_adjoint_rows,bool) or not isinstance(inference_gate_adjoint_rows,int)
            or inference_gate_adjoint_rows<0):
        raise ValueError('inference_gate_adjoint_rows must be a nonnegative integer')
    if inference_gate_adjoint_rows and training:
        raise ValueError('inference_gate_adjoint_rows requires training=False')
    if inference_gemm_policy not in ('torch','dtk2604'):
        raise ValueError('inference_gemm_policy must be torch or dtk2604')
    if inference_gemm_policy!='torch' and (training or capture):
        raise ValueError('inference_gemm_policy=dtk2604 requires eager inference')
    if (isinstance(inference_message_adjoint_edges,bool) or not isinstance(inference_message_adjoint_edges,int)
            or inference_message_adjoint_edges<0):
        raise ValueError('inference_message_adjoint_edges must be a nonnegative integer')
    if inference_message_adjoint_edges and (training or capture):
        raise ValueError('inference_message_adjoint_edges requires eager inference')
    if not isinstance(inference_basis_fusion,bool):
        raise ValueError('inference_basis_fusion must be bool')
    if inference_basis_fusion and (training or capture):
        raise ValueError('inference_basis_fusion requires eager inference')
    if inference_message_layout not in ('native','degree'):
        raise ValueError('inference_message_layout must be native or degree')
    if inference_message_layout=='degree' and policy!='reference':
        if training or capture:
            raise ValueError('Degree message layout currently requires eager inference')
        if layout!='fused_pack':
            raise ValueError('Degree message layout requires layout=fused_pack')
    gate_schedules=(training_gate_schedule,) if isinstance(training_gate_schedule,str) else training_gate_schedule
    if (not isinstance(gate_schedules,(tuple,list)) or not gate_schedules
            or any(not isinstance(value,str) or value not in ('materialized','cached','recompute') for value in gate_schedules)):
        raise ValueError('training_gate_schedule must contain materialized, cached or recompute')
    paged_gates=any(value!='materialized' for value in gate_schedules)
    if not isinstance(radial_checkpoint,bool):raise ValueError('radial_checkpoint must be a bool')
    if not training and policy!='reference' and (paged_gates or radial_checkpoint):
        raise ValueError('A training gate schedule requires training=True')
    if mixed_channel_tile is not None:
        if isinstance(mixed_channel_tile,bool) or not isinstance(mixed_channel_tile,int) or mixed_channel_tile not in (0,32,64):
            raise ValueError('mixed_channel_tile must be None, 0, 32 or 64')
        if not training and policy!='reference':
            raise ValueError('A mixed-adjoint channel tile requires training=True')
    if mixed_geometry_schedule not in ('fused','split'):
        raise ValueError('mixed_geometry_schedule must be fused or split')
    if training_gate_gemm_policy not in ('torch','k100_tiled'):
        raise ValueError('training_gate_gemm_policy must be torch or k100_tiled')
    if training_adjoint_channel_tile is not None:
        if (isinstance(training_adjoint_channel_tile,bool)
                or not isinstance(training_adjoint_channel_tile,int)
                or training_adjoint_channel_tile not in (0,32,64)):
            raise ValueError('training_adjoint_channel_tile must be None, 0, 32 or 64')
        if not training and policy!='reference':
            raise ValueError('The training adjoint channel tile requires training=True')
    if not training and policy!='reference' and (mixed_geometry_schedule!='fused'
            or training_gate_gemm_policy!='torch'):
        raise ValueError('Mixed geometry and gate gradient GEMM schedules require training=True')
    if paged_gates and policy!='reference' and (training_gate_gemm_policy!='torch'
            or mixed_geometry_schedule!='fused' or training_adjoint_channel_tile is not None):
        raise ValueError('Explicit first-adjoint, split-geometry and gate-GEMM schedules require materialized training gates')
    if capture and policy!='reference' and (training_gate_gemm_policy!='torch'
            or mixed_geometry_schedule!='fused' or training_adjoint_channel_tile is not None):
        raise ValueError('Explicit first-adjoint, split-geometry and gate-GEMM schedules require eager execution')
    if message_chunk_edges is not None and (not isinstance(message_chunk_edges,int) or isinstance(message_chunk_edges,bool) or message_chunk_edges<1):
        raise ValueError('message_chunk_edges must be a positive integer or None')
    from mace_ictc.models.operator_plans import mace_backbone_core
    core=mace_backbone_core(model)
    interactions=tuple(layer for layer in getattr(core,'interactions',()) if hasattr(layer,'tp'))
    per_layer=not isinstance(training_gate_schedule,str)
    if per_layer and len(gate_schedules)!=len(interactions):
        raise ValueError('A per-interaction training_gate_schedule must match the interaction count')
    if radial_checkpoint and capture and policy!='reference':raise ValueError('radial_checkpoint requires eager execution')
    if (policy!='reference' and inference_message_layout=='degree' and hasattr(core,'lmax')
            and hasattr(core,'ictd_fix_edge_lmax') and max(core.lmax,core.ictd_fix_edge_lmax)>3):
        raise ValueError('Degree message layout currently requires hidden/edge L<=3')
    if (policy!='reference' and (paged_gates or radial_checkpoint or mixed_channel_tile is not None
            or training_adjoint_channel_tile is not None) and hasattr(core,'lmax')
            and hasattr(core,'ictd_fix_edge_lmax') and max(core.lmax,core.ictd_fix_edge_lmax)>3):
        raise ValueError('Explicit training gate/mixed schedules currently require hidden/edge L<=3')
    if (policy!='reference' and message_schedule!='native' and hasattr(core,'lmax')
            and hasattr(core,'ictd_fix_edge_lmax') and max(core.lmax,core.ictd_fix_edge_lmax)>3):
        raise ValueError('Requested low-L message schedule requires hidden/edge L<=3')
    from mace_ictc.models.operator_plans import configure_mace_runtime_plan
    info = configure_mace_runtime_plan(
        model, policy=policy, training=training, capture=capture,
        gate_cache_bytes=gate_cache_bytes, generated_layout=layout,
        wide_channel_policy=wide_channel_policy,
        high_l_policy=high_l_policy,
        reference_buffer_policy=reference_buffer_policy,
        training_message_linear_layout=training_message_linear_layout,
    )
    selected=info['selected']=='generated'
    basis_enabled=False
    if selected and inference_basis_fusion and torch.version.hip:
        parameter=next(core.parameters())
        basis_enabled=(parameter.is_cuda and
            getattr(torch.cuda.get_device_properties(parameter.device),'gcnArchName','').split(':')[0]=='gfx926')
    from ictc_o3.ops.symmetric_contraction import ICTDBridgeUSymmetricContractionSO3
    from mace_ictc.models.pure_cartesian_ictd_fix import ElementConditionedLinearSO3
    indexed_count=0
    for module in core.modules():
        if isinstance(module,ElementConditionedLinearSO3):
            module._grouped_species_gemm=selected and species_linear_schedule=='grouped'
            if species_linear_schedule!='auto':
                module._shared_species_gemm=False
        if isinstance(module,ICTDBridgeUSymmetricContractionSO3):
            module._species_circuit=(selected and species_schedule=='indexed'
                and max(module.lmax,module.target_lmax)<=3 and module._factorized_circuit
                and not module._partitioned_circuit)
            module._hip_fused_basis=bool(basis_enabled and module._species_circuit
                and not getattr(module,'_staged_circuit',False)
                and not getattr(module,'_compact_circuit_adjoints',False))
            if training or not module._hip_fused_basis:
                module._frozen_projected_bank=None
            indexed_count+=int(module._species_circuit)
    info['species_schedule']=species_schedule if selected else 'reference'
    info['species_linear_schedule']=species_linear_schedule if selected else 'reference'
    info['counts']['grouped_species_linears']=sum(bool(getattr(module,'_grouped_species_gemm',False)) for module in core.modules())
    info['counts']['shared_species_gemm']=sum(bool(getattr(module,'_shared_species_gemm',False)) for module in core.modules())
    info['counts']['indexed_species_circuits']=indexed_count
    info['counts']['hip_basis_fusion']=sum(bool(getattr(module,'_hip_fused_basis',False)) for module in core.modules())
    info['inference_basis_fusion']=bool(info['counts']['hip_basis_fusion'])
    for index,layer in enumerate(interactions):
        if training or not selected:
            layer.tp._frozen_gate_weights=None
        degree_layout=selected and not training and inference_message_layout=='degree'
        layer.tp._inference_degree_layout=degree_layout
        layer.tp._inference_adjoint_channel_tile=inference_adjoint_channel_tile if selected and not training else 0
        layer.tp._inference_gate_adjoint_rows=inference_gate_adjoint_rows if selected and not training else 0
        layer.tp._inference_gemm_policy=inference_gemm_policy if selected and not training else 'torch'
        layer.tp._inference_message_adjoint_edges=inference_message_adjoint_edges if selected and not training else 0
        if hasattr(layer,'message_linear'):
            layer.message_linear._inference_degree_layout=degree_layout
        schedule=gate_schedules[index] if per_layer else gate_schedules[0]
        layer.tp._training_gate_schedule=schedule if selected and training else 'materialized'
        layer.tp._radial_checkpoint=bool(selected and training and radial_checkpoint)
        layer.tp._generated_second_channel_tile=mixed_channel_tile if selected and training else None
        layer.tp._training_adjoint_channel_tile=training_adjoint_channel_tile if selected and training else None
        layer.tp._generated_second_harmonic_split=bool(selected and training and mixed_geometry_schedule=='split')
        layer.tp._training_gate_gemm_policy=training_gate_gemm_policy if selected and training else 'torch'
        layer.tp._generated_specialized_adjoint=selected and message_schedule!='native'
        layer.tp._generated_second_roles=selected and message_schedule!='native'
        layer.tp._generated_compact_cartesian=False
        cached=getattr(layer.tp,'_bond_projection_adjoint_plan',None)
        if cached is not None:cached[1].pop('compact_cartesian_state',None)
        if selected and graph_reduction!='auto':
            layer.tp._generated_csr_forward=graph_reduction=='csr'
        layer.tp._generated_topology=topology_backend if selected else 'native'
        layer.tp._csr_slice_search=selected and csr_slice_search
        if selected and geometry_schedule!='auto':
            layer.tp._generated_geometry_streaming=geometry_schedule=='streamed'
        if selected and message_chunk_edges is not None:
            layer.tp._message_chunk_edges=message_chunk_edges
        if selected and csr_accumulator_limit is not None:
            layer.tp._csr_accumulator_limit=csr_accumulator_limit
        # Reconfiguration must not retain rejected experimental source adjoints.
        layer.tp._generated_csr_fused_adjoint=False
    info['message_schedule']=message_schedule if selected else 'reference'
    info['training_gate_schedule']=training_gate_schedule if selected and training else 'materialized'
    info['mixed_channel_tile']=mixed_channel_tile if selected and training else None
    info['training_adjoint_channel_tile']=training_adjoint_channel_tile if selected and training else None
    info['mixed_geometry_schedule']=mixed_geometry_schedule if selected and training else 'fused'
    info['training_gate_gemm_policy']=training_gate_gemm_policy if selected and training else 'torch'
    info['radial_checkpoint']=bool(selected and training and radial_checkpoint)
    info['inference_message_layout']=inference_message_layout if selected and not training else 'native'
    info['inference_adjoint_channel_tile']=inference_adjoint_channel_tile if selected and not training else 0
    info['inference_gate_adjoint_rows']=inference_gate_adjoint_rows if selected and not training else 0
    info['inference_gemm_policy']=inference_gemm_policy if selected and not training else 'torch'
    info['inference_message_adjoint_edges']=inference_message_adjoint_edges if selected and not training else 0
    info['counts']['degree_packed_inference']=sum(bool(getattr(layer.tp,'_inference_degree_layout',False))
                                                for layer in interactions)
    info['counts']['paged_training_gates']=sum(
        getattr(layer.tp,'_training_gate_schedule','materialized')!='materialized'
        for layer in getattr(core,'interactions',()) if hasattr(layer,'tp'))
    info.update(graph_reduction=graph_reduction if selected else 'reference',
                topology_backend=topology_backend if selected else 'native',
                csr_accumulator_limit=csr_accumulator_limit,csr_slice_search=selected and csr_slice_search)
    info.update(geometry_schedule=geometry_schedule if selected else 'reference',message_chunk_edges=message_chunk_edges)
    if selected:
        info['counts']['generated_csr_forward']=sum(bool(getattr(layer.tp,'_generated_csr_forward',False))
            for layer in getattr(core,'interactions',()) if hasattr(layer,'tp'))
    if selected and not training and torch.version.hip:
        parameter=next(core.parameters())
        if (parameter.is_cuda and
                getattr(torch.cuda.get_device_properties(parameter.device),'gcnArchName','').split(':')[0]=='gfx926'):
            # This DTK target cannot compile the optional Triton helpers. Use
            # qualified HIP harmonics and normalized SiLU, while reporting
            # the remaining PyTorch helpers explicitly.
            core._fused_direction_harmonics=getattr(core,'ictd_fix_edge_lmax',4)<=3
            if capture:
                # Keep the preceding capture path; HIP SiLU is qualified eager.
                for module in core.modules():
                    if hasattr(module,'_fuse_activation'):module._fuse_activation=False
                info['counts']['activation_fusion']=0
            for layer in interactions:layer.tp._generated_topology='native'
            info['topology_backend']='native'
            info['device_adaptations']={'target':'gfx926',
                'native_hip':['tp_adjoint','csr_forward','message_layout','species_circuit_first_order']
                    +([] if capture else ['normalized_silu'])
                    +(['basis_packing'] if info['inference_basis_fusion'] else [])
                    +(['direction_harmonics'] if core._fused_direction_harmonics else []),
                'torch_helpers':['csr_topology']+(['normalized_silu'] if capture else [])
                    +([] if core._fused_direction_harmonics else ['direction_harmonics'])}
    from ictc_o3.electrostatics.api import configure_electrostatics
    info['electrostatics'] = configure_electrostatics(
        model, backend='reference' if policy == 'reference' or capture else electrostatics_backend)
    return {**info, "layout": layout if info["selected"] == "generated" else None}


def prepare_for_export(model: torch.nn.Module, *, policy="auto", layout=None, dynamic=True,
                       reference_buffer_policy="auto") -> dict:
    """Prepare a frozen inference model before FX/AOTInductor export.

    None preserves an explicit layout, or uses packed on a fresh model.
    checkpoint preserves the checkpoint route. This API prepares, but does not
    export, a package; existing .pt2 binaries must be re-exported to change kernels.
    Qualified high-L cor3 keeps canonical U on CPU after plan preparation;
    reference_buffer_policy="device" retains conventional device buffers.
    """
    from ictc_o3.backends.k100.fp32_policy import strict_fp32
    strict_fp32()
    _layout(layout, optional=True)
    if dynamic and policy != "checkpoint":
        from mace_ictc.models.operator_plans import generated_plan_unavailable,mace_backbone_core
        core = mace_backbone_core(model)
        if not generated_plan_unavailable(core):
            scripted = [name for name, child in core.named_modules() if isinstance(child, torch.jit.ScriptModule)]
            if scripted:
                raise ValueError("Dynamic AOTI requires FX e3nn submodules: use create_model, or set "
                    "e3nn.set_optimization_defaults(jit_script_fx=False) before rebuilding/loading the model. "
                    f"Scripted modules remain: {scripted[:3]}. Use dynamic=False only for an intentionally static-N package.")
    from mace_ictc.interfaces.deployment_triton import configure_mace_deployment_plan
    from ictc_o3.electrostatics.api import configure_electrostatics
    configure_electrostatics(model, backend='reference')
    return configure_mace_deployment_plan(model, policy=policy, generated_layout=layout,
                                         reference_buffer_policy=reference_buffer_policy)
