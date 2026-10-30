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
    from mace_ictc.interfaces.foundation_training import FoundationTrainHP
    from mace_ictc.runtime.planning import generated_plan_unavailable, mace_backbone_core
    core = mace_backbone_core(model)
    if isinstance(core, FoundationTrainHP):
        result = core.check_training()
        if not training:
            result["reasons"].append("FoundationTrainHP coverage requires training=True")
        if capture:
            result["reasons"].append("Foundation training requires eager execution")
        result["supported"] = not result["reasons"]
        return result
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
                    electrostatics_backend="auto", plan=None, foundation_config=None) -> dict:
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


    Pass plan=RuntimePlan(...) for grouped configuration. Use either a plan
    or nondefault legacy keywords in a call; both share the same validation.
    FoundationTrainHP uses foundation_config=FoundationTrainingConfig(...)
    instead of generic backbone schedules or RuntimePlan. With no config,
    its current explicit foundation schedules are retained.
    """

    from mace_ictc.runtime.config import RuntimePlan
    from mace_ictc.runtime.configure import apply_runtime_plan
    options = dict(
        training=training,
        policy=policy,
        layout=layout,
        gate_cache_bytes=gate_cache_bytes,
        capture=capture,
        wide_channel_policy=wide_channel_policy,
        high_l_policy=high_l_policy,
        reference_buffer_policy=reference_buffer_policy,
        message_schedule=message_schedule,
        graph_reduction=graph_reduction,
        topology_backend=topology_backend,
        csr_accumulator_limit=csr_accumulator_limit,
        csr_slice_search=csr_slice_search,
        geometry_schedule=geometry_schedule,
        message_chunk_edges=message_chunk_edges,
        species_schedule=species_schedule,
        training_gate_schedule=training_gate_schedule,
        mixed_channel_tile=mixed_channel_tile,
        mixed_geometry_schedule=mixed_geometry_schedule,
        training_gate_gemm_policy=training_gate_gemm_policy,
        training_adjoint_channel_tile=training_adjoint_channel_tile,
        training_message_linear_layout=training_message_linear_layout,
        radial_checkpoint=radial_checkpoint,
        inference_message_layout=inference_message_layout,
        inference_adjoint_channel_tile=inference_adjoint_channel_tile,
        inference_gate_adjoint_rows=inference_gate_adjoint_rows,
        inference_gemm_policy=inference_gemm_policy,
        inference_message_adjoint_edges=inference_message_adjoint_edges,
        inference_basis_fusion=inference_basis_fusion,
        species_linear_schedule=species_linear_schedule,
        electrostatics_backend=electrostatics_backend,
    )
    requested = RuntimePlan.from_legacy(**options)
    from mace_ictc.interfaces.foundation_training import FoundationTrainHP
    from mace_ictc.runtime.planning import mace_backbone_core
    core = mace_backbone_core(model)
    if isinstance(core, FoundationTrainHP):
        if plan is not None:
            raise ValueError("Use FoundationTrainingConfig for MH-1, not RuntimePlan")
        generic = RuntimePlan.from_legacy(**{**options, "training": False, "policy": "auto", "capture": False})
        if generic != RuntimePlan():
            raise ValueError("Use foundation_config instead of generic backbone or inference schedules for MH-1")
        if policy not in ("auto", "generated", "reference"):
            raise ValueError("operator plan must be auto, reference or generated")
        if capture:
            raise ValueError("Foundation training requires eager execution")
        if policy == "reference":
            if foundation_config is not None:
                raise ValueError("A generated foundation config cannot accompany the native reference policy")
            core._reference_policy = True
            return dict(selected="reference", architecture="nonlinear MACE foundation", reasons=[])
        if not training:
            raise ValueError("FoundationTrainHP configuration requires training=True")
        coverage = core.check_training()
        if not coverage["supported"]:
            if policy == "generated":
                raise ValueError("; ".join(coverage["reasons"]))
            core._reference_policy = True
            return dict(selected="reference", **coverage)
        return core.configure_training(foundation_config or core.training_config)
    if foundation_config is not None:
        raise ValueError("foundation_config requires a FoundationTrainHP model")
    if plan is not None:
        if not isinstance(plan, RuntimePlan):
            raise TypeError("plan must be a RuntimePlan")
        if requested != RuntimePlan():
            raise ValueError("Use a RuntimePlan or nondefault legacy runtime keywords in one call")
        requested = plan
    return apply_runtime_plan(model, requested)


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
        from mace_ictc.runtime.planning import generated_plan_unavailable,mace_backbone_core
        core = mace_backbone_core(model)
        if not generated_plan_unavailable(core):
            scripted = [name for name, child in core.named_modules() if isinstance(child, torch.jit.ScriptModule)]
            if scripted:
                raise ValueError("Dynamic AOTI requires FX e3nn submodules: use create_model, or set "
                    "e3nn.set_optimization_defaults(jit_script_fx=False) before rebuilding/loading the model. "
                    f"Scripted modules remain: {scripted[:3]}. Use dynamic=False only for an intentionally static-N package.")
    from mace_ictc.runtime.export import configure_mace_deployment_plan
    from ictc_o3.electrostatics.api import configure_electrostatics
    configure_electrostatics(model, backend='reference')
    return configure_mace_deployment_plan(model, policy=policy, generated_layout=layout,
                                         reference_buffer_policy=reference_buffer_policy)
