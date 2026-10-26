# SPDX-License-Identifier: MIT
"""Structured model configuration; no operator selection during import."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MessagePlan:
    layout: str = 'packed'
    schedule: str = 'native'
    reduction: str = 'auto'
    topology_backend: str = 'native'
    csr_accumulator_limit: int | None = None
    csr_slice_search: bool = False
    geometry_schedule: str = 'auto'
    chunk_edges: int | None = None


@dataclass(frozen=True)
class ContractionPlan:
    wide_channel_policy: str = 'auto'
    high_l_policy: str = 'auto'
    reference_buffer_policy: str = 'auto'
    species_schedule: str = 'grouped'
    species_linear_schedule: str = 'auto'


@dataclass(frozen=True)
class TrainingPlan:
    gate_schedule: str | tuple[str, ...] | list[str] = 'materialized'
    mixed_channel_tile: int | None = None
    mixed_geometry_schedule: str = 'fused'
    gate_gemm_policy: str = 'torch'
    adjoint_channel_tile: int | None = None
    message_linear_layout: str = 'node_component'
    radial_checkpoint: bool = False


@dataclass(frozen=True)
class InferencePlan:
    gate_cache_bytes: int = 0
    message_layout: str = 'native'
    adjoint_channel_tile: int | str = 'auto'
    gate_adjoint_rows: int = 0
    gemm_policy: str = 'torch'
    message_adjoint_edges: int = 0
    basis_fusion: bool = False


@dataclass(frozen=True)
class ElectrostaticsPlan:
    backend: str = 'auto'


@dataclass(frozen=True)
class RuntimePlan:
    """Requested schedules, resolved once when configure_model is called.

    Architecture/device coverage is checked by the existing runtime planner.
    Tensor precision and parameter trainability stay under caller control.
    A plan contains settings only; no live tensors or model references are held.
    """
    training: bool = False
    policy: str = "auto"
    capture: bool = False
    message: MessagePlan = field(default_factory=MessagePlan)
    contraction: ContractionPlan = field(default_factory=ContractionPlan)
    training_schedule: TrainingPlan = field(default_factory=TrainingPlan)
    inference: InferencePlan = field(default_factory=InferencePlan)
    electrostatics: ElectrostaticsPlan = field(default_factory=ElectrostaticsPlan)

    def __post_init__(self):
        if not isinstance(self.message, MessagePlan):
            raise TypeError("message must be a MessagePlan")
        if not isinstance(self.contraction, ContractionPlan):
            raise TypeError("contraction must be a ContractionPlan")
        if not isinstance(self.training_schedule, TrainingPlan):
            raise TypeError("training_schedule must be a TrainingPlan")
        if not isinstance(self.inference, InferencePlan):
            raise TypeError("inference must be an InferencePlan")
        if not isinstance(self.electrostatics, ElectrostaticsPlan):
            raise TypeError("electrostatics must be an ElectrostaticsPlan")

    @classmethod
    def from_legacy(cls, **options):
        """Translate the established configure_model keywords without changing defaults."""
        unknown = set(options) - set(_LEGACY_DEFAULTS)
        if unknown:
            raise TypeError(f"Unknown runtime options: {sorted(unknown)}")
        values = {**_LEGACY_DEFAULTS, **options}
        return cls(
            training=values["training"],
            policy=values["policy"],
            capture=values["capture"],
            message=MessagePlan(
                layout=values["layout"],
                schedule=values["message_schedule"],
                reduction=values["graph_reduction"],
                topology_backend=values["topology_backend"],
                csr_accumulator_limit=values["csr_accumulator_limit"],
                csr_slice_search=values["csr_slice_search"],
                geometry_schedule=values["geometry_schedule"],
                chunk_edges=values["message_chunk_edges"],
            ),
            contraction=ContractionPlan(
                wide_channel_policy=values["wide_channel_policy"],
                high_l_policy=values["high_l_policy"],
                reference_buffer_policy=values["reference_buffer_policy"],
                species_schedule=values["species_schedule"],
                species_linear_schedule=values["species_linear_schedule"],
            ),
            training_schedule=TrainingPlan(
                gate_schedule=values["training_gate_schedule"],
                mixed_channel_tile=values["mixed_channel_tile"],
                mixed_geometry_schedule=values["mixed_geometry_schedule"],
                gate_gemm_policy=values["training_gate_gemm_policy"],
                adjoint_channel_tile=values["training_adjoint_channel_tile"],
                message_linear_layout=values["training_message_linear_layout"],
                radial_checkpoint=values["radial_checkpoint"],
            ),
            inference=InferencePlan(
                gate_cache_bytes=values["gate_cache_bytes"],
                message_layout=values["inference_message_layout"],
                adjoint_channel_tile=values["inference_adjoint_channel_tile"],
                gate_adjoint_rows=values["inference_gate_adjoint_rows"],
                gemm_policy=values["inference_gemm_policy"],
                message_adjoint_edges=values["inference_message_adjoint_edges"],
                basis_fusion=values["inference_basis_fusion"],
            ),
            electrostatics=ElectrostaticsPlan(
                backend=values["electrostatics_backend"],
            ),
        )

    def to_legacy(self):
        """Return a fresh keyword mapping for the model runtime planner."""
        return dict(
            training=self.training,
            policy=self.policy,
            capture=self.capture,
            layout=self.message.layout,
            message_schedule=self.message.schedule,
            graph_reduction=self.message.reduction,
            topology_backend=self.message.topology_backend,
            csr_accumulator_limit=self.message.csr_accumulator_limit,
            csr_slice_search=self.message.csr_slice_search,
            geometry_schedule=self.message.geometry_schedule,
            message_chunk_edges=self.message.chunk_edges,
            wide_channel_policy=self.contraction.wide_channel_policy,
            high_l_policy=self.contraction.high_l_policy,
            reference_buffer_policy=self.contraction.reference_buffer_policy,
            species_schedule=self.contraction.species_schedule,
            species_linear_schedule=self.contraction.species_linear_schedule,
            training_gate_schedule=self.training_schedule.gate_schedule,
            mixed_channel_tile=self.training_schedule.mixed_channel_tile,
            mixed_geometry_schedule=self.training_schedule.mixed_geometry_schedule,
            training_gate_gemm_policy=self.training_schedule.gate_gemm_policy,
            training_adjoint_channel_tile=self.training_schedule.adjoint_channel_tile,
            training_message_linear_layout=self.training_schedule.message_linear_layout,
            radial_checkpoint=self.training_schedule.radial_checkpoint,
            gate_cache_bytes=self.inference.gate_cache_bytes,
            inference_message_layout=self.inference.message_layout,
            inference_adjoint_channel_tile=self.inference.adjoint_channel_tile,
            inference_gate_adjoint_rows=self.inference.gate_adjoint_rows,
            inference_gemm_policy=self.inference.gemm_policy,
            inference_message_adjoint_edges=self.inference.message_adjoint_edges,
            inference_basis_fusion=self.inference.basis_fusion,
            electrostatics_backend=self.electrostatics.backend,
        )


_LEGACY_DEFAULTS = {
    "training": False,
    "policy": 'auto',
    "layout": 'packed',
    "gate_cache_bytes": 0,
    "capture": False,
    "wide_channel_policy": 'auto',
    "high_l_policy": 'auto',
    "reference_buffer_policy": 'auto',
    "message_schedule": 'native',
    "graph_reduction": 'auto',
    "topology_backend": 'native',
    "csr_accumulator_limit": None,
    "csr_slice_search": False,
    "geometry_schedule": 'auto',
    "message_chunk_edges": None,
    "species_schedule": 'grouped',
    "training_gate_schedule": 'materialized',
    "mixed_channel_tile": None,
    "mixed_geometry_schedule": 'fused',
    "training_gate_gemm_policy": 'torch',
    "training_adjoint_channel_tile": None,
    "training_message_linear_layout": 'node_component',
    "radial_checkpoint": False,
    "inference_message_layout": 'native',
    "inference_adjoint_channel_tile": 'auto',
    "inference_gate_adjoint_rows": 0,
    "inference_gemm_policy": 'torch',
    "inference_message_adjoint_edges": 0,
    "inference_basis_fusion": False,
    "species_linear_schedule": 'auto',
    "electrostatics_backend": 'auto',
}
