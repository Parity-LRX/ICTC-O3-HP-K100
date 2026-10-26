# SPDX-License-Identifier: MIT
"""Whole-model HP API and compatibility imports for ``ictc_o3`` operators.

Importing this namespace never builds CUDA extensions or selects a model plan.
See docs/HP_API.md for model usage and docs/O3_API.md for standalone operators.
"""
from .operators import (
    EdgeTensorProduct, GraphMessage, PackedGraphMessage, channel_linear,
    configure_symmetric_contraction, create_symmetric_contraction, direction_harmonics, normalized_silu,
)
from .runtime import check_model, configure_model, create_model, energy_and_forces, prepare_for_export
from ictc_o3.electrostatics.api import MeshElectrostatics
from ictc_o3.electrostatics.api import configure_electrostatics
from ictc_o3.electrostatics.api import mesh_spread
from ictc_o3.electrostatics.api import mesh_gather
from ictc_o3.electrostatics.api import mesh_shape_for_cell

__all__ = [
    "EdgeTensorProduct", "GraphMessage", "PackedGraphMessage", "direction_harmonics",
    "create_symmetric_contraction", "configure_symmetric_contraction", "channel_linear", "normalized_silu",
    "create_model", "check_model", "configure_model", "energy_and_forces", "prepare_for_export",
    "MeshElectrostatics", "configure_electrostatics", "mesh_spread", "mesh_gather", "mesh_shape_for_cell",
]
