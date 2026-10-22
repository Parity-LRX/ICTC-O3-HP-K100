# SPDX-License-Identifier: MIT
"""Public Cartesian operator API, alongside the ``mace_ictc`` model package.

Importing this package does not import MACE models, initialize a GPU or build
extensions. Operators use the shared HP backend when called. See docs/O3_API.md.
"""
from .operators import (
    EdgeTensorProduct,
    GraphMessage,
    PackedGraphMessage,
    channel_linear,
    configure_symmetric_contraction,
    create_symmetric_contraction,
    direction_harmonics,
    normalized_silu,
)
from .electrostatics import MeshElectrostatics, mesh_spread, mesh_gather, mesh_shape_for_cell

__all__ = [
    "EdgeTensorProduct",
    "GraphMessage",
    "PackedGraphMessage",
    "direction_harmonics",
    "create_symmetric_contraction",
    "configure_symmetric_contraction",
    "channel_linear",
    "normalized_silu",
    "MeshElectrostatics",
    "mesh_spread",
    "mesh_gather",
    "mesh_shape_for_cell",
]
