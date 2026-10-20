# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Supported high-performance API; see docs/HP_API.md for full usage.

Importing this namespace never builds CUDA extensions or selects a model plan.
"""
from .operators import (
    EdgeTensorProduct, GraphMessage, PackedGraphMessage, channel_linear,
    configure_symmetric_contraction, create_symmetric_contraction, direction_harmonics, normalized_silu,
)
from .runtime import check_model, configure_model, create_model, energy_and_forces, prepare_for_export

__all__ = [
    "EdgeTensorProduct", "GraphMessage", "PackedGraphMessage", "direction_harmonics",
    "create_symmetric_contraction", "configure_symmetric_contraction", "channel_linear", "normalized_silu",
    "create_model", "check_model", "configure_model", "energy_and_forces", "prepare_for_export",
]
