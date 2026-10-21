# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compatibility imports for the public ``ictc_o3`` operator API.

Keep the original paths resolvable for existing callers and serialized modules.
The implementation and new operator imports live in ``ictc_o3.operators``.
"""
from ictc_o3.operators import (
    EdgeTensorProduct,
    GraphMessage,
    PackedGraphMessage,
    channel_linear,
    configure_symmetric_contraction,
    create_symmetric_contraction,
    direction_harmonics,
    normalized_silu,
    _Edge,
    _EdgeAdjoint,
    _degree,
    _eager,
    _operand,
    _positive_integer,
    _tp_coverage,
)

__all__ = [
    "EdgeTensorProduct",
    "GraphMessage",
    "PackedGraphMessage",
    "direction_harmonics",
    "create_symmetric_contraction",
    "configure_symmetric_contraction",
    "channel_linear",
    "normalized_silu",
]
