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
