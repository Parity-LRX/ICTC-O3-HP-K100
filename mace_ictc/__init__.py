"""
ICTC-O3-HP-K100 - HIP optimized operators and MACE-compatible models.

Standalone high-performance library. Stable HP operator and
whole-network entrypoints are exposed in ``mace_ictc.hp``. The package name
is retained for checkpoint compatibility; no O2U1 package is required.
"""

__version__ = "0.1.0"

import torch
if not hasattr(torch, "compiler"):
    class _CompilerShim:
        @staticmethod
        def disable(fn=None, recursive=True):  # type: ignore[no-untyped-def]
            if fn is None:
                def decorator(inner):  # type: ignore[no-untyped-def]
                    return inner

                return decorator
            return fn

    torch.compiler = _CompilerShim()  # type: ignore[attr-defined]
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])

# Import the baseline ICTC-MACE model + small shared helpers (guarded so version metadata
# is still importable when heavy deps are absent).
try:
    from mace_ictc.models import (
        PureCartesianICTDFix,
        MainNet,
        MainNet2,
        RMSELoss,
        RobustScalarWeightedSum,
    )

    __all__ = [
        "PureCartesianICTDFix",
        "MainNet",
        "MainNet2",
        "RMSELoss",
        "RobustScalarWeightedSum",
    ]
except ImportError:
    # If dependencies are not installed, just define version
    __all__ = []
