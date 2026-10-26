"""Python fallback for the optional ICTC TP extension.

This module makes ``import ictc_o3.backends.k100._C_ictd_tp`` succeed even when
the compiled extension has not been built for the current platform. Runtime
feature probes still report the extension as unavailable because the compiled
operator symbols are intentionally absent.
"""


def has_cuda() -> bool:
    """Report that the optional compiled extension is not available."""
    return False