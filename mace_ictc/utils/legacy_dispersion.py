"""Validation of disabled dispersion fields in historical K100 checkpoints.

Dispersion implementations are excluded from the K100 distribution. These
helpers preserve metadata for models trained with dispersion disabled and reject
checkpoints that require a removed energy contribution.
"""

_MESSAGE = "Dispersion (pairwise C6/MBD) is not supported by this K100 distribution"


def normalize_dispersion_mode(*, long_range_dispersion=False, long_range_dispersion_mode=None):
    if long_range_dispersion or str(long_range_dispersion_mode or "none") != "none":
        raise ValueError(_MESSAGE)
    return "none"


def _require_disabled(mode):
    if str(mode or "none") != "none":
        raise ValueError(_MESSAGE)


def dispersion_mode_uses_canonical_edges(mode):
    _require_disabled(mode)
    return False


def dispersion_mode_uses_cutoff_edges(mode, *, mbd_operator_backend="edge_sparse"):
    _require_disabled(mode)
    return False


def dispersion_mode_needs_deployment_edges(mode, *, mbd_operator_backend="edge_sparse"):
    _require_disabled(mode)
    return False


def dispersion_deployment_graph_rule(mode, *, mbd_operator_backend="edge_sparse"):
    _require_disabled(mode)
    return "none"


def dispersion_training_graph_rule(mode, *, mbd_operator_backend="edge_sparse"):
    _require_disabled(mode)
    return "none"


def dispersion_train_deploy_graph_compatibility(mode, *, mbd_operator_backend="edge_sparse"):
    _require_disabled(mode)
    return "none"
