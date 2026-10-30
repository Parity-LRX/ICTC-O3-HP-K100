# SPDX-License-Identifier: MIT
"""FP64 energy reductions for eager native and adapted foundation models."""
from contextlib import contextmanager
import torch


def _energy_output_fp64(module, args, value):
    """Promote each atomic energy term before layer and graph reductions."""
    return value.to(dtype=torch.float64)


@contextmanager
def _native_energy_fp64(model):
    # Temporary hooks preserve canonical parameters, state keys and exports.
    handles = []
    try:
        modules = [model.atomic_energies_fn, *model.readouts]
        if hasattr(model, "pair_repulsion_fn"):
            modules.append(model.pair_repulsion_fn)
        for module in modules:
            handles.append(module.register_forward_hook(_energy_output_fp64))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def foundation_forward_fp64(model, data, **options):
    """Eager native/legacy foundation forward with FP64 energy reductions.

    Feature and readout GEMMs keep their original dtype. Each atomic energy
    term is promoted before summing interaction layers, applying scale/shift
    and summing atoms. An existing FP32 graph sum is never merely upcast.
    Hooks are scoped to this call and are absent from exported native models.
    """
    native = getattr(model, "native", model)
    with _native_energy_fp64(native):
        result = model(data, **options)
    if result["interaction_energy"].dtype != torch.float64:
        raise RuntimeError("Foundation energy reduction did not execute in FP64")
    return result
