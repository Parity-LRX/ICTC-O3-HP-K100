# SPDX-License-Identifier: MIT
"""Mesh configuration shared by model construction and checkpoint metadata."""
from numbers import Integral


def normalize_mesh_size(value):
    """Keep legacy integers, or validate a three-axis mesh configuration."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        if value > 0:
            return int(value)
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        if all(isinstance(v, Integral) and not isinstance(v, bool) and v > 0 for v in value):
            return tuple(int(v) for v in value)
    raise ValueError('mesh_size must be a positive integer or three positive integers')


def mesh_shape(value):
    value = normalize_mesh_size(value)
    return (value, value, value) if isinstance(value, int) else value


def parse_mesh_size(text):
    """CLI syntax: M or Mx,My,Mz (also accepts Mx x My x Mz)."""
    try:
        values = [int(v.strip()) for v in text.lower().replace('x', ',').split(',')]
        return normalize_mesh_size(values[0] if len(values) == 1 else values)
    except (TypeError, ValueError) as exc:
        raise ValueError('Expected a positive M or three positive sizes Mx,My,Mz') from exc
