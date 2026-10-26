"""Compatibility alias for released import and pickle paths."""
from importlib import import_module
import sys

_implementation = import_module("ictc_o3.ops.polynomial_gemm")
sys.modules[__name__] = _implementation
