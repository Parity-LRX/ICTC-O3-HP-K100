"""Compatibility alias for released prepared-kernel imports."""
from importlib import import_module
import sys

_implementation = import_module("mace_ictc.runtime.deployment_sparse")
sys.modules[__name__] = _implementation
