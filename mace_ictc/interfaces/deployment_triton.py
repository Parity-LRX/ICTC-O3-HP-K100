"""Compatibility alias for released runtime imports and pickles."""
from importlib import import_module
import sys

_implementation = import_module("mace_ictc.runtime.export")
sys.modules[__name__] = _implementation
