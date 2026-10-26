"""Compatibility alias for released model imports and full-model pickles."""
from importlib import import_module
import sys

_implementation = import_module("mace_ictc.models.mace.backbone")
sys.modules[__name__] = _implementation
