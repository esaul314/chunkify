"""Public package API and pass registration."""

# Auto-register passes on package import (e.g., when importing any submodule)
from . import passes  # noqa: F401
from .config import PipelineSpec
from .core_new import convert

__all__ = ["PipelineSpec", "convert"]
