# Auto-register passes on package import (e.g., when importing any submodule)
from . import passes  # noqa: F401

__all__: list[str] = []
