# Auto-register passes on package import (e.g., when importing any submodule)
import sys

print("DEBUG: pdf_chunker/__init__.py executed", file=sys.stderr)
from . import passes  # noqa: F401

__all__: list[str] = []
