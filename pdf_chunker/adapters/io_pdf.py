"""PDF IO adapters.

This module delineates the boundary for reading and describing PDF
resources without entangling higher-level parsing logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _normalize(path: str) -> Path:
    """Return an absolute ``Path`` for ``path``.

    This helper centralizes path normalization so both public functions
    remain side-effect free and easy to compose.
    """

    return Path(path).expanduser().resolve()


def read(path: str, exclude_pages: Optional[str] = None) -> bytes:
    """Read raw bytes from ``path``.

    Parameters
    ----------
    path:
        PDF file location.
    exclude_pages:
        Unused placeholder preserving the existing call signature.

    Returns
    -------
    bytes
        Raw byte content of the PDF file.
    """

    return _normalize(path).read_bytes()


def describe(path: str) -> dict[str, str]:
    """Describe a PDF resource without inspecting its contents."""

    normalized = _normalize(path)
    return {"type": "pdf_document", "source_path": str(normalized)}
