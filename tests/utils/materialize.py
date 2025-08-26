"""Helpers for materializing encoded fixtures."""

from __future__ import annotations

import base64
from pathlib import Path


def materialize_base64(src: Path, dst_dir: Path, filename: str) -> Path:
    """Decode a base64-encoded file into *dst_dir* with *filename*.

    Returns the path to the materialized file.
    """

    data = base64.b64decode(src.read_text())
    target = dst_dir / filename
    target.write_bytes(data)
    return target
