from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from pdf_chunker.framework import Artifact


def _rows(payload: Any) -> Iterable[Dict[str, Any]]:
    """Yield rows when payload is a list of dictionaries."""
    return payload if isinstance(payload, list) else []


def _serialize(rows: Iterable[Dict[str, Any]]) -> Iterator[str]:
    """Serialize dictionaries to JSON lines."""
    return (json.dumps(r, ensure_ascii=False) for r in rows)


def _write(path: str, lines: Iterable[str]) -> None:
    """Write iterable of lines to ``path`` with trailing newlines."""
    with Path(path).open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def maybe_write(
    artifact: Artifact, options: Dict[str, Any], timings: Dict[str, float] | None = None
) -> None:
    """Write artifact payload to JSONL if ``output_path`` is specified."""
    out_path = options.get("output_path")
    if not out_path:
        return

    lines = _serialize(_rows(artifact.payload))
    _write(out_path, lines)
