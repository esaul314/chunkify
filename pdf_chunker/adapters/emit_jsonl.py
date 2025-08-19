from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pdf_chunker.framework import Artifact


def _rows(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield rows when payload is a list of dictionaries."""
    return payload if isinstance(payload, list) else []


def _serialize(rows: Iterable[dict[str, Any]]) -> Iterator[str]:
    """Serialize dictionaries to JSON lines."""
    return (json.dumps(r, ensure_ascii=False) for r in rows)


def _write(path: str, lines: Iterable[str]) -> None:
    """Write ``lines`` to ``path`` with trailing newlines."""
    with Path(path).open("w", encoding="utf-8") as f:
        f.writelines(f"{line}\n" for line in lines)


def maybe_write(
    artifact: Artifact, options: dict[str, Any], timings: dict[str, float] | None = None
) -> None:
    """Write artifact payload to JSONL if ``output_path`` is specified."""
    out_path = options.get("output_path")
    if not out_path:
        return

    lines = _serialize(_rows(artifact.payload))
    _write(out_path, lines)
