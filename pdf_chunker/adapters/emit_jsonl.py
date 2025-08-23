from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pdf_chunker.framework import Artifact


def _rows(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield rows when payload is a list of dictionaries."""
    return payload if isinstance(payload, list) else []


def _maybe_drop_meta(
    rows: Iterable[dict[str, Any]], drop: bool
) -> Iterable[dict[str, Any]]:
    """Optionally remove metadata, keeping legacy defaults when present."""

    def _row(r: dict[str, Any]) -> dict[str, Any]:
        text = r.get("text", "")
        if drop:
            return {"text": text}
        meta = r.get("meta", {}).copy()
        if isinstance(meta.get("utterance_type"), dict) and not meta["utterance_type"]:
            meta["utterance_type"] = {"classification": "error", "tags": []}
        return {"text": text, "meta": meta}

    return (_row(r) for r in rows)


def _serialize(rows: Iterable[dict[str, Any]]) -> Iterator[str]:
    """Serialize dictionaries to JSON lines."""
    return (json.dumps(r, ensure_ascii=False) for r in rows)


def _write(path: str, lines: Iterable[str]) -> None:
    """Write ``lines`` to ``path`` with trailing newlines."""
    with Path(path).open("w", encoding="utf-8") as f:
        f.writelines(f"{line}\n" for line in lines)


def write(rows: Iterable[dict[str, Any]], path: str | None) -> None:
    """Write ``rows`` to JSONL at ``path`` when provided."""
    if not path:
        return
    _write(path, _serialize(rows))


def maybe_write(
    artifact: Artifact, options: dict[str, Any], timings: dict[str, float] | None = None
) -> None:
    """Write artifact payload to JSONL if ``output_path`` is specified."""
    rows = _maybe_drop_meta(_rows(artifact.payload), options.get("drop_meta", False))
    write(rows, options.get("output_path"))
