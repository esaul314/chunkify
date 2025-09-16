from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from pdf_chunker.framework import Artifact


_EMPTY_ROWS: tuple[dict[str, Any], ...] = ()


def _rows_from_items(items: Any) -> Iterable[dict[str, Any]]:
    """Return iterable rows when ``items`` already yields dictionaries."""

    if isinstance(items, list):
        return items
    if isinstance(items, Iterable) and not isinstance(items, (str, bytes)):
        return items
    return _EMPTY_ROWS


def _rows(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield rows when payload is a list or mapping exposing ``items``."""

    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return _rows_from_items(payload.get("items"))
    return _EMPTY_ROWS


def _copy_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow metadata copy without mutating ``meta``."""
    return {k: v for k, v in meta.items()}


def _maybe_drop_meta(rows: Iterable[dict[str, Any]], drop: bool) -> Iterable[dict[str, Any]]:
    """Skip empty metadata or drop it entirely when requested."""

    meta_key = os.getenv("PDF_CHUNKER_JSONL_META_KEY", "metadata")

    def _row(r: dict[str, Any]) -> dict[str, Any]:
        text = r.get("text", "")
        meta = r.get(meta_key)
        base = {"text": text}
        return base | ({meta_key: _copy_meta(meta)} if (meta and not drop) else {})

    return (_row(r) for r in rows)


def _serialize(rows: Iterable[dict[str, Any]]) -> Iterator[str]:
    """Serialize dictionaries to JSON lines."""
    return (json.dumps(r, ensure_ascii=False) for r in rows)


def _write(path: str, lines: Iterable[str]) -> None:
    """Write ``lines`` to ``path`` with trailing newlines."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
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
