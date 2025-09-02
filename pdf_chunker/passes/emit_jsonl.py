from __future__ import annotations

import os
import re
from typing import Any

from pdf_chunker.framework import Artifact, register

Row = dict[str, Any]
Doc = dict[str, Any]


def _metadata_key() -> str:
    return os.getenv("PDF_CHUNKER_JSONL_META_KEY", "metadata")


def _compat_chunk_id(chunk_id: str) -> str:
    match = re.search(r"_p(\d+)_c", chunk_id)
    if not match:
        return chunk_id
    page = max(int(match.group(1)) - 1, 0)
    return f"{chunk_id[: match.start(1)]}{page}{chunk_id[match.end(1):]}"


def _row(item: dict[str, Any]) -> Row:
    meta_key = _metadata_key()
    base = {"text": item.get("text", "")}
    meta = item.get("meta")
    if not meta:
        return base
    chunk_id = meta.get("chunk_id")
    meta = {**meta, **({"chunk_id": _compat_chunk_id(chunk_id)} if chunk_id else {})}
    return base | {meta_key: meta}


def _rows(doc: Doc) -> list[Row]:
    return [_row(i) for i in doc.get("items", []) if i.get("text")]


def _update_meta(meta: dict[str, Any] | None, count: int) -> dict[str, Any]:
    base = dict(meta or {})
    base.setdefault("metrics", {}).setdefault("emit_jsonl", {})["rows"] = count
    return base


class _EmitJsonlPass:
    name = "emit_jsonl"
    input_type = dict
    output_type = list

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload if isinstance(a.payload, dict) else {}
        rows = _rows(doc) if doc.get("type") == "chunks" else []
        meta = _update_meta(a.meta, len(rows))
        return Artifact(payload=rows, meta=meta)


emit_jsonl = register(_EmitJsonlPass())
