from __future__ import annotations

import os
import re
import json
from typing import Any, Iterable

from pdf_chunker.framework import Artifact, register
from pdf_chunker.utils import _truncate_chunk

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


def _max_chars() -> int:
    return int(os.getenv("PDF_CHUNKER_JSONL_MAX_CHARS", "8000"))


def _split(text: str, limit: int) -> list[str]:
    """Yield ``text`` slices no longer than ``limit`` using soft boundaries."""

    def parts(t: str) -> Iterable[str]:
        while t:
            chunk = _truncate_chunk(t, limit)
            yield chunk
            t = t[len(chunk) :].lstrip()

    return list(parts(text))


def _coherent(text: str, min_chars: int = 40) -> bool:
    stripped = text.strip()
    return (
        len(stripped) >= min_chars
        and re.match(r"^[\"'(]*[A-Z0-9]", stripped) is not None
        and re.search(r"[.!?][\"')\]]*$", stripped) is not None
    )


def _merge_overlap(prev: str, curr: str, max_len: int = 80) -> str:
    """Merge ``curr`` into ``prev`` removing duplicated prefix."""

    length = min(len(prev), len(curr), max_len)
    overlap = next((i for i in range(length, 0, -1) if prev.endswith(curr[:i])), 0)
    if overlap:
        trimmed = curr[overlap:].lstrip()
        return f"{prev}\n\n{trimmed}" if trimmed else prev

    prefix = curr.split("\n\n", 1)[0]
    if prefix and prefix in prev:
        trimmed = curr[len(prefix) :].lstrip()
        return f"{prev}\n\n{trimmed}" if trimmed else prev

    return f"{prev}\n\n{curr}"


def _coalesce(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge neighboring items while preserving semantic coherence."""
    buf: dict[str, Any] | None = None
    out: list[dict[str, Any]] = []

    for item in items:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        current = {**item, "text": text}
        if _coherent(text):
            if buf:
                merged = {**current, "text": f"{buf['text']}\n\n{text}"}
                out.append(merged if _coherent(merged["text"]) else current)
                buf = None
            else:
                out.append(current)
        elif out:
            prev = out[-1]
            merged_text = _merge_overlap(prev["text"], text)
            if _coherent(merged_text):
                out[-1] = {**prev, "text": merged_text}
        else:
            buf = current if not buf else {**buf, "text": f"{buf['text']}\n\n{text}"}

    if buf and _coherent(buf["text"]):
        out.append(buf)
    return out


def _rows_from_item(item: dict[str, Any]) -> list[Row]:
    meta_key = _metadata_key()
    max_chars = _max_chars()
    meta: dict[str, Any] = item.get("meta") or {}
    chunk_id = meta.get("chunk_id")
    if chunk_id:
        meta = {**meta, "chunk_id": _compat_chunk_id(chunk_id)}
    base_meta = {meta_key: meta} if meta else {}
    overhead = len(json.dumps({"text": "", **base_meta}, ensure_ascii=False)) - 2
    avail = max(max_chars - overhead, 0)
    pieces = _split(item.get("text", ""), avail)

    def build(idx_piece: tuple[int, str]) -> Row:
        idx, piece = idx_piece
        if meta and len(pieces) > 1:
            meta_part = {meta_key: {**meta, "chunk_part": idx}}
        else:
            meta_part = base_meta
        row = {"text": piece, **meta_part}
        while len(json.dumps(row, ensure_ascii=False)) > max_chars:
            allowed = avail - (len(json.dumps(row, ensure_ascii=False)) - max_chars)
            piece = _truncate_chunk(piece[:allowed], allowed)
            row = {"text": piece, **meta_part}
        return row

    return [build(x) for x in enumerate(pieces)]


def _rows(doc: Doc) -> list[Row]:
    items = _coalesce(doc.get("items", []))
    return [r for i in items for r in _rows_from_item(i)]


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
