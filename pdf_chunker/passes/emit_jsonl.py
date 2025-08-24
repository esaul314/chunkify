from __future__ import annotations

from typing import Any

from pdf_chunker.framework import Artifact, register

Row = dict[str, Any]
Doc = dict[str, Any]


def _row(item: dict[str, Any]) -> Row:
    base = {"text": item.get("text", "")}
    return base | ({"meta": item["meta"]} if "meta" in item else {})


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
