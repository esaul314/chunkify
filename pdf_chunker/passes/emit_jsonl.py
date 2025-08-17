from __future__ import annotations

from typing import Any, Dict, List

from pdf_chunker.framework import Artifact, register

Row = Dict[str, Any]
Doc = Dict[str, Any]


def _row(item: Dict[str, Any]) -> Row:
    return {"text": item.get("text", ""), "meta": item.get("meta", {})}


def _rows(doc: Doc) -> List[Row]:
    return [_row(i) for i in doc.get("items", []) if i.get("text")]


def _update_meta(meta: Dict[str, Any] | None, count: int) -> Dict[str, Any]:
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
