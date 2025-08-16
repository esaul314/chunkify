from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pdf_chunker.framework import Artifact, register

Block = dict[str, Any]
Doc = dict[str, Any]


def _iter_text(doc: Doc) -> Iterable[str]:
    return (
        block.get("text", "") for page in doc.get("pages", []) for block in page.get("blocks", [])
    )


def _split(doc: Doc) -> list[dict[str, str]]:
    from pdf_chunker.splitter import semantic_chunker  # local import to avoid heavy deps

    text = "\n".join(_iter_text(doc))
    return [{"text": chunk} for chunk in semantic_chunker(text)]


def _update_meta(meta: dict[str, Any] | None, count: int) -> dict[str, Any]:
    base = dict(meta or {})
    base.setdefault("metrics", {}).setdefault("split_semantic", {})["chunks"] = count
    return base


class _SplitSemanticPass:
    name = "split_semantic"
    input_type = dict
    output_type = list

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload
        if not isinstance(doc, dict) or doc.get("type") != "page_blocks":
            return a
        chunks = _split(doc)
        meta = _update_meta(a.meta, len(chunks))
        return Artifact(payload=chunks, meta=meta)


split_semantic = register(_SplitSemanticPass())
