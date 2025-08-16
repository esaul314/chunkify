from __future__ import annotations

from typing import Any, Dict, Mapping

from pdf_chunker.framework import Artifact, register


def _default_doc(src: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "page_blocks",
        "source_path": src.get("source_path", "<memory>"),
        "pages": [{"page": 1, "blocks": [{"text": "<stub pdf_parse>"}]}],
    }


def _update_meta(meta: Dict[str, Any] | None, pages: int) -> Dict[str, Any]:
    base = dict(meta or {})
    metrics = base.setdefault("metrics", {})
    metrics.setdefault("pdf_parse", {})["pages"] = pages
    return base


class _PdfParsePass:
    """
    Minimal stub: turn any payload into a simple page_blocks dict.
    Pure function; no side effects.
    """

    name = "pdf_parse"
    input_type = dict
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        src = a.payload if isinstance(a.payload, dict) else {}
        doc = _default_doc(src)
        meta = _update_meta(a.meta, len(doc["pages"]))
        return Artifact(payload=doc, meta=meta)


pdf_parse = register(_PdfParsePass())
