from __future__ import annotations

from typing import Any, Dict

from pdf_chunker.framework import Artifact, register


def _clean_block(block: Dict[str, Any]) -> Dict[str, Any]:
    from pdf_chunker import text_cleaning

    return {**block, "text": text_cleaning.clean_text(block.get("text", ""))}


def _clean_page(page: Dict[str, Any]) -> Dict[str, Any]:
    blocks = (_clean_block(b) for b in page.get("blocks", []))
    return {**page, "blocks": list(blocks)}


def _clean_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    pages = (_clean_page(p) for p in doc.get("pages", []))
    return {**doc, "pages": list(pages)}


class _TextCleanPass:
    name = "text_clean"
    input_type = object
    output_type = object

    def __call__(self, a: Artifact) -> Artifact:
        payload = a.payload
        if isinstance(payload, str):
            from pdf_chunker.text_cleaning import _clean_text_impl

            cleaned = _clean_text_impl(payload)
        elif isinstance(payload, dict) and payload.get("type") == "page_blocks":
            cleaned = _clean_doc(payload)
        else:
            return a

        meta = dict(a.meta or {})
        meta.setdefault("metrics", {})["normalized"] = True
        return Artifact(payload=cleaned, meta=meta)


text_clean = register(_TextCleanPass())
