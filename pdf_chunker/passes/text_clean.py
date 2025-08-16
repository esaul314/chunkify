from __future__ import annotations

from typing import Any, Dict

from pdf_chunker.framework import Artifact, register


def _clean_text(text: str) -> str:
    from pdf_chunker.text_cleaning import clean_text  # local import avoids heavy deps at import time

    return clean_text(text)


def _clean_block(block: Dict[str, Any]) -> Dict[str, Any]:
    return {**block, "text": _clean_text(block.get("text", ""))}


def _clean_page(page: Dict[str, Any]) -> Dict[str, Any]:
    return {**page, "blocks": [_clean_block(b) for b in page.get("blocks", [])]}


def _clean_page_blocks(doc: Dict[str, Any]) -> Dict[str, Any]:
    if doc.get("type") != "page_blocks":
        return doc
    return {**doc, "pages": [_clean_page(p) for p in doc.get("pages", [])]}


class _TextCleanPass:
    name = "text_clean"
    input_type = dict  # expects {"type": "page_blocks", ...}
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload
        out = _clean_page_blocks(doc) if isinstance(doc, dict) else doc
        meta = dict(a.meta or {})
        meta.setdefault("metrics", {}).setdefault("text_clean", {})["normalized"] = True
        return Artifact(payload=out, meta=meta)


text_clean = register(_TextCleanPass())
