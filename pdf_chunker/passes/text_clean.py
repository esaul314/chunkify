from __future__ import annotations

from typing import Any, Dict

from pdf_chunker.framework import Artifact, register


def _clean_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new block with normalized text."""
    from pdf_chunker import text_cleaning

    text = block.get("text", "")
    return {**block, "text": text_cleaning.clean_text(text)}


def _clean_page(page: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
    """Clean all blocks in ``page`` and return the block count."""
    blocks = [_clean_block(b) for b in page.get("blocks", [])]
    return {**page, "blocks": blocks}, len(blocks)


def _clean_doc(doc: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
    """Clean document pages and aggregate block metrics."""
    pages_with_counts = [_clean_page(p) for p in doc.get("pages", [])]
    pages = [p for p, _ in pages_with_counts]
    blocks = sum(c for _, c in pages_with_counts)
    return {**doc, "pages": pages}, blocks


class _TextCleanPass:
    name = "text_clean"
    input_type = object
    output_type = object

    def __call__(self, a: Artifact) -> Artifact:
        payload = a.payload
        block_count: int | None = None

        if isinstance(payload, str):
            from pdf_chunker.text_cleaning import _clean_text_impl

            cleaned = _clean_text_impl(payload)
        elif isinstance(payload, dict) and payload.get("type") == "page_blocks":
            cleaned, block_count = _clean_doc(payload)
        else:
            return a

        meta = dict(a.meta or {})
        metrics = meta.setdefault("metrics", {})
        metrics["normalized"] = True
        if block_count is not None:
            metrics.setdefault("text_clean", {})["blocks"] = block_count
        return Artifact(payload=cleaned, meta=meta)


text_clean = register(_TextCleanPass())
