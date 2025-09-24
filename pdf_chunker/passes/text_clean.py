from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Dict, cast

from pdf_chunker.framework import Artifact, register


def _style_of(span: Any) -> str | None:
    """Return the style tag for ``span`` regardless of representation."""

    if hasattr(span, "style"):
        return getattr(span, "style")
    if isinstance(span, Mapping):
        return span.get("style")
    return None


def _inline_style_metrics(pages: list[Dict[str, Any]]) -> dict[str, Any]:
    """Aggregate inline-style coverage and tag counts for ``pages``."""

    blocks = [
        block
        for page in pages
        for block in page.get("blocks", [])
    ]
    total_blocks = len(blocks)
    if total_blocks == 0:
        return {
            "inline_style_block_ratio": 0.0,
            "inline_style_tag_counts": {},
        }

    styled_blocks = [block for block in blocks if block.get("inline_styles")]
    tag_counts = Counter(
        style
        for block in styled_blocks
        for style in (
            _style_of(span)
            for span in block.get("inline_styles") or ()
        )
        if style is not None
    )
    return {
        "inline_style_block_ratio": len(styled_blocks) / total_blocks,
        "inline_style_tag_counts": dict(tag_counts),
    }


def _clean_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new block with normalized text."""
    from pdf_chunker import text_cleaning
    from pdf_chunker.inline_styles import (
        build_index_map,
        build_index_remapper,
        normalize_spans,
    )

    text = block.get("text", "")
    cleaned_text = text_cleaning.clean_text(text)

    original_styles = block.get("inline_styles")
    remapped_styles = original_styles
    if original_styles:
        remapper = build_index_remapper(build_index_map(text, cleaned_text))
        normalized = normalize_spans(original_styles, len(cleaned_text), remapper)
        remapped_styles = list(normalized)

    return {
        **block,
        "text": cleaned_text,
        "inline_styles": remapped_styles,
    }


def _clean_page(page: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
    """Clean all blocks in ``page`` and return the block count."""
    blocks = [_clean_block(b) for b in page.get("blocks", [])]
    return {**page, "blocks": blocks}, len(blocks)


def _clean_doc(doc: Dict[str, Any]) -> tuple[Dict[str, Any], int, dict[str, Any]]:
    """Clean document pages and aggregate block metrics."""
    pages_with_counts = [_clean_page(p) for p in doc.get("pages", [])]
    pages = [p for p, _ in pages_with_counts]
    blocks = sum(c for _, c in pages_with_counts)
    style_metrics = _inline_style_metrics(pages)
    return {**doc, "pages": pages}, blocks, style_metrics


class _TextCleanPass:
    name = "text_clean"
    input_type = object
    output_type = object

    def __call__(self, a: Artifact) -> Artifact:
        payload = a.payload
        block_count: int | None = None
        cleaned: str | Dict[str, Any]

        if isinstance(payload, str):
            from pdf_chunker.text_cleaning import _clean_text_impl

            cleaned = _clean_text_impl(payload)
        elif isinstance(payload, dict) and payload.get("type") == "page_blocks":
            typed_payload = cast(Dict[str, Any], payload)
            cleaned, block_count, style_metrics = _clean_doc(typed_payload)
        else:
            return a

        meta = dict(a.meta or {})
        metrics = meta.setdefault("metrics", {})
        metrics["normalized"] = True
        if block_count is not None:
            text_metrics = metrics.setdefault("text_clean", {})
            text_metrics["blocks"] = block_count
            text_metrics.update(style_metrics)
        return Artifact(payload=cleaned, meta=meta)


text_clean = register(_TextCleanPass())
