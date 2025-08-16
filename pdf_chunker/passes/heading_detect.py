from __future__ import annotations

"""Heading detection pass.

Annotates ``page_blocks`` documents with heading metadata and derives a
hierarchical heading structure. Pure transform: receives a document dict and
returns an updated copy with metrics added to ``meta``.
"""

from functools import reduce
from typing import Any, Dict, Iterable, List, Tuple

from pdf_chunker.framework import Artifact, register
from pdf_chunker.heading_detection import (
    TRAILING_PUNCTUATION,
    _detect_heading_fallback,
    _estimate_heading_level,
    get_heading_hierarchy,
)


Block = Dict[str, Any]


def _is_heading(block: Block) -> bool:
    text = block.get("text", "").strip()
    is_declared = block.get("type") == "heading" and not text.endswith(
        TRAILING_PUNCTUATION
    )
    return bool(text) and (is_declared or _detect_heading_fallback(text))


def _annotate_block(block: Block, current: str | None) -> Tuple[str | None, Block]:
    text = block.get("text", "").strip()
    if _is_heading(block):
        level = _estimate_heading_level(text)
        return (
            text,
            {
                **block,
                "text": text,
                "type": "heading",
                "is_heading": True,
                "heading_level": level,
                "heading_source": "heuristic",
            },
        )

    enriched = {**block, "text": text, "is_heading": False}
    if current:
        enriched["section_heading"] = current
    if enriched.get("type") == "heading":
        enriched["type"] = "paragraph"
    return current, enriched


def _annotate_blocks(blocks: Iterable[Block]) -> List[Block]:
    def step(state: Tuple[str | None, List[Block]], block: Block) -> Tuple[str | None, List[Block]]:
        heading, acc = state
        new_heading, annotated = _annotate_block(block, heading)
        return new_heading, [*acc, annotated]

    return reduce(step, blocks, (None, []))[1]


def _annotate_doc(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Block]]:
    pages = [
        {**p, "blocks": _annotate_blocks(p.get("blocks", []))}
        for p in doc.get("pages", [])
    ]
    all_blocks = [b for p in pages for b in p["blocks"]]
    hierarchy = get_heading_hierarchy(all_blocks)
    return {**doc, "pages": pages}, hierarchy


class _HeadingDetectPass:
    name = "heading_detect"
    input_type = dict
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload
        if not isinstance(doc, dict) or doc.get("type") != "page_blocks":
            return a

        updated, hierarchy = _annotate_doc(doc)
        meta = dict(a.meta or {})
        metrics = meta.setdefault("metrics", {}).setdefault("heading_detect", {})
        metrics["headings"] = len(hierarchy)
        meta["heading_hierarchy"] = hierarchy
        return Artifact(payload=updated, meta=meta)


heading_detect = register(_HeadingDetectPass())

