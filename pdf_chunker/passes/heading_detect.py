from __future__ import annotations

"""Heading detection pass using fallback heuristics.

Each block is annotated with heading metadata based on
``heading_detection._detect_heading_fallback``.  The implementation
leans on functional iteration to keep the transform stateless and
composable within the pipeline.
"""

from typing import Any, Dict, Iterable, List

from pdf_chunker.framework import Artifact, register
from pdf_chunker.heading_detection import (
    _detect_heading_fallback,
    _estimate_heading_level,
    get_heading_hierarchy,
)


Block = Dict[str, Any]


def _annotate(block: Block) -> Block:
    text = block.get("text", "").strip()
    is_heading = _detect_heading_fallback(text)
    enriched = {
        **block,
        "text": text,
        "is_heading": is_heading,
        "heading_level": _estimate_heading_level(text) if is_heading else None,
        "heading_source": "fallback" if is_heading else None,
    }
    if is_heading:
        enriched["type"] = "heading"
    elif enriched.get("type") == "heading":
        enriched["type"] = "paragraph"
    return enriched


def annotate_headings(blocks: Iterable[Block]) -> List[Block]:
    return [_annotate(b) for b in blocks]


class _HeadingDetectPass:
    name = "heading_detect"
    input_type = list
    output_type = list

    def __call__(self, a: Artifact) -> Artifact:
        blocks = a.payload
        if not isinstance(blocks, list):
            return a

        enhanced = annotate_headings(blocks)
        hierarchy = get_heading_hierarchy(enhanced)

        meta = dict(a.meta or {})
        metrics = meta.setdefault("metrics", {}).setdefault("heading_detect", {})
        metrics["headings"] = len(hierarchy)
        meta["heading_hierarchy"] = hierarchy
        return Artifact(payload=enhanced, meta=meta)


heading_detect = register(_HeadingDetectPass())
