from __future__ import annotations

"""Heading detection pass.

Pure transform that enriches blocks with heading metadata and derives a
heading hierarchy.  The pass operates on ``list`` structures to remain
composable within the functional pipeline.
"""

from functools import reduce
from typing import Any, Dict, Iterable, List, Tuple

from pdf_chunker.framework import Artifact, register
from pdf_chunker.heading_detection import (
    detect_headings_hybrid,
    get_heading_hierarchy,
)


Block = Dict[str, Any]


def enhance_blocks_with_heading_metadata(
    blocks: Iterable[Block], extraction_method: str = "unknown"
) -> List[Block]:
    """Return blocks annotated with heading metadata."""

    blocks_list = list(blocks)
    heading_map = {
        h["text"].strip().lower(): h
        for h in detect_headings_hybrid(blocks_list, extraction_method)
    }

    def step(
        state: Tuple[str | None, List[Block]], block: Block
    ) -> Tuple[str | None, List[Block]]:
        current, acc = state
        text = block.get("text", "").strip()
        key = text.lower()
        info = heading_map.get(key)
        if info:
            enriched = {
                **block,
                "text": text,
                "is_heading": True,
                "heading_level": info["level"],
                "heading_source": info["source"],
                "type": "heading",
            }
            return text, [*acc, enriched]

        enriched = {
            **block,
            "text": text,
            "is_heading": False,
            **({"section_heading": current} if current else {}),
        }
        if enriched.get("type") == "heading":
            enriched["type"] = "paragraph"
        return current, [*acc, enriched]

    return reduce(step, blocks_list, (None, []))[1]


class _HeadingDetectPass:
    name = "heading_detect"
    input_type = list
    output_type = list

    def __call__(self, a: Artifact) -> Artifact:
        blocks = a.payload
        if not isinstance(blocks, list):
            return a

        extraction_method = (a.meta or {}).get("extraction_method", "unknown")
        enhanced = enhance_blocks_with_heading_metadata(blocks, extraction_method)
        hierarchy = get_heading_hierarchy(enhanced)

        meta = dict(a.meta or {})
        metrics = meta.setdefault("metrics", {}).setdefault("heading_detect", {})
        metrics["headings"] = len(hierarchy)
        meta["heading_hierarchy"] = hierarchy
        return Artifact(payload=enhanced, meta=meta)


heading_detect = register(_HeadingDetectPass())

