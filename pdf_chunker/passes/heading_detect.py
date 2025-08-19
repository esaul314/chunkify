from __future__ import annotations

"""Heading detection pass using fallback heuristics.

Each block is annotated with heading metadata based on
``heading_detection._detect_heading_fallback``.  The implementation
leans on functional iteration to keep the transform stateless and
composable within the pipeline.
"""

from typing import Any, Dict, Iterable, List, Optional

from pdf_chunker.framework import Artifact, register
from pdf_chunker.heading_detection import (
    TRAILING_PUNCTUATION,
    _detect_heading_fallback,
    _estimate_heading_level,
    _has_heading_starter,
    get_heading_hierarchy,
)
import re


Block = Dict[str, Any]


def _estimate_threshold(text: str) -> Optional[int]:
    words = text.split()
    checks = (
        (len(words) <= 3 and not text.endswith(TRAILING_PUNCTUATION), 3),
        (text.isupper() and len(words) <= 8, 8),
        (
            text.istitle() and len(words) <= 10 and not text.endswith(TRAILING_PUNCTUATION),
            10,
        ),
        (
            _has_heading_starter(words)
            and len(words) <= 8
            and not text.endswith(TRAILING_PUNCTUATION),
            8,
        ),
        (
            len(words) >= 2
            and re.match(r"^[\d\.\-]+$", words[0])
            and len(words) <= 8
            and not text.endswith(TRAILING_PUNCTUATION),
            8,
        ),
    )
    return next((thr for cond, thr in checks if cond), None)


def _annotate(block: Block) -> Block:
    text = block.get("text", "").strip()
    is_heading = _detect_heading_fallback(text)
    threshold = _estimate_threshold(text) if is_heading else None
    enriched = {
        **block,
        "text": text,
        "is_heading": is_heading,
        "heading_level": _estimate_heading_level(text) if is_heading else None,
        "heading_threshold": threshold,
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
