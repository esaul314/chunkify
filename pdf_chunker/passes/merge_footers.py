"""Merge short trailing footer lines without mutating page blocks or metrics."""

from __future__ import annotations

from itertools import takewhile
from typing import Any

from pdf_chunker.framework import Artifact, register
from pdf_chunker.passes.metrics import with_metrics

Block = dict[str, Any]
MAX_LEN = 40
MAX_LINES = 3


def _merge_trailing_short_lines(blocks: list[Block]) -> tuple[list[Block], int]:
    """Merge short trailing lines into a single footer block."""
    tail = list(
        takewhile(
            lambda b: len(b.get("text", "")) <= MAX_LEN,
            reversed(blocks),
        )
    )
    if len(tail) == len(blocks):
        return blocks, 0
    if 1 < len(tail) <= MAX_LINES:
        keep = blocks[: len(blocks) - len(tail)]
        merged_text = " ".join(b.get("text", "") for b in reversed(tail))
        merged_block = {**tail[-1], "text": merged_text}
        return [*keep, merged_block], len(tail) - 1
    return blocks, 0


def _merge_doc(doc: dict[str, Any]) -> tuple[dict[str, Any], int]:
    pages, merged_total = [], 0
    for page in doc.get("pages", []):
        merged, count = _merge_trailing_short_lines(page.get("blocks", []))
        pages.append({**page, "blocks": merged})
        merged_total += count
    return {**doc, "pages": pages}, merged_total


class _MergeFootersPass:
    name = "merge_footers"
    input_type = object
    output_type = object

    def __call__(self, a: Artifact[object]) -> Artifact[object]:
        payload = a.payload
        if not isinstance(payload, dict) or payload.get("type") != "page_blocks":
            return a
        merged_doc, merged_count = _merge_doc(payload)
        meta = with_metrics(a.meta, pass_name=self.name, updates={"merged_lines": merged_count})
        return Artifact(payload=merged_doc, meta=meta)


merge_footers = register(_MergeFootersPass())
