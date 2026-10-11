"""Detect end markers within the last three pages and report copied metrics."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from pdf_chunker.framework import Artifact, register
from pdf_chunker.passes.metrics import with_metrics

Block = dict[str, Any]
Page = dict[str, Any]

DOT_LEADER_RE = re.compile(r"(?:\.\s*){3,}")
DOC_END_RE = re.compile(r"^(?:THE\s+)?END(?:\s+OF\s+DOCUMENT)?\.?$")

logger = logging.getLogger(__name__)


def _is_page_number(text: str) -> bool:
    return text.strip().isdigit()


def _is_dot_leader(text: str) -> bool:
    return bool(DOT_LEADER_RE.fullmatch(text.strip()))


def _is_artifact(text: str) -> bool:
    return _is_page_number(text) or _is_dot_leader(text)


def _is_doc_end_page(blocks: Iterable[Block]) -> bool:
    texts = [
        b.get("text", "").strip() for b in blocks if b.get("text") and not _is_artifact(b["text"])
    ]
    return len(texts) == 1 and bool(DOC_END_RE.fullmatch(texts[0]))


def _doc_end_index(pages: list[Page]) -> int | None:
    """Select the latest qualifying marker, restricted to the final three pages."""
    for offset, page in enumerate(reversed(pages[-3:]), 1):
        idx = len(pages) - offset
        if _is_doc_end_page(page.get("blocks", [])):
            return idx
    return None


def _truncate_pages(pages: list[Page]) -> tuple[list[Page], int]:
    idx = _doc_end_index(pages)
    if idx is None:
        logger.debug("no doc-end marker found")
        return pages, 0
    tail = len(pages) - idx - 1
    logger.info(
        "truncating %d trailing page(s) after marker at page %d",
        tail,
        idx + 1,
    )
    return pages[: idx + 1], tail


class _DetectDocEndPass:
    name = "detect_doc_end"
    input_type = object
    output_type = object

    def __call__(self, a: Artifact[object]) -> Artifact[object]:
        payload = a.payload
        if not isinstance(payload, dict) or payload.get("type") != "page_blocks":
            return a
        pages, truncated = _truncate_pages(payload.get("pages", []))
        meta = with_metrics(a.meta, pass_name=self.name, updates={"truncated_pages": truncated})
        return Artifact(payload={**payload, "pages": pages}, meta=meta)


detect_doc_end = register(_DetectDocEndPass())
