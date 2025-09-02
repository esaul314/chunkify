from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

import logging

from pdf_chunker.framework import Artifact, register

Block = Dict[str, Any]
Page = Dict[str, Any]

DOT_LEADER_RE = re.compile(r"(?:\.\s*){3,}")
DOC_END_RE = re.compile(r"(?i)^(?:the\s+)?end(?:\s+of\s+document)?\.?$")

logger = logging.getLogger(__name__)


def _is_page_number(text: str) -> bool:
    return text.strip().isdigit()


def _is_dot_leader(text: str) -> bool:
    return bool(DOT_LEADER_RE.fullmatch(text.strip()))


def _is_artifact(text: str) -> bool:
    return _is_page_number(text) or _is_dot_leader(text)


def _is_doc_end_page(blocks: Iterable[Block]) -> bool:
    texts = [
        b.get("text", "").strip()
        for b in blocks
        if b.get("text") and not _is_artifact(b["text"])
    ]
    return len(texts) == 1 and bool(DOC_END_RE.fullmatch(texts[0]))


def _doc_end_index(pages: List[Page]) -> int | None:
    total = len(pages)
    for i, page in enumerate(pages):
        if _is_doc_end_page(page.get("blocks", [])):
            tail = total - i - 1
            if _should_truncate(total, tail):
                return i
    return None


def _should_truncate(total: int, tail: int) -> bool:
    return 0 < tail <= 2


def _truncate_pages(pages: List[Page]) -> Tuple[List[Page], int]:
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

    def __call__(self, a: Artifact) -> Artifact:
        payload = a.payload
        if not isinstance(payload, dict) or payload.get("type") != "page_blocks":
            return a
        pages, truncated = _truncate_pages(payload.get("pages", []))
        meta = dict(a.meta or {})
        metrics = meta.setdefault("metrics", {})
        metrics.setdefault("detect_doc_end", {})["truncated_pages"] = truncated
        return Artifact(payload={**payload, "pages": pages}, meta=meta)


detect_doc_end = register(_DetectDocEndPass())
