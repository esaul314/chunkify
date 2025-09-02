from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from pdf_chunker.framework import Artifact, register

Block = Dict[str, Any]
Page = Dict[str, Any]

DOT_LEADER_RE = re.compile(r"(?:\.\s*){3,}")
DOC_END_RE = re.compile(r"(?i)^(?:the\s+)?end(?:\s+of\s+document)?\.?$")


def _is_page_number(text: str) -> bool:
    return text.strip().isdigit()


def _is_dot_leader(text: str) -> bool:
    return bool(DOT_LEADER_RE.fullmatch(text.strip()))


def _is_artifact(text: str) -> bool:
    return _is_page_number(text) or _is_dot_leader(text)


def _page_text(blocks: Iterable[Block]) -> str:
    return " ".join(
        b.get("text", "").strip()
        for b in blocks
        if b.get("text") and not _is_artifact(b["text"])
    ).strip()


def _is_doc_end_page(blocks: Iterable[Block]) -> bool:
    return bool(DOC_END_RE.fullmatch(_page_text(blocks)))


def _truncate_pages(pages: List[Page]) -> Tuple[List[Page], int]:
    for idx, page in enumerate(pages):
        if _is_doc_end_page(page.get("blocks", [])):
            return pages[: idx + 1], len(pages) - idx - 1
    return pages, 0


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
