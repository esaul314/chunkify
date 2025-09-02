from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from pdf_chunker.framework import Artifact, register

Block = Dict[str, Any]
Page = Dict[str, Any]

DOT_LEADER_RE = re.compile(r"(?:\.\s*){3,}")
DOC_END_RE = re.compile(r"(?i)^(?:the\s+)?end(?:\s+of\s+document)?\.?")


def _is_doc_end(text: str) -> bool:
    clean = text.strip()
    return bool(DOC_END_RE.fullmatch(clean)) and not DOT_LEADER_RE.search(clean)


def _truncate_pages(pages: List[Page]) -> Tuple[List[Page], int]:
    for idx, page in enumerate(pages):
        if any(_is_doc_end(b.get("text", "")) for b in page.get("blocks", [])):
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
