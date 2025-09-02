from __future__ import annotations

from functools import reduce
from typing import Any, Dict, Tuple

from pdf_chunker.framework import Artifact, register
from pdf_chunker import page_artifacts

Block = Dict[str, Any]


def _clean_block(block: Block, page_num: int) -> Tuple[Block, bool]:
    text = block.get("text", "")
    cleaned = page_artifacts._flatten_markdown_table(text)
    changed = cleaned != text
    return {**block, "text": cleaned}, changed


def _clean_page(page: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    page_num = page.get("page", 0)
    blocks = page.get("blocks", [])
    cleaned_blocks, changed_flags = zip(*(_clean_block(b, page_num) for b in blocks)) if blocks else ([], [])
    return {**page, "blocks": list(cleaned_blocks)}, sum(changed_flags)


def _clean_doc(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    def step(acc: Tuple[list[Dict[str, Any]], int], page: Dict[str, Any]) -> Tuple[list[Dict[str, Any]], int]:
        pages, total = acc
        cleaned, changed = _clean_page(page)
        return [*pages, cleaned], total + changed

    pages, total = reduce(step, doc.get("pages", []), ([], 0))
    return {**doc, "pages": pages}, total


class _DetectPageArtifactsPass:
    name = "detect_page_artifacts"
    input_type = object
    output_type = object

    def __call__(self, a: Artifact) -> Artifact:
        payload = a.payload
        if not isinstance(payload, dict) or payload.get("type") != "page_blocks":
            return a
        cleaned_doc, changed = _clean_doc(payload)
        meta = dict(a.meta or {})
        metrics = meta.setdefault("metrics", {})
        metrics.setdefault("detect_page_artifacts", {})["blocks_cleaned"] = changed
        return Artifact(payload=cleaned_doc, meta=meta)


detect_page_artifacts = register(_DetectPageArtifactsPass())
