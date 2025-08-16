"""List detection pass.

Annotates ``page_blocks`` documents with list metadata. Blocks that
start with bullets or numbered markers, or continue such items, are marked
as ``list_item`` with a corresponding ``list_kind``.
"""

from functools import reduce
from typing import Any, Dict, Iterable, List, Tuple

from pdf_chunker.framework import Artifact, register
from pdf_chunker.list_detection import (
    is_bullet_continuation,
    is_numbered_continuation,
    starts_with_bullet,
    starts_with_number,
)


Block = Dict[str, Any]


def _annotate(prev: Block | None, block: Block) -> Block:
    text = block.get("text", "")
    prev_text = prev.get("text", "") if prev else ""
    kind = (
        "bullet"
        if starts_with_bullet(text) or (prev and is_bullet_continuation(prev_text, text))
        else "numbered"
        if starts_with_number(text) or (prev and is_numbered_continuation(prev_text, text))
        else None
    )
    return {**block, "type": "list_item", "list_kind": kind} if kind else block


def _annotate_blocks(blocks: Iterable[Block]) -> List[Block]:
    def step(state: Tuple[Block | None, List[Block]], block: Block) -> Tuple[Block, List[Block]]:
        prev, acc = state
        annotated = _annotate(prev, block)
        return annotated, [*acc, annotated]

    return reduce(step, blocks, (None, []))[1]


def _annotate_page(page: Dict[str, Any]) -> Dict[str, Any]:
    return {**page, "blocks": _annotate_blocks(page.get("blocks", []))}


def _annotate_doc(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    pages = [_annotate_page(p) for p in doc.get("pages", [])]
    blocks = [b for p in pages for b in p.get("blocks", [])]
    metrics = {
        "bullet_items": sum(1 for b in blocks if b.get("list_kind") == "bullet"),
        "numbered_items": sum(1 for b in blocks if b.get("list_kind") == "numbered"),
    }
    return {**doc, "pages": pages}, metrics


class _ListDetectPass:
    name = "list_detect"
    input_type = dict
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload
        if not isinstance(doc, dict) or doc.get("type") != "page_blocks":
            return a

        updated, counts = _annotate_doc(doc)
        meta = dict(a.meta or {})
        meta.setdefault("metrics", {}).setdefault("list_detect", {}).update(counts)
        return Artifact(payload=updated, meta=meta)


list_detect = register(_ListDetectPass())
