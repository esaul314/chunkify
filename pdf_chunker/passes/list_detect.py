"""List detection pass.

Annotates ``page_blocks`` documents with list metadata. Blocks that
start with bullets or numbered markers, or continue such items, are marked
as ``list_item`` with a corresponding ``list_kind``.
"""

from __future__ import annotations

import re
from functools import reduce
from typing import Any, Dict, Iterable, List, Tuple

from pdf_chunker.framework import Artifact, register


BULLET_CHARS = "*•◦▪‣·●◉○‧"
BULLET_CHARS_ESC = re.escape(BULLET_CHARS)
HYPHEN_BULLET_PREFIX = "- "
NUMBERED_RE = re.compile(r"\s*\d+[.)]")


def starts_with_bullet(text: str) -> bool:
    """Return True if ``text`` begins with a bullet marker or hyphen bullet."""

    stripped = text.lstrip()
    return stripped.startswith(tuple(BULLET_CHARS)) or stripped.startswith(
        HYPHEN_BULLET_PREFIX
    )


def _last_non_empty_line(text: str) -> str:
    return next(
        (line.strip() for line in reversed(text.splitlines()) if line.strip()),
        "",
    )


def is_bullet_continuation(curr: str, nxt: str) -> bool:
    """Return True when ``nxt`` continues a bullet item from ``curr``."""

    last_line = _last_non_empty_line(curr)
    return last_line.endswith(tuple(BULLET_CHARS)) and nxt[:1].islower()


def is_bullet_fragment(curr: str, nxt: str) -> bool:
    """Return True when ``nxt`` starts with text that continues the last bullet in ``curr``."""

    last_line = _last_non_empty_line(curr)
    return (
        starts_with_bullet(last_line)
        and not last_line.rstrip().endswith((".", "!", "?"))
        and nxt[:1].islower()
    )


def split_bullet_fragment(text: str) -> Tuple[str, str]:
    """Split the first line from the remainder, if any."""

    if "\n" not in text:
        return text.strip(), ""
    first, rest = text.split("\n", 1)
    return first.strip(), rest.lstrip()


def is_bullet_list_pair(curr: str, nxt: str) -> bool:
    """Return True when ``curr`` and ``nxt`` belong to the same bullet list."""

    colon_bullet = curr.rstrip().endswith(":") or re.search(
        rf":\s*(?:[{BULLET_CHARS_ESC}]|-)", curr
    )
    has_bullet = starts_with_bullet(curr) or any(
        starts_with_bullet(line) for line in curr.splitlines()
    )
    return starts_with_bullet(nxt) and (has_bullet or colon_bullet)


def starts_with_number(text: str) -> bool:
    """Return True if ``text`` begins with a numbered list marker."""

    return bool(NUMBERED_RE.match(text))


def is_numbered_list_pair(curr: str, nxt: str) -> bool:
    """Return True when ``curr`` and ``nxt`` belong to the same numbered list."""

    has_number = starts_with_number(curr) or any(
        starts_with_number(line) for line in curr.splitlines()
    )
    return starts_with_number(nxt) and has_number


def is_numbered_continuation(curr: str, nxt: str) -> bool:
    """Return True when ``nxt`` continues a numbered item from ``curr``."""

    return (
        starts_with_number(curr)
        and not starts_with_number(nxt)
        and not curr.rstrip().endswith((".", "!", "?"))
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
