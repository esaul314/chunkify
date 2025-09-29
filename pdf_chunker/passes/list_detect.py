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
LEADING_BULLET_RE = re.compile(rf"^\s*(?:[{BULLET_CHARS_ESC}]\s+)")
LEADING_HYPHEN_RE = re.compile(r"^\s*-\s+")
INLINE_COLON_BULLET_RE = re.compile(
    rf":\s*(?:[{BULLET_CHARS_ESC}]\s+|-\s+)"
)
NUMBERED_RE = re.compile(r"\s*\d+[.)]\s+")


def starts_with_bullet(text: str) -> bool:
    """Return True if ``text`` begins with a bullet marker or hyphen bullet."""

    return bool(
        LEADING_BULLET_RE.match(text) or LEADING_HYPHEN_RE.match(text)
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
    """Return True when ``nxt`` continues the last bullet in ``curr``."""

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


def _block_contains_bullet_marker(text: str) -> bool:
    """Return True when any line in ``text`` begins with a bullet marker."""

    return starts_with_bullet(text) or any(
        starts_with_bullet(line) for line in text.splitlines()
    )


def colon_leads_bullet_list(text: str) -> bool:
    """Return True when a trailing colon signals an inline bullet leader."""

    stripped = text.rstrip()
    inline_marker = bool(INLINE_COLON_BULLET_RE.search(text))
    has_bullet = _block_contains_bullet_marker(text)
    return inline_marker or (stripped.endswith(":") and has_bullet)


def is_bullet_list_pair(curr: str, nxt: str) -> bool:
    """Return True when ``curr`` and ``nxt`` belong to the same bullet list."""

    if not starts_with_bullet(nxt):
        return False
    has_bullet = _block_contains_bullet_marker(curr)
    return has_bullet or colon_leads_bullet_list(curr)


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


def _list_kind(curr: str, prev: str | None) -> str | None:
    if starts_with_bullet(curr) or (prev and is_bullet_continuation(prev, curr)):
        return "bullet"
    if starts_with_number(curr) or (prev and is_numbered_continuation(prev, curr)):
        return "numbered"
    return None


def _annotate(prev: Block | None, block: Block) -> Block:
    prev_text = prev.get("text") if prev else None
    kind = _list_kind(block.get("text", ""), prev_text)
    return {**block, "type": "list_item", "list_kind": kind} if kind else block


def _annotate_blocks(blocks: Iterable[Block]) -> List[Block]:
    def step(
        state: Tuple[Block | None, List[Block]], block: Block
    ) -> Tuple[Block | None, List[Block]]:
        prev, acc = state
        annotated = _annotate(prev, block)
        return annotated, [*acc, annotated]

    initial: Tuple[Block | None, List[Block]] = (None, [])
    return reduce(step, blocks, initial)[1]


def _annotate_page(page: Dict[str, Any]) -> Dict[str, Any]:
    return {**page, "blocks": _annotate_blocks(page.get("blocks", []))}


def _count_kinds(blocks: Iterable[Block]) -> Dict[str, int]:
    return {
        "bullet_items": sum(b.get("list_kind") == "bullet" for b in blocks),
        "numbered_items": sum(b.get("list_kind") == "numbered" for b in blocks),
    }


def _annotate_doc(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    pages = [_annotate_page(p) for p in doc.get("pages", [])]
    blocks = [b for p in pages for b in p.get("blocks", [])]
    return {**doc, "pages": pages}, _count_kinds(blocks)


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
