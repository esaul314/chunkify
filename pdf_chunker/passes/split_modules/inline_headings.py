"""Inline heading detection and promotion for split_semantic.

This module handles detection and extraction of headings that appear inline
with body text in document blocks, as well as promotion of blocks to heading
type based on their inline style attributes.

Functions:
    split_inline_heading: Split a block when it contains an inline heading
    split_inline_heading_records: Apply inline heading splitting to records
    promote_inline_heading: Promote a block to heading based on inline styles
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from pdf_chunker.passes.split_semantic_inline import (
    _body_styles,
    _heading_styles,
    _leading_heading_candidate,
    _next_non_whitespace,
    _span_bounds,
    _span_style,
    _trimmed_segment,
)
from pdf_chunker.passes.split_semantic_lists import (
    _STYLED_LIST_KIND,
)

Block = dict[str, Any]
Record = tuple[int, Block, str]


def split_inline_heading(block: Block, text: str) -> tuple[Block, Block] | None:
    """Split ``block`` into heading and body when inline heading detected.

    Returns a tuple of (heading_block, body_block) if an inline heading is
    detected at the start of the text, or None if no inline heading found.

    Detection criteria:
    - Block must be a paragraph type (or None)
    - Block must have inline_styles
    - Leading heading candidate must exist
    - Heading text must be ≤6 words
    - Body text must be ≥5 words
    - Next character after heading must not be punctuation or lowercase
    """
    if not text or block.get("type") not in {None, "paragraph"}:
        return None
    styles = tuple(block.get("inline_styles") or ())
    if not styles:
        return None
    candidate = _leading_heading_candidate(text, styles)
    if candidate is None:
        return None
    _, end = candidate
    prefix = text[:end]
    suffix = text[end:]
    heading_text, heading_lead_trim = _trimmed_segment(prefix)
    body_text = suffix.lstrip()
    if not heading_text or not body_text:
        return None
    if len(heading_text.split()) > 6:
        return None
    if len(body_text.split()) < 5:
        return None
    trailer = _next_non_whitespace(text, end)
    if trailer is None or trailer in ",;:-–—" or trailer.islower():
        return None
    body_lead_trim = len(suffix) - len(body_text)
    text_length = len(text)
    heading_limit = len(heading_text)
    body_offset = end + body_lead_trim
    body_limit = len(body_text)
    heading_styles_result = _heading_styles(
        styles,
        text_length,
        end,
        heading_lead_trim,
        heading_limit,
    )
    body_styles_result = _body_styles(styles, text_length, body_offset, body_limit)
    base = {key: value for key, value in dict(block).items() if key != "text"}
    base.pop("inline_styles", None)
    heading_block: Block = {
        **{k: v for k, v in base.items() if k != "list_kind"},
        "type": "heading",
        "text": heading_text,
    }
    if heading_styles_result:
        heading_block["inline_styles"] = heading_styles_result
    body_block: Block = {
        **base,
        "type": "list_item",
        "list_kind": base.get("list_kind") or _STYLED_LIST_KIND,
        "text": body_text,
    }
    if body_styles_result:
        body_block["inline_styles"] = body_styles_result
    return heading_block, body_block


def split_inline_heading_records(
    records: Iterable[Record],
) -> Iterator[Record]:
    """Yield records with inline headings split into separate records.

    For each record, checks if the block contains an inline heading. If so,
    yields two records: one for the heading and one for the body text.
    Otherwise yields the original record unchanged.
    """
    for page, block, text in records:
        split = split_inline_heading(block, text)
        if not split:
            yield page, block, text
            continue
        heading_block, body_block = split
        yield page, heading_block, heading_block.get("text", "")
        yield page, body_block, body_block.get("text", "")


def promote_inline_heading(block: Block, text: str) -> Block:
    """Return ``block`` promoted to a heading when inline styles indicate one.

    A block is promoted to heading type if:
    - It's not already a heading
    - It has inline styles
    - The text has ≤12 words
    - At least one style covers the entire text and is a heading style
      (bold, italic, small_caps, caps, uppercase, or large)
    """
    if block.get("type") == "heading":
        return block

    styles = tuple(block.get("inline_styles") or ())
    if not styles:
        return block

    length = len(text)

    def _covers_entire(style: Any) -> bool:
        bounds = _span_bounds(style, length)
        if bounds is None:
            return False
        start, end = bounds
        return start == 0 and end >= length

    def _is_heading_style(style: Any) -> bool:
        flavor = _span_style(style).lower()
        return flavor in {"bold", "italic", "small_caps", "caps", "uppercase", "large"}

    word_limit = len(tuple(token for token in text.split() if token))
    if word_limit > 12:
        return block

    if any(_covers_entire(style) and _is_heading_style(style) for style in styles):
        return {**block, "type": "heading"}

    return block


# Aliases for backward compatibility (underscore-prefixed)
_split_inline_heading = split_inline_heading
_split_inline_heading_records = split_inline_heading_records
_promote_inline_heading = promote_inline_heading
