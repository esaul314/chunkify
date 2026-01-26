"""List boundary detection utilities for split_semantic pass.

This module contains functions for detecting list item boundaries
and determining when lists should be split or kept together.

Extracted from split_semantic.py for modularity.
"""

from __future__ import annotations

import re
from itertools import accumulate
from typing import TYPE_CHECKING, Any

from pdf_chunker.passes.sentence_fusion import _is_continuation_lead

if TYPE_CHECKING:
    from pdf_chunker.strategies.bullets import BulletHeuristicStrategy

Block = dict[str, Any]
Record = tuple[int, Block, str]

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_NUMBERED_ANYWHERE_RE = re.compile(r"(?:^|\n|\s)(\d+)[.)]\s+")


# ---------------------------------------------------------------------------
# Strategy resolution
# ---------------------------------------------------------------------------


def _resolve_bullet_strategy(
    strategy: BulletHeuristicStrategy | None,
) -> BulletHeuristicStrategy:
    """Return the provided strategy or the default."""
    from pdf_chunker.strategies.bullets import default_bullet_strategy

    return strategy or default_bullet_strategy()


# ---------------------------------------------------------------------------
# List number extraction
# ---------------------------------------------------------------------------


def first_list_number(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> int | None:
    """Return the first numbered list item number in text, if any."""
    heuristics = _resolve_bullet_strategy(strategy)
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not line or not heuristics.starts_with_number(line):
        match = _NUMBERED_ANYWHERE_RE.search(text)
        return int(match.group(1)) if match else None
    match = re.match(r"^(\d+)[.)]\s+", line)
    return int(match.group(1)) if match else None


def last_list_number(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> int | None:
    """Return the last numbered list item number in text, if any."""
    heuristics = _resolve_bullet_strategy(strategy)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        if heuristics.starts_with_number(line):
            match = re.match(r"^(\d+)[.)]\s+", line)
            return int(match.group(1)) if match else None
    matches = _NUMBERED_ANYWHERE_RE.findall(text)
    return int(matches[-1]) if matches else None


# ---------------------------------------------------------------------------
# List-like detection
# ---------------------------------------------------------------------------


def starts_list_like(
    block: Block,
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Return True if the block/text represents a list item start."""
    from pdf_chunker.passes.split_semantic_lists import _block_list_kind

    kind = _block_list_kind(block)
    if kind:
        return True
    if block.get("type") == "list_item":
        return True
    stripped = text.lstrip()
    if not stripped:
        return False
    heuristics = _resolve_bullet_strategy(strategy)
    return heuristics.starts_with_bullet(stripped) or heuristics.starts_with_number(stripped)


def record_is_list_like(
    record: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Return True if the record represents a list item."""
    _, block, text = record
    return starts_list_like(block, text, strategy=strategy)


# ---------------------------------------------------------------------------
# List tail detection
# ---------------------------------------------------------------------------


def list_tail_split_index(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> int | None:
    """Return the index where a list block transitions into narrative text.

    This identifies the position where a list ends and prose begins,
    which is useful for splitting mixed list/prose blocks.
    """
    heuristics = _resolve_bullet_strategy(strategy)

    parts = text.splitlines(keepends=True)
    if len(parts) <= 1:
        return None
    offsets = tuple(accumulate(len(part) for part in parts))
    candidates = zip(range(1, len(parts)), parts[1:], offsets[:-1], strict=False)
    for idx, part, offset in candidates:
        stripped = part.lstrip()
        if not stripped:
            continue
        if heuristics.starts_with_bullet(stripped) or heuristics.starts_with_number(stripped):
            continue
        indent = len(part) - len(stripped)
        if indent:
            continue
        prefix = "".join(parts[:idx]).rstrip()
        if not prefix or prefix[-1] not in ".?!":
            continue
        if _is_continuation_lead(stripped) or stripped[0].islower():
            continue
        return offset
    return None


# ---------------------------------------------------------------------------
# List boundary emission
# ---------------------------------------------------------------------------


def should_emit_list_boundary(
    previous: Record,
    block: Block,
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Determine if a chunk boundary should be emitted between list items.

    Returns True when the current item should start a new chunk.
    """
    from pdf_chunker.passes.split_modules.footers import record_is_footer_candidate

    _, prev_block, prev_text = previous
    if prev_text.rstrip().endswith(":"):
        return False
    if starts_list_like(prev_block, prev_text, strategy=strategy):
        return False
    prev_num = last_list_number(prev_text, strategy=strategy)
    next_num = first_list_number(text, strategy=strategy)
    if prev_num is not None and next_num is not None and next_num == prev_num + 1:
        return False
    return not record_is_footer_candidate(previous, strategy=strategy)


def colon_bullet_boundary(
    prev_text: str,
    block: Block,
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Check if a colon-prefixed bullet boundary exists."""
    return prev_text.rstrip().endswith(":") and starts_list_like(
        block,
        text,
        strategy=strategy,
    )
