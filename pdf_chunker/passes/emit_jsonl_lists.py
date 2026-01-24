"""List detection and manipulation utilities for emit_jsonl.

These functions handle bullet/numbered list detection, gap collapsing,
list introduction parsing, and incomplete list detection.

Extracted from emit_jsonl.py for clarity and testability.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import chain, takewhile
from typing import TYPE_CHECKING

from pdf_chunker.passes.emit_jsonl_text import (
    first_non_empty_line,
    trim_trailing_empty,
)

if TYPE_CHECKING:
    from pdf_chunker.strategies.bullets import BulletHeuristicStrategy

# ---------------------------------------------------------------------------
# Constants / Compiled Patterns
# ---------------------------------------------------------------------------

_LIST_GAP_RE = re.compile(r"\n{2,}(?=\s*(?:[-\*\u2022]|\d+\.))")
_INLINE_BULLET_RE = re.compile(r":\s*[•\-\*\u2022]\s+\w")
_INLINE_NUMBER_RE = re.compile(r":\s*\d+[.)]\s+\w")


# ---------------------------------------------------------------------------
# Core list line detection
# ---------------------------------------------------------------------------


def is_list_line(
    line: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Check if a line starts with a bullet or numbered marker."""
    from pdf_chunker.strategies.bullets import default_bullet_strategy

    stripped = line.lstrip()
    if not stripped:
        return False
    heuristics = strategy or default_bullet_strategy()
    return heuristics.starts_with_bullet(stripped) or heuristics.starts_with_number(stripped)


# ---------------------------------------------------------------------------
# Inline list detection
# ---------------------------------------------------------------------------


def has_inline_list_start(text: str) -> bool:
    """Return True if text contains inline list start (colon followed by bullet/number)."""
    return bool(_INLINE_BULLET_RE.search(text) or _INLINE_NUMBER_RE.search(text))


def count_list_items(text: str) -> tuple[int, int]:
    """Count bullet and numbered list items in text.

    Returns (bullet_count, numbered_count).
    """
    bullet_matches = len(re.findall(r"[•\-\*\u2022]\s+\w", text))
    numbered_matches = len(re.findall(r"\d+[.)]\s+\w", text))
    return bullet_matches, numbered_matches


# ---------------------------------------------------------------------------
# Incomplete list detection predicates
# ---------------------------------------------------------------------------


def ends_with_list_intro_colon(
    lines: list[str],
    *,
    is_list_line_fn: Callable[[str], bool] | None = None,
) -> bool:
    """Check if text is a pure list introduction ending with colon.

    Example: "Here is a guide:" with no bullet items present.
    """
    if not lines:
        return False
    last_line = lines[-1].rstrip()
    if not last_line.endswith(":"):
        return False
    # Verify no bullets present - this is a pure introduction
    all_text = "\n".join(lines)
    bullets, numbers = count_list_items(all_text)
    return bullets == 0 and numbers == 0


def has_single_inline_bullet(
    lines: list[str],
    *,
    is_list_line_fn: Callable[[str], bool] | None = None,
) -> bool:
    """Check if text has a colon followed by single inline bullet.

    Example: "List intro: • single item" - incomplete because only one item.
    """
    if not lines:
        return False
    last_line = lines[-1].rstrip()
    if not has_inline_list_start(last_line):
        return False
    # Check if there's only one list item total
    all_text = "\n".join(lines)
    bullets, numbers = count_list_items(all_text)
    return (bullets + numbers) == 1


def has_unterminated_bullet_item(
    lines: list[str],
    *,
    is_list_line_fn: Callable[[str], bool] | None = None,
) -> bool:
    """Check if text has a single bullet item without sentence terminator.

    Example: "Intro:\\n• First item" where "First item" lacks a period.
    """
    predicate = is_list_line_fn or is_list_line

    if len(lines) < 2:
        return False

    # Count list items at line start
    bullet_count = sum(1 for ln in lines if predicate(ln.strip()))
    if bullet_count != 1:
        return False

    # Find the bullet line
    first_bullet_idx = next((i for i, ln in enumerate(lines) if predicate(ln.strip())), None)
    if first_bullet_idx is None:
        return False

    # Check if there's intro text ending with colon before the bullet
    if first_bullet_idx > 0:
        pre_bullet = "\n".join(lines[:first_bullet_idx])
        if pre_bullet.rstrip().endswith(":"):
            return True

    # Check if single bullet item lacks sentence terminator
    last_line = lines[-1].strip()
    return predicate(last_line) and not re.search(r"[.!?][\"')\]]*$", last_line)


def has_incomplete_list(
    text: str,
    *,
    is_list_line_fn: Callable[[str], bool] | None = None,
) -> bool:
    """Return True if text appears to have an incomplete list.

    A text is considered to have an incomplete list if it:
    1. Ends with a colon (list introduction without the list items)
    2. Has an inline list start (colon followed by single bullet item)
    3. Has only a single list item when the context suggests more items follow
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    return (
        ends_with_list_intro_colon(lines, is_list_line_fn=is_list_line_fn)
        or has_single_inline_bullet(lines, is_list_line_fn=is_list_line_fn)
        or has_unterminated_bullet_item(lines, is_list_line_fn=is_list_line_fn)
    )


# ---------------------------------------------------------------------------
# List gap and boundary manipulation
# ---------------------------------------------------------------------------


def collapse_list_gaps(
    text: str,
    *,
    is_list_line_fn: Callable[[str], bool] | None = None,
) -> str:
    """Collapse multiple newlines before list items to single newlines."""
    predicate = is_list_line_fn or is_list_line

    def repl(match: re.Match[str]) -> str:
        prior = text[: match.start()]
        prev_line = prior.splitlines()[-1] if "\n" in prior else prior
        return "\n" if not predicate(prev_line) else match.group(0)

    return _LIST_GAP_RE.sub(repl, text)


def split_inline_list_start(
    line: str,
    *,
    is_list_line_fn: Callable[[str], bool] | None = None,
) -> tuple[str, str] | None:
    """Split a line at an inline list start if present.

    Returns (preamble, list_start) or None if no inline list start found.
    """
    predicate = is_list_line_fn or is_list_line
    for idx, char in enumerate(line):
        if char in "-\u2022*" and (idx == 0 or line[idx - 1].isspace()):
            tail = line[idx:].lstrip()
            if predicate(tail):
                return line[:idx].rstrip(), tail
        if char.isdigit() and (idx == 0 or line[idx - 1].isspace()):
            end = idx
            while end < len(line) and line[end].isdigit():
                end += 1
            if (
                end < len(line)
                and line[end] == "."
                and end + 1 < len(line)
                and line[end + 1].isspace()
            ):
                tail = line[idx:].lstrip()
                if predicate(tail):
                    return line[:idx].rstrip(), tail
    return None


# ---------------------------------------------------------------------------
# List intro detection and peeling
# ---------------------------------------------------------------------------


def list_intro_start(text: str) -> int:
    """Return the index where a trailing list introduction begins."""
    return max(
        (
            pos + span
            for token, span in (("\n\n", 2), (". ", 2), ("! ", 2), ("? ", 2))
            if (pos := text.rfind(token)) != -1
        ),
        default=-1,
    )


def peel_list_intro(text: str) -> tuple[str, str]:
    """Split text into non-intro content and the trailing list preamble."""
    stripped = text.rstrip()
    colon_idx = max(stripped.rfind(":"), stripped.rfind("："))
    if colon_idx == -1:
        return text, ""
    prefix = stripped[: colon_idx + 1]
    start = list_intro_start(prefix)
    if start <= 0:
        return text, ""
    return prefix[:start].rstrip(), prefix[start:].lstrip()


def partition_preamble(lines: list[str]) -> tuple[list[str], list[str]]:
    """Partition lines into main content and trailing preamble."""
    if not lines:
        return [], []

    idx = len(lines)
    while idx > 0 and lines[idx - 1].strip():
        idx -= 1
    while idx > 0 and not lines[idx - 1].strip():
        idx -= 1

    if idx == 0:
        return lines, []
    return lines[:idx], lines[idx:]


# ---------------------------------------------------------------------------
# List rebalancing
# ---------------------------------------------------------------------------


def compose_intro_with_chunk(intro: str, chunk: str, separators: int) -> str:
    """Compose intro and chunk with controlled blank-line separators."""
    intro_lines = intro.splitlines()
    chunk_body = chunk.strip("\n")
    chunk_lines = chunk_body.splitlines() if chunk_body else []

    if not intro_lines:
        return chunk_body
    if not chunk_lines:
        return "\n".join(intro_lines)

    desired_gaps = max(separators, 1)
    spacer = [""] * max(desired_gaps - 1, 0)
    return "\n".join(chain(intro_lines, spacer, chunk_lines))


def prepend_intro(intro: str, rest: str) -> str:
    """Attach intro ahead of rest while normalizing spacing."""
    intro_core = intro.strip("\n")
    if not rest:
        return intro_core

    leading_newlines = len(rest) - len(rest.lstrip("\n"))
    tail = rest[leading_newlines:]
    if not intro_core:
        return tail.strip("\n")

    trailing_intro_newlines = len(intro) - len(intro.rstrip("\n"))
    separators = trailing_intro_newlines + (leading_newlines or 1)
    return compose_intro_with_chunk(intro_core, tail, separators)


def rebalance_lists(
    raw: str,
    rest: str,
    *,
    is_list_line_fn: Callable[[str], bool] | None = None,
) -> tuple[str, str]:
    """Shift trailing context or list block into rest when it starts with a list."""
    predicate = is_list_line_fn or is_list_line

    if not rest or not predicate(first_non_empty_line(rest)):
        return raw, rest

    lines = trim_trailing_empty(raw.splitlines())
    trimmed = "\n".join(lines)
    has_list = any(predicate(ln) for ln in lines)

    if not has_list:
        kept, intro = peel_list_intro(trimmed)
        if intro:
            return kept, prepend_intro(intro, rest)

    # Determine split point: last non-list line if raw already contains list items,
    # otherwise the preceding blank line so that list introductions move with the list.
    idx = next(
        (
            i
            for i, ln in enumerate(reversed(lines))
            if ((ln.strip() and not predicate(ln)) if has_list else not ln.strip())
        ),
        len(lines),
    )
    start = len(lines) - idx
    if not has_list and start == 0:
        return trimmed, rest
    block = lines[start:]
    if not block:
        return trimmed, rest

    moved = "\n".join(block).strip()
    kept = "\n".join(lines[:start]).rstrip()
    return kept, prepend_intro(moved, rest)


# ---------------------------------------------------------------------------
# List reservation (for splitting)
# ---------------------------------------------------------------------------


def reserve_for_list(
    text: str,
    limit: int,
    *,
    is_list_line_fn: Callable[[str], bool] | None = None,
) -> tuple[str, str, str | None]:
    """Reserve space for a list block during text splitting.

    Returns (chunk_text, remainder, intro_hint).
    """
    predicate = is_list_line_fn or is_list_line
    collapsed = collapse_list_gaps(text, is_list_line_fn=predicate)
    lines = collapsed.splitlines()

    inline = next(
        (
            (idx, result)
            for idx, line in enumerate(lines)
            if (result := split_inline_list_start(line, is_list_line_fn=predicate))
        ),
        None,
    )
    list_idx = next((i for i, ln in enumerate(lines) if predicate(ln)), len(lines))

    if inline and inline[0] <= list_idx:
        idx, (head, tail) = inline
        pre_lines = [*lines[:idx], head] if head else lines[:idx]
        tail_lines = [tail, *lines[idx + 1 :]]
    elif list_idx < len(lines):
        pre_lines = lines[:list_idx]
        tail_lines = lines[list_idx:]
    else:
        return collapsed, "", None

    if not pre_lines:
        return collapsed, "", None

    block_lines = list(takewhile(lambda ln: not ln.strip() or predicate(ln), tail_lines))
    if not block_lines:
        return collapsed, "", None

    rest_lines = tail_lines[len(block_lines) :]
    trimmed_pre = trim_trailing_empty(pre_lines)
    trailing_gaps = pre_lines[len(trimmed_pre) :]

    if not trimmed_pre:
        return collapsed, "", None

    pre_text = "\n".join(trimmed_pre)
    block_text = "\n".join(block_lines)
    combined_len = len(pre_text) + (1 if pre_text and block_text else 0) + len(block_text)
    if combined_len <= limit:
        return collapsed, "", None

    keep_lines, intro_lines = partition_preamble(trimmed_pre)
    if not intro_lines and len(keep_lines) > 1 and any(predicate(ln) for ln in block_lines):
        candidate_intro = keep_lines[-1]
        if candidate_intro.strip() and not predicate(candidate_intro):
            keep_lines = keep_lines[:-1]
            intro_lines = [candidate_intro, *intro_lines]
    if not keep_lines:
        return collapsed, "", None

    chunk_text = "\n".join(keep_lines)
    remainder_parts = [
        *intro_lines,
        *trailing_gaps,
        *block_lines,
        *rest_lines,
    ]
    remainder = "\n".join(remainder_parts).lstrip("\n")
    intro_line = first_non_empty_line("\n".join(intro_lines)) if intro_lines else ""
    intro_hint = intro_line if intro_line.strip() else None
    return chunk_text, remainder, intro_hint


# ---------------------------------------------------------------------------
# List block detection
# ---------------------------------------------------------------------------


def starts_with_list_block(
    text: str,
    *,
    min_items: int = 2,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Return True if text starts with a multi-item bullet or numbered list.

    Unlike `starts_with_orphan_bullet`, this detects when a chunk begins with
    a proper list block (multiple items), indicating that a preceding short
    intro paragraph might belong with this list.

    Args:
        text: The text to check.
        min_items: Minimum number of list items to consider it a "block" (default 2).
        strategy: Bullet detection strategy to use.

    Returns:
        True if the text starts with a list block of at least `min_items` items.
    """
    from pdf_chunker.strategies.bullets import default_bullet_strategy

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    heuristics = strategy or default_bullet_strategy()
    first_line = lines[0].strip()

    # Check if first line is a list item
    is_bullet = heuristics.starts_with_bullet(first_line)
    is_number = heuristics.starts_with_number(first_line)
    if not (is_bullet or is_number):
        return False

    # Count consecutive list items from the start
    list_item_count = 0
    for ln in lines:
        stripped = ln.strip()
        if heuristics.starts_with_bullet(stripped) or heuristics.starts_with_number(stripped):
            list_item_count += 1
        elif list_item_count > 0:
            # Non-list line after list items - stop counting
            break

    return list_item_count >= min_items


# ---------------------------------------------------------------------------
# Orphan bullet detection
# ---------------------------------------------------------------------------


def starts_with_orphan_bullet(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
    coherent_fn: Callable[[str], bool] | None = None,
    word_count_fn: Callable[[str], int] | None = None,
) -> bool:
    """Return True if text starts with a single bullet item (orphaned list fragment).

    An orphan bullet is when text begins with a single bullet point that appears
    to be a fragment of a list from a previous chunk. Complete single-item lists
    are NOT considered orphans, even if they lack terminal punctuation.
    """
    from pdf_chunker.passes.emit_jsonl_text import word_count as default_word_count
    from pdf_chunker.strategies.bullets import default_bullet_strategy

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    first_line = lines[0].strip()
    heuristics = strategy or default_bullet_strategy()
    wc_fn = word_count_fn or default_word_count

    # Check if first line is a bullet
    is_bullet = heuristics.starts_with_bullet(first_line)
    is_number = heuristics.starts_with_number(first_line)
    if not (is_bullet or is_number):
        return False

    # If there's only one line, check if it looks like a complete item
    if len(lines) == 1:
        # A coherent item or numbered item with enough words is NOT orphaned
        if coherent_fn and coherent_fn(first_line):
            return False
        # Numbered items with sufficient words are likely complete even without punctuation
        return not (is_number and wc_fn(first_line) >= 6)

    # If second line is NOT a bullet, first bullet is orphaned
    # (unless the first line looks complete)
    second_line = lines[1].strip()
    is_second_list_line = heuristics.starts_with_bullet(
        second_line
    ) or heuristics.starts_with_number(second_line)
    if is_second_list_line:
        return False

    # First line is a bullet, but following content is not a list
    # Check if the first line alone looks complete
    if coherent_fn and coherent_fn(first_line):
        return False
    return not (is_number and wc_fn(first_line) >= 6)
