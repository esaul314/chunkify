"""Footer detection and stripping for split_semantic pass.

This module consolidates all footer-related heuristics that were previously
embedded in split_semantic.py. Footer detection is complex because:

1. Footers can appear as trailing bullet lines in a block
2. Footers can be separate blocks that follow content
3. Footers often have short lines with specific patterns
4. Context matters: what precedes a potential footer affects classification

Design philosophy:
- Pure functions: no side effects, all state passed explicitly
- Strategy pattern: BulletHeuristicStrategy controls detection behavior
- Explicit returns: functions return Optional types when filtering
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

from pdf_chunker.page_artifacts import (
    _bullet_body,
    _drop_trailing_bullet_footers,
    _footer_bullet_signals,
    _header_invites_footer,
)
from pdf_chunker.strategies.bullets import (
    BulletHeuristicStrategy,
    default_bullet_strategy,
)

if TYPE_CHECKING:
    pass

Block = dict[str, Any]
Record = tuple[int, Block, str]


def _resolve_bullet_strategy(
    strategy: BulletHeuristicStrategy | None,
) -> BulletHeuristicStrategy:
    """Return the strategy or the default if None."""
    return strategy if strategy is not None else default_bullet_strategy()


def _previous_non_empty_line(lines: tuple[str, ...]) -> str:
    """Return the last non-empty line from a tuple of lines."""
    return next((line for line in reversed(lines) if line.strip()), "")


def _footer_context_allows(previous_line: str, trailing_count: int) -> bool:
    """Check if context allows treating trailing lines as footers.

    Footer classification depends on what precedes the candidate:
    - Bullet signals in the previous line suggest continuation, not footer
    - Headers that "invite" footers (like page numbers) allow footer classification
    """
    return any(
        (
            _footer_bullet_signals("", previous_line),
            _header_invites_footer(previous_line, trailing_count),
        )
    )


def _footer_line_is_artifact(line: str, previous_line: str) -> bool:
    """Check if a line looks like a footer artifact.

    A line is a footer artifact if:
    - It has no bullet body (empty after stripping bullet marker)
    - OR it has footer bullet signals relative to the previous line
    """
    body = _bullet_body(line)
    return not body or _footer_bullet_signals(body, previous_line)


def resolve_footer_suffix(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Identify trailing lines that should be treated as footer artifacts.

    This function analyzes the end of a block to find lines that look like
    footers rather than content. It uses heuristics based on:
    - Bullet pattern detection
    - Context from preceding lines
    - Artifact signal matching

    Args:
        lines: Tuple of stripped lines from a text block

    Returns:
        Tuple of lines identified as footer suffix (empty if none detected)
    """
    pruned = tuple(_drop_trailing_bullet_footers(list(lines)))
    if len(pruned) == len(lines):
        return tuple()
    suffix = lines[len(pruned):]
    if not suffix:
        return tuple()
    previous_line = _previous_non_empty_line(pruned)
    if not _footer_context_allows(previous_line, len(suffix)):
        return tuple()
    if not all(_footer_line_is_artifact(line, previous_line) for line in suffix):
        return tuple()
    return suffix


def record_trailing_footer_lines(
    record: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[str, ...]:
    """Return trailing bullet lines that heuristically resemble footers.

    Examines the end of a record's text for lines that look like footer
    artifacts (page numbers, short bullet fragments, etc.).

    Args:
        record: (page, block, text) tuple
        strategy: Bullet detection strategy (uses default if None)

    Returns:
        Tuple of lines identified as footer artifacts (empty if none)
    """
    heuristics = _resolve_bullet_strategy(strategy)
    _, block, text = record

    # Only process list-like blocks
    if not _starts_list_like(block, text, strategy=heuristics):
        return tuple()

    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not lines:
        return tuple()

    suffix = resolve_footer_suffix(lines)
    bullet_like = tuple(
        line
        for line in suffix
        if heuristics.starts_with_bullet(line) or heuristics.starts_with_number(line)
    )
    return bullet_like if bullet_like == suffix else tuple()


def record_is_footer_candidate(
    record: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Check if a record contains footer-like trailing content.

    Args:
        record: (page, block, text) tuple
        strategy: Bullet detection strategy

    Returns:
        True if the record has trailing footer lines
    """
    return bool(record_trailing_footer_lines(record, strategy=strategy))


def _trim_footer_suffix(text: str, suffix: tuple[str, ...]) -> str:
    """Return text with trailing suffix bullet lines removed.

    Carefully removes exactly the suffix lines from the end of text,
    preserving all other content.

    Args:
        text: Original text
        suffix: Lines to remove from the end

    Returns:
        Text with suffix removed
    """
    if not suffix:
        return text
    lines = text.splitlines()
    if not lines:
        return text

    trimmed = list(lines)
    suffix_lines = tuple(line.strip() for line in suffix if line.strip())
    if not suffix_lines:
        return text

    index = len(trimmed) - 1
    for candidate in reversed(suffix_lines):
        while index >= 0 and not trimmed[index].strip():
            trimmed.pop()
            index -= 1
        if index < 0:
            return text
        if trimmed[index].strip() != candidate:
            return text
        trimmed.pop()
        index -= 1

    while trimmed and not trimmed[-1].strip():
        trimmed.pop()

    return "\n".join(trimmed)


def strip_footer_suffix(
    record: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> Record | None:
    """Return record without footer bullets or None if empty after stripping.

    Args:
        record: (page, block, text) tuple
        strategy: Bullet detection strategy

    Returns:
        Modified record without footer suffix, or None if nothing remains
    """
    heuristics = _resolve_bullet_strategy(strategy)
    page, block, text = record

    suffix = record_trailing_footer_lines(record, strategy=heuristics)
    if not suffix:
        return record

    trimmed = _trim_footer_suffix(text, suffix)
    if trimmed == text:
        return record
    if not trimmed.strip():
        return None

    # Update block with trimmed text and record source
    updated_block: Block = block
    if isinstance(block, Mapping):
        original = {
            key: value for key, value in dict(block).items() if key != "source_blocks"
        }
        original["text"] = text
        existing_sources = tuple(
            {key: value for key, value in dict(source).items() if key != "source_blocks"}
            for source in block.get("source_blocks", ())
            if isinstance(source, Mapping)
        )
        updated = dict(block)
        updated["text"] = trimmed
        if existing_sources or original.get("text"):
            updated["source_blocks"] = existing_sources + (original,)
        updated_block = cast(Block, updated)

    return page, updated_block, trimmed


def is_footer_artifact_record(
    previous: Record,
    current: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Check if current record resembles a stray footer list.

    This detects small bullet blocks that appear to be footer artifacts
    based on their position, size, and context.

    Args:
        previous: The preceding (page, block, text) record
        current: The candidate footer record
        strategy: Bullet detection strategy

    Returns:
        True if current looks like a footer artifact
    """
    heuristics = _resolve_bullet_strategy(strategy)
    prev_page, prev_block, prev_text = previous
    page, block, text = current

    # Must be on same page
    if page != prev_page:
        return False

    stripped_lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not stripped_lines or len(stripped_lines) > 2:
        return False

    # All lines must look like bullets/numbers
    if not all(
        heuristics.starts_with_bullet(line) or heuristics.starts_with_number(line)
        for line in stripped_lines
    ):
        return False

    # Must be short (not substantive content)
    word_total = sum(len(line.split()) for line in stripped_lines)
    if word_total > 20:
        return False

    # Context must allow footer classification
    previous_line = _previous_non_empty_line(tuple(prev_text.splitlines()))
    if not _footer_context_allows(previous_line, len(stripped_lines)):
        return False
    if not all(_footer_line_is_artifact(line, previous_line) for line in stripped_lines):
        return False

    # Check bbox width if available (narrow = likely footer)
    width = None
    if isinstance(block, Mapping):
        bbox = block.get("bbox")
        if isinstance(bbox, tuple) and len(bbox) == 4:
            try:
                width = float(bbox[2]) - float(bbox[0])
            except (TypeError, ValueError):
                width = None
    if width is not None and width > 260:
        return False

    # Previous block should not be a list (footers follow prose)
    return not _starts_list_like(prev_block, prev_text, strategy=heuristics)


def strip_footer_suffixes(
    records: Iterable[Record],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[Record, ...]:
    """Remove footer suffix bullets from all records.

    Processes records in order, stripping footer suffixes and filtering
    out records that become empty or look like footer artifacts.

    Args:
        records: Iterable of (page, block, text) tuples
        strategy: Bullet detection strategy

    Returns:
        Tuple of cleaned records
    """
    heuristics = _resolve_bullet_strategy(strategy)
    cleaned: list[Record] = []

    for record in records:
        trimmed = strip_footer_suffix(record, strategy=heuristics)
        if trimmed is None:
            continue
        if cleaned and is_footer_artifact_record(
            cleaned[-1], trimmed, strategy=heuristics
        ):
            continue
        cleaned.append(trimmed)

    return tuple(cleaned)


# ---------------------------------------------------------------------------
# Helper function (imported from parent module to avoid circular imports)
# ---------------------------------------------------------------------------


def _starts_list_like(
    block: Block,
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Check if a block looks like it starts a list.

    This is a simplified version that avoids importing from the parent
    split_semantic module to prevent circular imports.
    """
    heuristics = _resolve_bullet_strategy(strategy)
    first_line = text.split("\n", 1)[0].strip() if text else ""
    return (
        heuristics.starts_with_bullet(first_line)
        or heuristics.starts_with_number(first_line)
    )
