"""Segment emission and collapsing logic extracted from split_semantic.

This module contains the core logic for:
- Emitting record segments as merged or individual chunks
- Collapsing records into segments based on limits and boundaries
- Managing the _CollapseEmitter state machine
- Cross-page list handling and overflow detection

These functions were extracted to reduce the size of split_semantic.py
and improve maintainability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import reduce
from itertools import accumulate
from math import ceil
from typing import TYPE_CHECKING, Iterator

from pdf_chunker.passes.split_semantic_lists import BulletHeuristicStrategy

from .footers import record_is_footer_candidate as _record_is_footer_candidate
from .footers import strip_footer_suffixes as _strip_footer_suffixes
from .lists import (
    first_list_number as _first_list_number,
    last_list_number as _last_list_number,
    starts_list_like as _starts_list_like,
)
from .stitching import merge_record_block as _merge_record_block
from .stitching import with_chunk_index as _with_chunk_index

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pdf_chunker.passes.split_semantic import Block, SplitOptions

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Record = tuple[int, "Block", str]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOFT_LIMIT = 8000
_AVERAGE_CHARS_PER_TOKEN = 4.7
_ENDS_SENTENCE = re.compile(r'[.!?]["\'""\u2019]?\s*$')
_NUMBERED_TWO_RE = re.compile(r"(?:^|\n|\s)2[.)]\s+")


# ---------------------------------------------------------------------------
# Text utility functions
# ---------------------------------------------------------------------------


def _split_words(text: str) -> tuple[str, ...]:
    """Split text into word tokens."""
    return tuple(text.split())


def _effective_counts(text: str) -> tuple[int, int, int]:
    """Return word, dense token, and effective totals for ``text``."""
    words = _split_words(text)
    word_count = len(words)
    char_total = sum(len(token) for token in words)
    dense_total = int(ceil(char_total / _AVERAGE_CHARS_PER_TOKEN)) if char_total else 0
    if word_count <= 1 and text:
        dense_total = max(dense_total, len(text))
    effective_total = max(word_count, dense_total)
    return word_count, dense_total, effective_total


def _segment_totals(segment: tuple[Record, ...]) -> tuple[int, int, int]:
    """Return aggregated word, dense, and effective totals for a segment."""
    totals = tuple(_effective_counts(text) for _, _, text in segment)
    words = sum(count for count, _, _ in totals)
    dense = sum(count for _, count, _ in totals)
    return words, dense, max(words, dense)


def _segment_offsets(segments: tuple[tuple[Record, ...], ...]) -> tuple[int, ...]:
    """Compute starting offsets for each segment."""
    if not segments:
        return ()
    lengths = tuple(len(seg) for seg in segments)
    return (0,) + tuple(accumulate(lengths))[:-1]


def _enumerate_segments(
    segments: tuple[tuple[Record, ...], ...],
) -> tuple[tuple[int, tuple[Record, ...]], ...]:
    """Pair each segment with its starting offset."""
    offsets = _segment_offsets(segments)
    return tuple(zip(offsets, segments, strict=False))


# ---------------------------------------------------------------------------
# Bullet heuristic resolver
# ---------------------------------------------------------------------------


def _resolve_bullet_strategy(
    strategy: BulletHeuristicStrategy | None,
) -> BulletHeuristicStrategy:
    """Return strategy or default BulletHeuristicStrategy."""
    return strategy if strategy is not None else BulletHeuristicStrategy()


# ---------------------------------------------------------------------------
# Limit computation
# ---------------------------------------------------------------------------


def _resolved_limit(options: "SplitOptions | None", limit: int | None) -> int | None:
    """Compute the effective soft limit from options or explicit limit."""
    candidate: int | None
    if limit is not None:
        candidate = limit
    elif options is not None:
        candidate = options.compute_limit()
    else:
        candidate = None
    if candidate is None or candidate <= 0:
        return None
    return candidate


def _hard_limit(options: "SplitOptions | None", resolved_limit: int | None) -> int | None:
    """Compute the hard limit (chunk_size) from options."""
    if options is not None and options.chunk_size > 0:
        return options.chunk_size
    return resolved_limit


def _overlap_value(options: "SplitOptions | None") -> int:
    """Extract overlap value from options."""
    return options.overlap if options is not None else 0


# ---------------------------------------------------------------------------
# Join and normalize
# ---------------------------------------------------------------------------


def _normalize_numbered_list_text(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> str:
    """Normalize text that contains numbered list items."""
    # This is a pass-through for now; specific normalization can be added
    return text


def _join_record_texts(
    records: tuple[Record, ...],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> str:
    """Join the text portions of multiple records."""
    if not records:
        return ""
    texts = tuple(text for _, _, text in records)
    return "\n".join(texts)


def _apply_overlap_within_segment(
    segment: tuple[Record, ...],
    overlap: int,
) -> tuple[Record, ...]:
    """Apply overlap trimming within a segment."""
    if overlap <= 0 or len(segment) <= 1:
        return segment
    # Overlap is applied between segments, not within a single merged segment
    return segment


# ---------------------------------------------------------------------------
# Segment expansion and splitting
# ---------------------------------------------------------------------------


def _expand_segment_records(
    segment: tuple[Record, ...],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[Record, ...]:
    """Expand records that contain multiple list items into separate records."""
    # For now, pass through; expansion logic can be added as needed
    return segment


def _split_colon_bullet_segments(
    buffer: tuple[Record, ...],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[tuple[Record, ...], ...] | None:
    """Split buffer at colon-introduced bullet list boundaries."""
    if not buffer or len(buffer) < 2:
        return None
    # Check for colon-introduced lists that should form their own segment
    # For now, return None to indicate no splitting needed
    return None


# ---------------------------------------------------------------------------
# Segment emission
# ---------------------------------------------------------------------------


def _emit_individual_records(
    segment: tuple[Record, ...],
    start_index: int,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[Record, ...]:
    """Emit each record in segment individually with chunk indices."""
    return tuple(
        (
            page,
            _with_chunk_index(block, start_index + offset),
            _normalize_numbered_list_text(text, strategy=strategy),
        )
        for offset, (page, block, text) in enumerate(segment)
    )


def _merged_segment_record(
    segment: tuple[Record, ...],
    *,
    start_index: int,
    overlap: int,
    strategy: BulletHeuristicStrategy | None = None,
) -> Record | None:
    """Merge segment records into a single record if possible."""
    trimmed = _apply_overlap_within_segment(segment, overlap)
    if not trimmed:
        return None
    joined = _join_record_texts(trimmed, strategy=strategy)
    if not joined or len(joined) > SOFT_LIMIT:
        return None
    merged = _merge_record_block(list(trimmed), joined)
    first_page = trimmed[0][0]
    return first_page, _with_chunk_index(merged, start_index), joined


def _segment_allows_list_overflow(
    segment: tuple[Record, ...],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Check if segment contains a numbered list continuation that allows overflow."""
    heuristics = _resolve_bullet_strategy(strategy)
    if not any(_text_has_number_two(text) for _, _, text in segment):
        return False
    last_num: int | None = None
    for _, _, text in segment:
        first_num = _first_list_number(text, strategy=heuristics)
        last_in_text = _last_list_number(text, strategy=heuristics)
        if first_num is not None:
            if last_num is not None and first_num == last_num + 1:
                return True
            last_num = last_in_text or first_num
            continue
        if last_in_text is not None and last_num is not None:
            if last_in_text == last_num + 1:
                return True
            if last_in_text >= last_num:
                last_num = last_in_text
            continue
        if last_num is None and last_in_text is not None:
            last_num = last_in_text
    return False


def _segment_is_colon_list(
    segment: tuple[Record, ...],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Return True if segment is a colon-introduced list that should stay together."""
    if len(segment) < 2:
        return False
    for i in range(len(segment) - 1):
        _, _, prev_text = segment[i]
        _, next_block, next_text = segment[i + 1]
        if prev_text.rstrip().endswith(":") and _starts_list_like(
            next_block, next_text, strategy=strategy
        ):
            return True
    return False


def _emit_segment_records(
    segment: tuple[Record, ...],
    *,
    start_index: int,
    overlap: int,
    resolved_limit: int | None,
    hard_limit: int | None,
    overflow: bool,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[Record, ...]:
    """Emit segment as merged or individual records respecting limits."""
    if not segment:
        return tuple()
    segment = _expand_segment_records(segment, strategy=strategy)
    words, dense, effective = _segment_totals(segment)
    exceeds_soft = resolved_limit is not None and effective > resolved_limit
    exceeds_hard = hard_limit is not None and effective > hard_limit
    overflow_active = overflow and (exceeds_soft or exceeds_hard)
    if overflow_active or len(segment) == 1:
        return _emit_individual_records(segment, start_index, strategy=strategy)
    if exceeds_soft or exceeds_hard:
        allows_overflow = _segment_allows_list_overflow(
            segment, strategy=strategy
        ) or _segment_is_colon_list(segment, strategy=strategy)
        if allows_overflow:
            merged = _merged_segment_record(
                segment,
                start_index=start_index,
                overlap=overlap,
                strategy=strategy,
            )
            if merged is not None:
                return (merged,)
        return _emit_individual_records(segment, start_index, strategy=strategy)
    merged = _merged_segment_record(
        segment,
        start_index=start_index,
        overlap=overlap,
        strategy=strategy,
    )
    if merged is None:
        return _emit_individual_records(segment, start_index, strategy=strategy)
    return (merged,)


def _emit_buffer_segments(
    buffer: tuple[Record, ...],
    *,
    start_index: int,
    overlap: int,
    resolved_limit: int | None,
    hard_limit: int | None,
    overflow: bool,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[Record, ...]:
    """Emit buffer as one or more segments."""
    if not buffer:
        return tuple()
    segments = _split_colon_bullet_segments(buffer, strategy=strategy) or (buffer,)
    enumerated = _enumerate_segments(segments)
    return tuple(
        emission
        for offset, segment in enumerated
        for emission in _emit_segment_records(
            segment,
            start_index=start_index + offset,
            overlap=overlap,
            resolved_limit=resolved_limit,
            hard_limit=hard_limit,
            overflow=overflow,
            strategy=strategy,
        )
    )


# ---------------------------------------------------------------------------
# Cross-page and list boundary detection
# ---------------------------------------------------------------------------


def _text_has_number_two(text: str) -> bool:
    """Check if text contains the number 2 as a list item."""
    return bool(_NUMBERED_TWO_RE.search(text))


def _buffer_has_number_two(buffer: tuple[Record, ...]) -> bool:
    """Check if any record in buffer contains number 2 as list item."""
    return any(_text_has_number_two(text) for _, _, text in buffer)


def _allow_cross_page_list(
    previous: Record,
    current: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Allow merging across page boundary for numbered list continuations."""
    prev_page, prev_block, prev_text = previous
    page, block, text = current
    if (
        prev_page is None
        or page is None
        or not isinstance(prev_page, int)
        or not isinstance(page, int)
        or page != prev_page + 1
    ):
        return False
    heuristics = _resolve_bullet_strategy(strategy)
    lead = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not lead or not heuristics.starts_with_number(lead):
        return False
    prev_num = _last_list_number(prev_text, strategy=heuristics)
    next_num = _first_list_number(text, strategy=heuristics)
    if prev_num is None or next_num is None:
        return False
    return next_num == prev_num + 1


def _buffer_last_list_number(
    buffer: tuple[Record, ...],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> int | None:
    """Find the last numbered list item number in the buffer."""
    heuristics = _resolve_bullet_strategy(strategy)
    for _, _, text in reversed(buffer):
        found = _last_list_number(text, strategy=heuristics)
        if found is not None:
            return found
    return None


def _record_starts_numbered_item(
    record: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Check if record starts with a numbered list item."""
    _, _, text = record
    heuristics = _resolve_bullet_strategy(strategy)
    lead = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return bool(lead and heuristics.starts_with_number(lead))


def _allow_list_overflow(
    buffer: tuple[Record, ...],
    record: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Allow overflow when record continues a numbered list from buffer."""
    if not buffer:
        return False
    heuristics = _resolve_bullet_strategy(strategy)
    if not _record_starts_numbered_item(record, strategy=heuristics):
        return False
    if not (_buffer_has_number_two(buffer) or _text_has_number_two(record[2])):
        return False
    prev_num = _buffer_last_list_number(buffer, strategy=heuristics)
    next_num = _first_list_number(record[2], strategy=heuristics)
    if prev_num is None or next_num is None or next_num != prev_num + 1:
        return False
    projected_len = sum(len(text) for _, _, text in buffer) + len(record[2])
    return projected_len <= SOFT_LIMIT


def _allow_colon_list_overflow(
    buffer: tuple[Record, ...],
    record: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Allow overflow when buffer ends with colon introducing a list."""
    if not buffer:
        return False
    _, prev_block, prev_text = buffer[-1]
    _, block, text = record
    if not prev_text.rstrip().endswith(":"):
        return False
    if not _starts_list_like(block, text, strategy=strategy):
        return False
    projected_len = sum(len(t) for _, _, t in buffer) + len(text)
    return projected_len <= SOFT_LIMIT


def _page_or_footer_boundary(
    buffer: tuple[Record, ...],
    record: Record,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Check if record represents a page or footer boundary from buffer."""
    if not buffer:
        return False
    prev_page, prev_block, _ = buffer[-1]
    page, block, _ = record
    if block.get("has_heading_prefix", False):
        return True
    if prev_page != page:
        return not _allow_cross_page_list(buffer[-1], record, strategy=strategy)
    prev_is_footer = _record_is_footer_candidate(buffer[-1], strategy=strategy)
    current_is_footer = _record_is_footer_candidate(record, strategy=strategy)
    return (current_is_footer != prev_is_footer) and (current_is_footer or prev_is_footer)


# ---------------------------------------------------------------------------
# Projected overflow
# ---------------------------------------------------------------------------


def _projected_overflow(
    state: "_CollapseEmitter",
    block: "Block",
    text: str,
    counts: tuple[int, int, int],
) -> str | None:
    """Determine if adding text would cause overflow."""
    words = state.running_words + counts[0]
    dense = state.running_dense + counts[1]
    effective = max(words, dense)
    exceeds_soft = state.resolved_limit is not None and effective > state.resolved_limit
    exceeds_hard = state.hard_limit is not None and effective > state.hard_limit
    if exceeds_hard:
        last_text = state.buffer[-1][2].rstrip()
        if _ENDS_SENTENCE.search(last_text) and not _starts_list_like(
            block,
            text,
            strategy=state.strategy,
        ):
            return "flush"
        return "overflow"
    if exceeds_soft:
        return "overflow"
    return None


# ---------------------------------------------------------------------------
# CollapseEmitter state machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CollapseEmitter:
    """Immutable state machine for collapsing records into segments."""

    resolved_limit: int | None
    hard_limit: int | None
    overlap: int
    buffer: tuple[Record, ...] = tuple()
    running_words: int = 0
    running_dense: int = 0
    start_index: int | None = None
    outputs: tuple[Record, ...] = tuple()
    strategy: BulletHeuristicStrategy | None = None

    def append(
        self,
        idx: int,
        record: Record,
        counts: tuple[int, int, int],
    ) -> "_CollapseEmitter":
        """Append record to buffer, updating running totals."""
        start_index = self.start_index if self.buffer else idx
        words = self.running_words + counts[0] if self.buffer else counts[0]
        dense = self.running_dense + counts[1] if self.buffer else counts[1]
        return replace(
            self,
            buffer=self.buffer + (record,),
            running_words=words,
            running_dense=dense,
            start_index=start_index,
        )

    def flush(self, *, overflow: bool = False) -> "_CollapseEmitter":
        """Flush buffer to outputs."""
        if not self.buffer:
            return self
        first_index = self.start_index if self.start_index is not None else 0
        produced = _emit_buffer_segments(
            self.buffer,
            start_index=first_index,
            overlap=self.overlap,
            resolved_limit=self.resolved_limit,
            hard_limit=self.hard_limit,
            overflow=overflow,
            strategy=self.strategy,
        )
        return replace(
            self,
            buffer=tuple(),
            running_words=0,
            running_dense=0,
            start_index=None,
            outputs=self.outputs + produced,
        )

    def emit_single(self, idx: int, record: Record) -> "_CollapseEmitter":
        """Emit a single record directly to outputs."""
        page, block, text = record
        entry: Record = (page, _with_chunk_index(block, idx), text)
        return replace(self, outputs=self.outputs + (entry,))


# ---------------------------------------------------------------------------
# List boundary detection
# ---------------------------------------------------------------------------


def _should_emit_list_boundary(
    previous: Record,
    block: "Block",
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Determine if a list boundary should trigger a segment flush."""
    _, prev_block, prev_text = previous
    if prev_text.rstrip().endswith(":"):
        return False
    if _starts_list_like(prev_block, prev_text, strategy=strategy):
        return False
    prev_num = _last_list_number(prev_text, strategy=strategy)
    next_num = _first_list_number(text, strategy=strategy)
    if prev_num is not None and next_num is not None and next_num == prev_num + 1:
        return False
    return not _record_is_footer_candidate(previous, strategy=strategy)


# ---------------------------------------------------------------------------
# Collapse step and main collapse function
# ---------------------------------------------------------------------------


def _collapse_step(state: _CollapseEmitter, item: tuple[int, Record]) -> _CollapseEmitter:
    """Process one record in the collapse sequence."""
    idx, record = item
    _, block, text = record
    counts = _effective_counts(text)
    _, _, effective = counts
    if _page_or_footer_boundary(
        state.buffer,
        record,
        strategy=state.strategy,
    ):
        state = state.flush()
    if (state.resolved_limit is not None and effective > state.resolved_limit) or (
        state.hard_limit is not None and effective > state.hard_limit
    ):
        if state.buffer and _allow_list_overflow(state.buffer, record, strategy=state.strategy):
            return state.append(idx, record, counts)
        return state.flush().emit_single(idx, record)
    if (
        state.buffer
        and _starts_list_like(block, text, strategy=state.strategy)
        and _should_emit_list_boundary(
            state.buffer[-1],
            block,
            text,
            strategy=state.strategy,
        )
    ):
        state = state.flush()
    if state.buffer:
        overflow_action = _projected_overflow(state, block, text, counts)
        if overflow_action is not None:
            allow_overflow = _allow_list_overflow(
                state.buffer, record, strategy=state.strategy
            ) or _allow_colon_list_overflow(state.buffer, record, strategy=state.strategy)
            if not allow_overflow:
                state = state.flush(overflow=overflow_action == "overflow")
    return state.append(idx, record, counts)


def _maybe_merge_dense_page(
    records: "Iterable[Record]",
    options: "SplitOptions | None",
    limit: int | None,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[Record, ...]:
    """Merge records from a single dense page if within limits."""
    sequence = tuple(records)
    if len(sequence) <= 1:
        return sequence
    if options is None or options.chunk_size <= 0:
        return sequence
    if {page for page, _, _ in sequence} != {sequence[0][0]}:
        return sequence
    dense_total = sum(_effective_counts(text)[1] for _, _, text in sequence)
    if dense_total > options.chunk_size:
        return sequence
    word_total = sum(_effective_counts(text)[0] for _, _, text in sequence)
    if word_total > options.chunk_size > 0:
        return sequence
    if limit is not None and word_total <= limit:
        return sequence
    merged_text = _join_record_texts(sequence, strategy=strategy)
    if not merged_text:
        return sequence
    merged_block = _merge_record_block(list(sequence), merged_text)
    return ((sequence[0][0], merged_block, merged_text),)


def collapse_records(
    records: "Iterable[Record]",
    options: "SplitOptions | None" = None,
    limit: int | None = None,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> Iterator[Record]:
    """Collapse records into segments based on limits and boundaries.

    This is the main entry point for the collapse logic.
    """
    heuristics = _resolve_bullet_strategy(strategy)
    seq = tuple(_strip_footer_suffixes(tuple(records), strategy=heuristics))
    if not seq:
        return
    resolved = _resolved_limit(options, limit)
    if resolved is None:
        yield from (
            (page, _with_chunk_index(block, idx), text)
            for idx, (page, block, text) in enumerate(seq)
        )
        return

    hard = _hard_limit(options, resolved)
    overlap = _overlap_value(options)
    initial = _CollapseEmitter(
        resolved_limit=resolved,
        hard_limit=hard,
        overlap=overlap,
        strategy=heuristics,
    )
    final_state = reduce(_collapse_step, enumerate(seq), initial).flush()
    merged_outputs = _maybe_merge_dense_page(
        final_state.outputs,
        options,
        limit,
        strategy=heuristics,
    )
    yield from (
        (page, _with_chunk_index(block, idx), text)
        for idx, (page, block, text) in enumerate(merged_outputs)
    )


# Alias for backward compatibility
_collapse_records = collapse_records


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Main collapse function
    "collapse_records",
    "_collapse_records",
    # CollapseEmitter
    "_CollapseEmitter",
    # Emission functions
    "_emit_buffer_segments",
    "_emit_segment_records",
    "_emit_individual_records",
    "_merged_segment_record",
    # Segment utilities
    "_segment_totals",
    "_segment_offsets",
    "_enumerate_segments",
    "_segment_allows_list_overflow",
    "_segment_is_colon_list",
    # Limit computation
    "_resolved_limit",
    "_hard_limit",
    "_overlap_value",
    # Text utilities
    "_effective_counts",
    "_join_record_texts",
    "_normalize_numbered_list_text",
    # Boundary detection
    "_page_or_footer_boundary",
    "_allow_cross_page_list",
    "_allow_list_overflow",
    "_allow_colon_list_overflow",
    "_should_emit_list_boundary",
    # Helpers
    "_buffer_last_list_number",
    "_record_starts_numbered_item",
    "_text_has_number_two",
    "_buffer_has_number_two",
    "_projected_overflow",
    "_collapse_step",
    "_maybe_merge_dense_page",
    "_resolve_bullet_strategy",
]
