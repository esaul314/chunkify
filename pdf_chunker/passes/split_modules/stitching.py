"""Block stitching and merging for split_semantic pass.

This module handles stitching adjacent blocks together based on
continuation patterns, Q&A sequences, and other merge heuristics.

Key responsibilities:
1. Stitch block continuations (lowercase leads, Q&A sequences)
2. Merge record blocks into unified blocks
3. Apply chunk indexing for tracking

Extracted from split_semantic.py for modularity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import reduce
from typing import TYPE_CHECKING, Any

from pdf_chunker.passes.sentence_fusion import (
    _is_continuation_lead,
    _is_qa_sequence_continuation,
    _last_sentence,
)
from pdf_chunker.passes.split_modules.lists import starts_list_like
from pdf_chunker.passes.split_semantic_lists import (
    _apply_envelope,
    _resolve_envelope,
)
from pdf_chunker.passes.transform_log import TransformationLog, maybe_record

if TYPE_CHECKING:
    from pdf_chunker.strategies.bullets import BulletHeuristicStrategy

Block = dict[str, Any]
Record = tuple[int, Block, str]


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

import logging

_logger = logging.getLogger(__name__)


def _warn_stitching_issue(message: str, *, page: int | None = None) -> None:
    """Log a stitching-related warning with optional page context."""
    if page is not None:
        _logger.debug("[page %d] %s", page, message)
    else:
        _logger.debug("%s", message)


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------


def _is_heading(block: Block) -> bool:
    """Check if a block is a heading based on its type or has_heading_prefix."""
    if not isinstance(block, Mapping):
        return False
    return block.get("type") == "heading" or block.get("has_heading_prefix", False)


# ---------------------------------------------------------------------------
# Block stitching
# ---------------------------------------------------------------------------


def stitch_block_continuations(
    seq: Iterable[Record],
    limit: int | None,
    *,
    strategy: BulletHeuristicStrategy | None = None,
    transform_log: TransformationLog | None = None,
) -> list[Record]:
    """Stitch adjacent blocks that form logical continuations.

    This function merges blocks based on several heuristics with explicit
    precedence order. Decisions are logged to transform_log for auditability.

    **Precedence Order (evaluated top to bottom):**

    1. **Q&A sequence continuations** (CRITICAL) - e.g., Q1: → Q2:
       Always merge. Takes priority over heading checks because Q&A
       sequences should stay together even when blocks have heading prefix.

    2. **Heading boundary** (BOUNDARY) - current block is a heading
       Never merge into headings. Preserves document structure.

    3. **Previous heading boundary** (BOUNDARY) - previous block is heading
       Never merge from headings (except Q&A, handled above).

    4. **List-like start** (HIGH) - current block starts a list
       Split to preserve list structure.

    5. **Continuation lead** (LOW) - lowercase sentence fragment
       Merge if text starts with continuation pattern and context available.

    6. **Default** - no pattern matched
       Keep blocks separate (preserve original boundaries).

    All decisions are logged with reasons when transform_log is provided.
    See docs/MERGE_DECISIONS.md for full pattern reference.

    Args:
        seq: Sequence of (page, block, text) records
        limit: Optional word limit for context merging
        strategy: Bullet heuristic strategy for list detection
        transform_log: Optional log for recording transformations

    Returns:
        List of stitched records with merged continuations

    See Also:
        pdf_chunker.patterns.PatternRegistry.should_merge: Pattern-based decisions
        docs/MERGE_DECISIONS.md: Auto-generated decision reference
    """

    def _consume(
        acc: list[Record],
        cur: Record,
    ) -> list[Record]:
        page, block, text = cur
        if not acc:
            return [*acc, cur]

        prev_text = acc[-1][2]

        # Check for Q&A sequence continuation FIRST (e.g., Q1: -> Q2:)
        # This takes priority over heading checks because Q&A sequences
        # should stay together even when the previous block has a heading prefix
        if _is_qa_sequence_continuation(prev_text, text):
            # Merge Q&A sequences together
            merged = f"{prev_text}\n{text}".strip()
            maybe_record(
                transform_log,
                "merged",
                "split_semantic._stitch",
                "qa_sequence_merge",
                prev_text[:100],
                merged[:100],
                details={"page": acc[-1][0], "next_page": page},
            )
            return [*acc[:-1], (acc[-1][0], acc[-1][1], merged)]

        # Don't merge heading blocks with other content
        if _is_heading(block):
            maybe_record(
                transform_log,
                "boundary",
                "split_semantic._stitch",
                "heading_boundary_kept",
                prev_text[:100],
                prev_text[:100],
                details={"page": page, "reason": "current_block_is_heading"},
            )
            return [*acc, cur]
        # Don't merge into heading blocks (unless it's a Q&A sequence, handled above)
        if _is_heading(acc[-1][1]):
            maybe_record(
                transform_log,
                "boundary",
                "split_semantic._stitch",
                "heading_boundary_kept",
                prev_text[:100],
                prev_text[:100],
                details={"page": acc[-1][0], "reason": "previous_block_is_heading"},
            )
            return [*acc, cur]
        if starts_list_like(block, text, strategy=strategy):
            maybe_record(
                transform_log,
                "boundary",
                "split_semantic._stitch",
                "list_boundary_kept",
                prev_text[:100],
                prev_text[:100],
                details={"page": page},
            )
            return [*acc, cur]
        lead = text.lstrip()

        if not lead or not _is_continuation_lead(lead):
            return [*acc, cur]
        context = _last_sentence(prev_text)
        if not context or text.lstrip().startswith(context):
            return [*acc, cur]
        context_words = tuple(context.split())
        text_words = tuple(text.split())
        if limit is not None and len(text_words) + len(context_words) > limit:
            _warn_stitching_issue(
                "continuation context skipped due to chunk limit",
                page=acc[-1][0],
            )
            merged = f"{prev_text} {text}".strip()
            maybe_record(
                transform_log,
                "merged",
                "split_semantic._stitch",
                "continuation_merge_limit_skip",
                prev_text[:100],
                merged[:100],
                details={"page": acc[-1][0], "context_words": len(context_words)},
            )
            return [*acc[:-1], (acc[-1][0], acc[-1][1], merged)]
        enriched = f"{context} {text}".strip()
        maybe_record(
            transform_log,
            "merged",
            "split_semantic._stitch",
            "continuation_context_prepend",
            text[:100],
            enriched[:100],
            details={"page": page, "context_len": len(context)},
        )
        return [*acc, (page, block, enriched)]

    return reduce(_consume, seq, [])


# ---------------------------------------------------------------------------
# Block merging
# ---------------------------------------------------------------------------


def merge_record_block(records: list[Record], text: str) -> Block:
    """Merge multiple records into a single unified block.

    Combines blocks from multiple records, preserving source block
    information for traceability.

    Args:
        records: List of (page, block, text) records to merge
        text: The merged text content

    Returns:
        A unified block with source_blocks tracking
    """
    blocks = tuple(block for _, block, _ in records)
    envelope = _resolve_envelope(blocks)
    first = blocks[0] if blocks else {}
    merged = _apply_envelope(first, text, envelope)
    if not blocks:
        return merged

    def _snapshot(candidate: Block) -> Block:
        if not isinstance(candidate, Mapping):
            return {}
        base = {**candidate}
        base.pop("source_blocks", None)
        return base

    existing_sources = (
        tuple(
            _snapshot(source)
            for source in first.get("source_blocks", ())
            if isinstance(first, Mapping) and isinstance(source, Mapping)
        )
        if isinstance(first, Mapping)
        else tuple()
    )
    snapshots = existing_sources + tuple(
        _snapshot(block) for block in blocks if isinstance(block, Mapping)
    )
    return {**merged, "source_blocks": snapshots} if snapshots else merged


# ---------------------------------------------------------------------------
# Chunk indexing
# ---------------------------------------------------------------------------


def with_chunk_index(block: Block, index: int) -> Block:
    """Add chunk start index to a block for tracking.

    Args:
        block: The block to annotate
        index: The chunk start index

    Returns:
        Block with _chunk_start_index field added
    """
    return {**block, "_chunk_start_index": index}


# ---------------------------------------------------------------------------
# Heading detection utility (exported for use by split_semantic.py)
# ---------------------------------------------------------------------------


def is_heading(block: Block) -> bool:
    """Check if a block is a heading.

    Public API wrapper around _is_heading for external use.
    """
    return _is_heading(block)
