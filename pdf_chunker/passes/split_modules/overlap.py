"""Boundary overlap management for split_semantic pass.

This module handles the overlap between consecutive chunks to provide
context continuity in RAG (Retrieval-Augmented Generation) applications.

Why Overlap Matters
-------------------
RAG systems retrieve chunks based on query similarity. When a relevant passage
spans two chunks, neither chunk alone may score highly. Overlap duplicates
text at boundaries so queries hitting the boundary region match both chunks,
improving retrieval recall.

Key Responsibilities
--------------------
1. Restore missing overlap words when chunks were split
2. Trim duplicate sentence prefixes at chunk boundaries
3. Manage overlap window calculations

Design Rationale
----------------
- Industry heuristic: 10-30% overlap relative to chunk_size
- Current default: 100 words / 400 word chunks = 25%
- Overlap words are prepended to each chunk (except the first) from the
  tail of the previous chunk

See Also
--------
- docs/RAG_OVERLAP_ALIGNMENT_PLAN.md — Full design rationale and tuning guidance
- pipeline_rag.yaml — User-facing overlap configuration

Extracted from split_semantic.py for modularity.
"""

from __future__ import annotations

import re
from itertools import accumulate
from typing import TYPE_CHECKING

from pdf_chunker.passes.sentence_fusion import _last_sentence
from pdf_chunker.passes.split_semantic_lists import _looks_like_caption

if TYPE_CHECKING:
    pass

# Token pattern for word splitting
_TOKEN_PATTERN = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# Word splitting utilities
# ---------------------------------------------------------------------------


def split_words(text: str) -> tuple[str, ...]:
    """Split text into a tuple of whitespace-separated words."""
    return tuple(text.split())


# ---------------------------------------------------------------------------
# Overlap window calculation
# ---------------------------------------------------------------------------


def overlap_window(
    prev_words: tuple[str, ...],
    current_words: tuple[str, ...],
    limit: int,
) -> int:
    """Find the overlap window size between previous and current word tuples.

    Returns the number of words that match between the end of prev_words
    and the start of current_words, up to limit.
    """
    match_limit = min(limit, len(prev_words), len(current_words))
    return next(
        (size for size in range(match_limit, 0, -1) if prev_words[-size:] == current_words[:size]),
        0,
    )


def overlap_text(words: tuple[str, ...], size: int) -> str:
    """Return the first `size` words joined as text."""
    return " ".join(words[:size]).strip()


# ---------------------------------------------------------------------------
# Overlap token filtering
# ---------------------------------------------------------------------------


def is_overlap_token(token: str) -> bool:
    """Check if a token should be included in overlap restoration.

    Filters out empty tokens and non-content characters, but preserves
    bullet markers and list punctuation.
    """
    stripped = token.strip()
    if not stripped:
        return False
    if any(char.isalnum() for char in stripped):
        return True
    return stripped in {"•", "-", "–", "—"}


# ---------------------------------------------------------------------------
# Overlap restoration
# ---------------------------------------------------------------------------


def missing_overlap_prefix(
    previous_words: tuple[str, ...],
    current_words: tuple[str, ...],
    overlap: int,
) -> tuple[str, ...]:
    """Calculate words missing from the overlap prefix.

    When chunks are split, some overlap words may be missing from the
    start of the current chunk. This function calculates which words
    should be prepended.
    """
    if overlap <= 0 or not previous_words or not current_words:
        return tuple()

    window = min(overlap, len(previous_words))
    matched = overlap_window(previous_words, current_words, window)
    if matched >= window:
        return tuple()

    overlap_words = previous_words[-window:]
    missing = overlap_words[: window - matched]
    filtered = tuple(token for token in missing if is_overlap_token(token))
    return filtered


def prepend_words(words: tuple[str, ...], text: str) -> str:
    """Prepend words to text with appropriate spacing."""
    if not words:
        return text
    prefix = " ".join(words)
    if not prefix:
        return text
    glue = "" if not text or text[0].isspace() else " "
    return f"{prefix}{glue}{text}"


def restore_chunk_overlap(previous: str, current: str, overlap: int) -> str:
    """Restore missing overlap words between chunks.

    Examines the boundary between previous and current chunks and adds
    any missing overlap words to the start of current.
    """
    if not previous or not current:
        return current

    previous_words = split_words(previous)
    current_words = split_words(current)
    missing = missing_overlap_prefix(previous_words, current_words, overlap)
    if missing:
        return prepend_words(missing, current)
    return trim_sentence_prefix(previous, current)


def restore_overlap_words(chunks: list[str], overlap: int) -> list[str]:
    """Inject overlap words at chunk boundaries for RAG retrieval.

    Overlap ensures queries spanning chunk boundaries match both chunks.
    The overlap words are prepended to each chunk (except the first) from
    the tail of the previous chunk.

    Parameters
    ----------
    chunks : list[str]
        List of chunk texts to process.
    overlap : int
        Number of words to overlap between consecutive chunks.

    Returns
    -------
    list[str]
        Chunks with overlap words prepended where needed.

    Design rationale
    ----------------
    See docs/RAG_OVERLAP_ALIGNMENT_PLAN.md for full explanation.
    Industry heuristic: 10-30% overlap relative to chunk_size.
    Current default: 100 words / 400 word chunks = 25%.
    """
    if overlap <= 0:
        return chunks

    restored = accumulate(
        chunks,
        lambda previous, current: restore_chunk_overlap(previous, current, overlap),
        initial="",
    )
    next(restored, None)  # Skip initial empty string
    return list(restored)


# ---------------------------------------------------------------------------
# Sentence prefix trimming
# ---------------------------------------------------------------------------


def trim_sentence_prefix(previous_text: str, text: str) -> str:
    """Trim duplicate sentence prefix from text if it overlaps with previous.

    When a sentence ends in the previous chunk and is repeated at the start
    of the current chunk, this function removes the duplicate.
    """
    if not previous_text or not text:
        return text
    sentence = _last_sentence(previous_text)
    if not sentence:
        return text
    candidate = sentence.strip()
    if not candidate or candidate[-1] not in {".", "?", "!"}:
        return text
    # Don't trim chapter/section headings
    if re.search(r"Chapter\s+\d+", candidate):
        return text
    if not text.startswith(candidate):
        return text
    remainder = text[len(candidate) :]
    if not remainder or not remainder.strip():
        return text
    # Preserve structural markers
    match = re.search(
        r"((?:Chapter|Section|Part)\s+[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?)\.?$",
        candidate,
    )
    preserved = match.group(0) if match else ""
    leading_space = remainder.startswith(" ")
    gap = remainder[1:] if leading_space else remainder
    if preserved:
        if gap.startswith("\n"):
            return f"{preserved}{gap}"
        spacer = "" if not gap or gap[0].isspace() else " "
        return f"{preserved}{spacer}{gap}"
    return gap


# ---------------------------------------------------------------------------
# Boundary overlap trimming
# ---------------------------------------------------------------------------


def should_trim_overlap(segment: str) -> bool:
    """Check if an overlap segment should be trimmed.

    Only trim segments that end with sentence-ending punctuation.
    """
    return bool(segment) and segment[-1] in {".", "?", "!"}


def trim_tokens(text: str, count: int) -> str:
    """Remove the first `count` tokens from text."""
    matches = list(_TOKEN_PATTERN.finditer(text))
    if len(matches) >= count:
        cut = matches[count - 1].end()
        return text[cut:].lstrip(" ")
    return ""


def trim_boundary_overlap(prev_text: str, text: str, overlap: int) -> str:
    """Trim duplicate overlap at chunk boundaries.

    When text starts with words that overlap with the end of prev_text,
    trim those words to avoid duplication.
    """
    if overlap <= 0 or not prev_text or not text:
        return text
    previous_words = split_words(prev_text)
    current_words = split_words(text)
    window = min(overlap, len(previous_words))
    if not window:
        return text
    matched = overlap_window(previous_words, current_words, window)
    if not matched or len(current_words) <= matched:
        return text
    # Don't trim captions
    if _looks_like_caption(text):
        return text
    overlap_segment = overlap_text(current_words, matched)
    if not should_trim_overlap(overlap_segment):
        return text
    return trim_tokens(text, matched)
