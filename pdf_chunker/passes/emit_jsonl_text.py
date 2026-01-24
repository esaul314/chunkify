"""Pure text manipulation utilities for emit_jsonl.

These are stateless, configuration-free functions for text analysis and transformation.
Extracted from emit_jsonl.py for clarity and testability.
"""

from __future__ import annotations

import re
from itertools import dropwhile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTINUATION_LEADS = frozenset(
    word.lower()
    for word in (
        "And",
        "But",
        "So",
        "However",
        "Therefore",
        "Yet",
        "Still",
        "Also",
        "Meanwhile",
        "Additionally",
        "Then",
        "Thus",
        "Instead",
        "Nevertheless",
        "Nonetheless",
        "Consequently",
        "Moreover",
    )
)

_LEAD_WORD_RE = re.compile(r"[\w']+")
_LAST_SENTENCE_RE = re.compile(r"([^.?!]+[.?!][\"')\]]*)\s*$", re.DOTALL)
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*")
_SENTENCE_TERMINATORS = ".?!"
_CLOSING_PUNCTUATION = "\"')]}"

_CAPTION_PREFIXES = (
    "figure",
    "fig.",
    "table",
    "tbl.",
    "image",
    "img.",
    "diagram",
)

_CAPTION_LABEL_RE = re.compile(
    rf"^(?:{'|'.join(_CAPTION_PREFIXES)})\s+(?:[a-z]*\d[\w.-]*|[ivxlcdm]+)"
)


# ---------------------------------------------------------------------------
# Basic text utilities
# ---------------------------------------------------------------------------


def word_count(text: str) -> int:
    """Count words in text using word boundary matching."""
    return len(re.findall(r"\b\w+\b", text))


def normalize(text: str) -> str:
    """Normalize text by collapsing whitespace and lowercasing."""
    return re.sub(r"\s+", " ", text).strip().lower()


def contains(haystack: str, needle: str) -> bool:
    """Check if needle is contained in haystack (non-empty needle required)."""
    return bool(needle and needle in haystack)


def first_non_empty_line(text: str) -> str:
    """Return the first non-empty line from text."""
    return next((ln for ln in text.splitlines() if ln.strip()), "")


def last_non_empty_line(text: str) -> str:
    """Return the last non-empty line from text."""
    return next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")


def trim_trailing_empty(lines: list[str]) -> list[str]:
    """Remove trailing empty lines from a list."""
    return list(reversed(list(dropwhile(lambda ln: not ln.strip(), reversed(lines)))))


def leading_word(text: str) -> str:
    """Extract the first word from text, lowercased."""
    match = _LEAD_WORD_RE.match(text)
    return match.group(0).lower() if match else ""


def starts_with_continuation(text: str) -> bool:
    """Check if text starts with a continuation word (And, But, However, etc.)."""
    return leading_word(text.lstrip()) in _CONTINUATION_LEADS


def last_sentence(text: str) -> str | None:
    """Extract the last sentence from text."""
    stripped = text.strip()
    if not stripped:
        return None
    match = _LAST_SENTENCE_RE.search(stripped)
    return match.group(1).strip() if match else stripped


# ---------------------------------------------------------------------------
# Sentence boundary detection
# ---------------------------------------------------------------------------


def has_sentence_ending(text: str) -> bool:
    """Check if text ends with sentence-terminating punctuation."""
    stripped = text.rstrip()
    trimmed = stripped.rstrip(_CLOSING_PUNCTUATION)
    return bool(trimmed) and trimmed[-1] in _SENTENCE_TERMINATORS


def starts_mid_sentence(text: str) -> bool:
    """Check if text appears to start mid-sentence (lowercase or no capital)."""
    stripped = text.strip()
    return bool(stripped) and re.match(r"^[\"'(]*[A-Z0-9]", stripped) is None


# ---------------------------------------------------------------------------
# Overlap and deduplication utilities
# ---------------------------------------------------------------------------


def overlap_len(prev_lower: str, curr_lower: str) -> int:
    """Find length of overlapping suffix of prev that matches prefix of curr."""
    length = min(len(prev_lower), len(curr_lower))
    return next(
        (i for i in range(length, 0, -1) if prev_lower.endswith(curr_lower[:i])),
        0,
    )


def prefix_contained_len(haystack: str, needle: str) -> int:
    """Find length of needle prefix that is contained in haystack as complete sentence."""
    length = len(needle)

    def _match(index: int) -> bool:
        segment = needle[:index]
        if not segment.strip():
            return False
        if index < length and not needle[index].isspace():
            return False
        if not has_sentence_ending(segment):
            return False
        position = haystack.find(segment)
        if position == -1:
            return False
        preceding = haystack[position - 1] if position else ""
        return not preceding.isalnum()

    return next((idx for idx in range(length, 0, -1) if _match(idx)), 0)


# ---------------------------------------------------------------------------
# Caption detection utilities
# ---------------------------------------------------------------------------


def contains_caption_line(text: str) -> bool:
    """Check if text contains a line that looks like a figure/table caption."""
    lines = tuple(line.strip().lower() for line in text.splitlines())
    return any(line.startswith(prefix) for prefix in _CAPTION_PREFIXES for line in lines)


def looks_like_caption_label(text: str) -> bool:
    """Check if text looks like a caption label (Figure 1, Table 2, etc.)."""
    normalized = text.strip().lower()
    return bool(normalized and _CAPTION_LABEL_RE.match(normalized))


def caption_overlap(prev: str, curr: str, prefix: str) -> bool:
    """Check if overlap between prev and curr is a caption continuation."""
    snippet = prefix.strip()
    if not snippet:
        return False
    first_line = first_non_empty_line(curr).strip()
    if not first_line or not first_line.lower().startswith(snippet.lower()):
        return False
    return contains_caption_line(prev)


# ---------------------------------------------------------------------------
# Overlap trimming
# ---------------------------------------------------------------------------


def trim_overlap(prev: str, curr: str) -> str:
    """Remove duplicated prefix from curr that already exists in prev.

    This function is part of the overlap/dedup pipeline and runs at chunk-level
    during _split(). It removes content from curr that already appears in prev.

    IMPORTANT: If curr is entirely contained in prev, returns empty string.
    This can cause silent data loss with repetitive/uniform content.
    See docs/emit_jsonl_refactoring_assessment.md "Overlap and Deduplication"
    for the full interaction model.

    Safeguards that preserve content:
    - Caption labels (Figure 1, Table 2) are never trimmed
    - Overlap must be <90% of curr length to trigger trimming
    - Alphanumeric boundaries prevent mid-word breaks

    Args:
        prev: Previous chunk text
        curr: Current chunk text to potentially trim

    Returns:
        curr with overlapping prefix removed, or "" if curr ⊂ prev
    """
    prev_lower, curr_lower = prev.lower(), curr.lower()
    # WARNING: Returns "" if entire chunk is substring of previous chunk.
    # This is intentional for dedup but can cause data loss with uniform text.
    if contains(prev_lower, curr_lower):
        return ""
    overlap = overlap_len(prev_lower, curr_lower)
    if overlap and overlap < len(curr) * 0.9:
        prefix = curr[:overlap]
        if caption_overlap(prev, curr, prefix):
            return curr
        prev_index = len(prev) - overlap
        prev_char = prev[prev_index - 1] if prev_index > 0 else ""
        next_non_space = next((ch for ch in curr[overlap:] if not ch.isspace()), "")
        stripped_prefix = prefix.strip()
        words = re.findall(r"\b\w+\b", stripped_prefix)
        single_title = len(words) == 1 and words[0][0].isupper() and words[0][1:].islower()
        if looks_like_caption_label(stripped_prefix):
            return curr
        if prev_char.isalnum():
            return curr
        if single_title and (next_non_space.islower() or next_non_space.isdigit()):
            return curr
        return curr[overlap:].lstrip()
    prefix = curr_lower.split("\n\n", 1)[0]
    return curr[len(prefix) :].lstrip() if contains(prev_lower, prefix) else curr
