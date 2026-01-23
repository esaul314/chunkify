from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import reduce
from itertools import accumulate, chain, dropwhile, repeat, takewhile
from typing import Any, cast

from pdf_chunker.framework import Artifact, register
from pdf_chunker.strategies.bullets import (
    BulletHeuristicStrategy,
    default_bullet_strategy,
)
from pdf_chunker.utils import _truncate_chunk

_log = logging.getLogger(__name__)

Row = dict[str, Any]
Doc = dict[str, Any]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmitConfig:
    """Centralized configuration for emit_jsonl pass.

    All thresholds and limits are defined here with sensible defaults.
    Use `from_env()` to read overrides from environment variables.
    """

    # Minimum words for a standalone chunk (below this, chunks are merged)
    min_words: int = 50

    # Metadata key name in JSONL output
    metadata_key: str = "metadata"

    # Hard limit for JSONL line size (characters)
    max_chars: int = 8000

    # Target chunk size for optimal RAG/LoRA performance (~320 words)
    target_chunk_chars: int = 2000

    # Threshold below which items are always merged forward (very short)
    very_short_words: int | None = None  # None = min(30, min_words)

    # Maximum character count for merged chunks
    max_merge_chars: int = 2000

    # Minimum word count for a standalone row
    min_row_words: int | None = None  # None = min(15, min_words)

    # Minimum word count below which rows MUST be merged (critical short)
    critical_short_words: int = 5

    # Enable debug logging for deduplication
    dedup_debug: bool = False

    @classmethod
    def from_env(cls) -> EmitConfig:
        """Create configuration from environment variables."""
        min_words = int(os.getenv("PDF_CHUNKER_JSONL_MIN_WORDS", "50"))

        very_short_explicit = os.getenv("PDF_CHUNKER_VERY_SHORT_WORDS")
        min_row_explicit = os.getenv("PDF_CHUNKER_MIN_ROW_WORDS")

        return cls(
            min_words=min_words,
            metadata_key=os.getenv("PDF_CHUNKER_JSONL_META_KEY", "metadata"),
            max_chars=int(os.getenv("PDF_CHUNKER_JSONL_MAX_CHARS", "8000")),
            target_chunk_chars=int(os.getenv("PDF_CHUNKER_TARGET_CHUNK_CHARS", "2000")),
            very_short_words=int(very_short_explicit) if very_short_explicit else None,
            max_merge_chars=int(os.getenv("PDF_CHUNKER_MAX_MERGE_CHARS", "2000")),
            min_row_words=int(min_row_explicit) if min_row_explicit else None,
            critical_short_words=int(os.getenv("PDF_CHUNKER_CRITICAL_SHORT_WORDS", "5")),
            dedup_debug=bool(os.getenv("PDF_CHUNKER_DEDUP_DEBUG")),
        )

    def get_very_short_threshold(self) -> int:
        """Compute effective very-short threshold."""
        if self.very_short_words is not None:
            return self.very_short_words
        return min(30, self.min_words)

    def get_min_row_words(self) -> int:
        """Compute effective minimum row words."""
        if self.min_row_words is not None:
            return self.min_row_words
        return min(15, self.min_words)


def _config() -> EmitConfig:
    """Get configuration from environment.

    Note: Reads env vars fresh each call to support test monkeypatching.
    The overhead is negligible for this use case.
    """
    return EmitConfig.from_env()


# ---------------------------------------------------------------------------
# Legacy compatibility wrappers (delegating to EmitConfig)
# ---------------------------------------------------------------------------


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


def _min_words() -> int:
    return _config().min_words


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _metadata_key() -> str:
    return _config().metadata_key


def _compat_chunk_id(chunk_id: str) -> str:
    match = re.search(r"_p(\d+)_c", chunk_id)
    if not match:
        return chunk_id
    page = max(int(match.group(1)) - 1, 0)
    return f"{chunk_id[: match.start(1)]}{page}{chunk_id[match.end(1) :]}"


def _max_chars() -> int:
    """Hard limit for JSONL line size."""
    return _config().max_chars


def _target_chunk_chars() -> int:
    """Target chunk size for optimal RAG/LoRA performance."""
    return _config().target_chunk_chars


def _resolve_bullet_strategy(
    strategy: BulletHeuristicStrategy | None,
) -> BulletHeuristicStrategy:
    return strategy or default_bullet_strategy()


def _is_list_line(
    line: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    heuristics = _resolve_bullet_strategy(strategy)
    return heuristics.starts_with_bullet(stripped) or heuristics.starts_with_number(stripped)


def _first_non_empty_line(text: str) -> str:
    return next((ln for ln in text.splitlines() if ln.strip()), "")


def _last_non_empty_line(text: str) -> str:
    return next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")


def _trim_trailing_empty(lines: list[str]) -> list[str]:
    return list(reversed(list(dropwhile(lambda ln: not ln.strip(), reversed(lines)))))


def _partition_preamble(lines: list[str]) -> tuple[list[str], list[str]]:
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


_LIST_GAP_RE = re.compile(r"\n{2,}(?=\s*(?:[-\*\u2022]|\d+\.))")


def _collapse_list_gaps(
    text: str,
    *,
    is_list_line: Callable[[str], bool] | None = None,
) -> str:
    predicate = is_list_line or _is_list_line

    def repl(match: re.Match[str]) -> str:
        prior = text[: match.start()]
        prev_line = prior.splitlines()[-1] if "\n" in prior else prior
        return "\n" if not predicate(prev_line) else match.group(0)

    return _LIST_GAP_RE.sub(repl, text)


def _split_inline_list_start(
    line: str,
    *,
    is_list_line: Callable[[str], bool] | None = None,
) -> tuple[str, str] | None:
    predicate = is_list_line or _is_list_line
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


def _reserve_for_list(
    text: str,
    limit: int,
    *,
    is_list_line: Callable[[str], bool] | None = None,
) -> tuple[str, str, str | None]:
    predicate = is_list_line or _is_list_line
    collapsed = _collapse_list_gaps(text, is_list_line=predicate)
    lines = collapsed.splitlines()

    inline = next(
        (
            (idx, result)
            for idx, line in enumerate(lines)
            if (result := _split_inline_list_start(line, is_list_line=predicate))
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
    trimmed_pre = _trim_trailing_empty(pre_lines)
    trailing_gaps = pre_lines[len(trimmed_pre) :]

    if not trimmed_pre:
        return collapsed, "", None

    pre_text = "\n".join(trimmed_pre)
    block_text = "\n".join(block_lines)
    combined_len = len(pre_text) + (1 if pre_text and block_text else 0) + len(block_text)
    if combined_len <= limit:
        return collapsed, "", None

    keep_lines, intro_lines = _partition_preamble(trimmed_pre)
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
    intro_line = _first_non_empty_line("\n".join(intro_lines)) if intro_lines else ""
    intro_hint = intro_line if intro_line.strip() else None
    return chunk_text, remainder, intro_hint


def _list_intro_start(text: str) -> int:
    """Return the index where a trailing list introduction begins."""

    return max(
        (
            pos + span
            for token, span in (("\n\n", 2), (". ", 2), ("! ", 2), ("? ", 2))
            if (pos := text.rfind(token)) != -1
        ),
        default=-1,
    )


def _peel_list_intro(text: str) -> tuple[str, str]:
    """Split ``text`` into non-intro content and the trailing list preamble."""

    stripped = text.rstrip()
    colon_idx = max(stripped.rfind(":"), stripped.rfind("："))
    if colon_idx == -1:
        return text, ""
    prefix = stripped[: colon_idx + 1]
    start = _list_intro_start(prefix)
    if start <= 0:
        return text, ""
    return prefix[:start].rstrip(), prefix[start:].lstrip()


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
_LEAD_WORD = re.compile(r"[\w']+")
_LAST_SENTENCE_RE = re.compile(r"([^.?!]+[.?!][\"')\]]*)\s*$", re.DOTALL)


def _leading_word(text: str) -> str:
    match = _LEAD_WORD.match(text)
    return match.group(0).lower() if match else ""


def _starts_with_continuation(text: str) -> bool:
    return _leading_word(text.lstrip()) in _CONTINUATION_LEADS


def _last_sentence(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    match = _LAST_SENTENCE_RE.search(stripped)
    return match.group(1).strip() if match else stripped


def _compose_intro_with_chunk(intro: str, chunk: str, separators: int) -> str:
    """Compose ``intro`` and ``chunk`` with controlled blank-line separators."""

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


def _prepend_intro(intro: str, rest: str) -> str:
    """Attach ``intro`` ahead of ``rest`` while normalizing spacing."""

    intro_core = intro.strip("\n")
    if not rest:
        return intro_core

    leading_newlines = len(rest) - len(rest.lstrip("\n"))
    tail = rest[leading_newlines:]
    if not intro_core:
        return tail.strip("\n")

    trailing_intro_newlines = len(intro) - len(intro.rstrip("\n"))
    separators = trailing_intro_newlines + (leading_newlines or 1)
    return _compose_intro_with_chunk(intro_core, tail, separators)


def _rebalance_lists(
    raw: str,
    rest: str,
    *,
    is_list_line: Callable[[str], bool] | None = None,
) -> tuple[str, str]:
    """Shift trailing context or list block into ``rest`` when it starts with a list."""

    predicate = is_list_line or _is_list_line

    if not rest or not predicate(_first_non_empty_line(rest)):
        return raw, rest

    lines = _trim_trailing_empty(raw.splitlines())
    trimmed = "\n".join(lines)
    has_list = any(predicate(ln) for ln in lines)

    if not has_list:
        kept, intro = _peel_list_intro(trimmed)
        if intro:
            return kept, _prepend_intro(intro, rest)

    # Determine split point: last non-list line if ``raw`` already contains list items,
    # otherwise the preceding blank line so that list introductions move with the list.
    # fmt: off
    idx = next(
        (
            i
            for i, ln in enumerate(reversed(lines))
            if (
                (ln.strip() and not predicate(ln)) if has_list else not ln.strip()
            )
        ),
        len(lines),
    )
    # fmt: on
    start = len(lines) - idx
    if not has_list and start == 0:
        return trimmed, rest
    block = lines[start:]
    if not block:
        return trimmed, rest

    moved = "\n".join(block).strip()
    kept = "\n".join(lines[:start]).rstrip()
    return kept, _prepend_intro(moved, rest)


def _truncate_with_remainder(text: str, limit: int) -> tuple[str, str]:
    if len(text) <= limit or limit <= 0:
        return text, ""

    if limit <= 100:
        prefix = text[:limit]
        chunk = prefix.rstrip() or prefix
        return chunk, text[len(chunk) :]

    truncate_point = limit - 100
    sentence_endings = (". ", ".\n", "! ", "!\n", "? ", "?\n")
    best_sentence = max(
        (
            pos
            for ending in sentence_endings
            if (pos := text.rfind(ending, 0, truncate_point)) > truncate_point * 0.7
        ),
        default=-1,
    )
    sentence_idx: int | None = best_sentence + 1 if best_sentence > 0 else None

    paragraph_idx_raw = text.rfind("\n\n", 0, truncate_point)
    paragraph_idx: int | None = (
        paragraph_idx_raw if paragraph_idx_raw > truncate_point * 0.7 else None
    )

    word_idx_raw = text.rfind(" ", 0, truncate_point)
    word_idx: int | None = word_idx_raw if word_idx_raw > truncate_point * 0.8 else None

    for idx in (sentence_idx, paragraph_idx, word_idx):
        if idx:
            chunk = text[:idx].rstrip()
            if chunk:
                return chunk, text[idx:]

    fallback = text[:truncate_point]
    chunk = fallback.rstrip() or fallback
    return chunk, text[len(chunk) :]


def _split(
    text: str,
    limit: int,
    *,
    is_list_line: Callable[[str], bool] | None = None,
) -> list[str]:
    """Yield ``text`` slices no longer than ``limit`` using soft boundaries."""

    predicate = is_list_line or _is_list_line

    def step(
        state: tuple[list[str], str, str | None], _: object
    ) -> tuple[list[str], str, str | None]:
        pieces, remaining, intro_hint = state
        if not remaining:
            return state

        candidate, rem, next_intro = _reserve_for_list(
            remaining,
            limit,
            is_list_line=predicate,
        )
        source = candidate or remaining
        first = _first_non_empty_line(source)
        second = source.splitlines()[1] if "\n" in source else ""
        is_list = predicate(first) or (predicate(second) and len(first) < limit)

        if is_list and len(source) > limit:
            suffix = f"\n{rem}" if rem else ""
            raw, rest = f"{source}{suffix}", ""
        else:
            raw, leftover = _truncate_with_remainder(source, limit)
            suffix = f"\n{rem}" if rem else ""
            rest = f"{leftover}{suffix}"
        raw, rest = (
            _collapse_list_gaps(
                raw,
                is_list_line=predicate,
            ),
            _collapse_list_gaps(rest, is_list_line=predicate),
        )
        raw, rest = _rebalance_lists(raw, rest, is_list_line=predicate)
        raw_intro_line = _first_non_empty_line(raw)
        rest_first_line = _first_non_empty_line(rest)
        if (
            intro_hint
            and intro_hint.strip()
            and raw_intro_line.strip() == intro_hint.strip()
            and rest_first_line
            and predicate(rest_first_line)
        ):
            rest_lines = rest.splitlines()
            block_lines = list(takewhile(lambda ln: not ln.strip() or predicate(ln), rest_lines))
            block_text = "\n".join(block_lines).lstrip("\n")
            if block_text:
                raw = f"{raw.rstrip()}\n{block_text}".strip("\n")
                rest = "\n".join(rest_lines[len(block_lines) :])
        raw_first_line = _first_non_empty_line(raw)
        skip_trim = bool(pieces) and (
            predicate(raw_first_line)
            or (intro_hint and intro_hint.strip() and raw_first_line.strip() == intro_hint.strip())
        )
        trimmed = raw if not pieces or skip_trim else _trim_overlap(pieces[-1], raw)
        if trimmed and trimmed.strip():
            if pieces and predicate(_first_non_empty_line(trimmed)):
                merged = f"{pieces[-1].rstrip()}\n{trimmed.lstrip()}"
                if len(merged) <= limit:  # noqa: SIM108
                    pieces = [*pieces[:-1], merged]
                else:
                    pieces = [*pieces, trimmed]
            else:
                pieces = [*pieces, trimmed]
        return pieces, rest.lstrip(), next_intro

    states: Iterable[tuple[list[str], str, str | None]] = accumulate(
        repeat(None),
        step,
        initial=([], text, None),
    )
    return next(p for p, r, _ in states if not r)


_INLINE_BULLET_RE = re.compile(r":\s*[•\-\*\u2022]\s+\w")
_INLINE_NUMBER_RE = re.compile(r":\s*\d+[.)]\s+\w")


def _has_inline_list_start(text: str) -> bool:
    """Return True if text contains an inline list start (colon followed by bullet/number)."""
    return bool(_INLINE_BULLET_RE.search(text) or _INLINE_NUMBER_RE.search(text))


# ---------------------------------------------------------------------------
# Incomplete list detection predicates
# ---------------------------------------------------------------------------


def _count_list_items(text: str) -> tuple[int, int]:
    """Count bullet and numbered list items in text.

    Returns (bullet_count, numbered_count).
    """
    bullet_matches = len(re.findall(r"[•\-\*\u2022]\s+\w", text))
    numbered_matches = len(re.findall(r"\d+[.)]\s+\w", text))
    return bullet_matches, numbered_matches


def _ends_with_list_intro_colon(lines: list[str]) -> bool:
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
    bullets, numbers = _count_list_items(all_text)
    return bullets == 0 and numbers == 0


def _has_single_inline_bullet(lines: list[str]) -> bool:
    """Check if text has a colon followed by single inline bullet.

    Example: "List intro: • single item" - incomplete because only one item.
    """
    if not lines:
        return False
    last_line = lines[-1].rstrip()
    if not _has_inline_list_start(last_line):
        return False
    # Check if there's only one list item total
    all_text = "\n".join(lines)
    bullets, numbers = _count_list_items(all_text)
    return (bullets + numbers) == 1


def _has_unterminated_bullet_item(lines: list[str]) -> bool:
    """Check if text has a single bullet item without sentence terminator.

    Example: "Intro:\n• First item" where "First item" lacks a period.
    """
    if len(lines) < 2:
        return False

    # Count list items at line start
    bullet_count = sum(1 for ln in lines if _is_list_line(ln.strip()))
    if bullet_count != 1:
        return False

    # Find the bullet line
    first_bullet_idx = next((i for i, ln in enumerate(lines) if _is_list_line(ln.strip())), None)
    if first_bullet_idx is None:
        return False

    # Check if there's intro text ending with colon before the bullet
    if first_bullet_idx > 0:
        pre_bullet = "\n".join(lines[:first_bullet_idx])
        if pre_bullet.rstrip().endswith(":"):
            return True

    # Check if single bullet item lacks sentence terminator
    last_line = lines[-1].strip()
    return _is_list_line(last_line) and not re.search(r"[.!?][\"')\]]*$", last_line)


def _has_incomplete_list(text: str) -> bool:
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
        _ends_with_list_intro_colon(lines)
        or _has_single_inline_bullet(lines)
        or _has_unterminated_bullet_item(lines)
    )


def _coherent(text: str, min_chars: int = 40) -> bool:
    """Return True if text appears to be a semantically complete unit.

    A text is coherent if it:
    1. Has sufficient length (min_chars)
    2. Starts with a capital letter/digit (proper sentence start)
    3. Ends with sentence-ending punctuation
    4. Does NOT appear to have an incomplete list
    """
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    if re.match(r"^[\"'(]*[A-Z0-9]", stripped) is None:
        return False
    if re.search(r"[.!?][\"')\]]*$", stripped) is None:
        return False
    # A text with an incomplete list is not coherent
    return not _has_incomplete_list(stripped)


def _merge_incomplete_lists(rows: list[Row]) -> list[Row]:
    """Merge rows with incomplete lists with their continuations.

    This handles cases like:
    - Row N: "The challenge is X." (13 words, coherent)
    - Row N+1: "Here is a guide: • Item" (7 words, incomplete list)
    - Row N+2: "Full item explanation..." (continuation)

    These should all be merged because:
    1. The short intro belongs with the list introduction
    2. The list intro is incomplete without its item content

    If forward merge exceeds size limits, try backward merge instead.
    """
    if len(rows) < 2:
        return rows

    min_words = _min_row_words()
    max_chars = _max_merge_chars()
    result: list[Row] = []
    i = 0

    while i < len(rows):
        row = rows[i]
        text = row.get("text", "")
        words = _word_count(text)

        # Check if this row is short enough to consider merging
        # OR if it has an incomplete list that needs continuation
        needs_merge = words < min_words or _has_incomplete_list(text)

        if needs_merge and i + 1 < len(rows):
            next_row = rows[i + 1]
            next_text = next_row.get("text", "")
            merged_chars = len(text) + len(next_text) + 2

            # Decide if we should merge based on context
            should_merge = False
            merge_reason = ""

            # Case 1: Current row is short and next has incomplete list
            if words < min_words and _has_incomplete_list(next_text) or _has_incomplete_list(text):
                should_merge = True
                merge_reason = (
                    "incomplete_list" if _has_incomplete_list(text) else "short+incomplete_next"
                )

            if should_merge:
                if merged_chars <= max_chars:
                    # Forward merge fits - do it
                    _log.debug(
                        "merge_incomplete_lists: forward merge (%s), %d+%d chars",
                        merge_reason,
                        len(text),
                        len(next_text),
                    )
                    merged_text = f"{text.rstrip()}\n\n{next_text}".strip()
                    # Replace current row with merged content but don't advance yet
                    # We might need to merge more if still incomplete
                    rows = [*rows[:i], {**next_row, "text": merged_text}, *rows[i + 2 :]]
                    continue  # Re-check merged row
                elif result and _has_incomplete_list(text):
                    # Forward merge too large - try backward merge
                    prev = result[-1]
                    prev_text = prev.get("text", "")
                    backward_chars = len(prev_text) + len(text) + 2
                    if backward_chars <= max_chars:
                        _log.debug(
                            "merge_incomplete: backward (fwd too large), %d+%d chars",
                            len(prev_text),
                            len(text),
                        )
                        merged_text = f"{prev_text.rstrip()}\n\n{text}".strip()
                        result[-1] = {**prev, "text": merged_text}
                        i += 1
                        continue

        result.append(row)
        i += 1

    return result


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle and needle in haystack)


def _contains_caption_line(text: str) -> bool:
    lines = tuple(line.strip().lower() for line in text.splitlines())
    return any(line.startswith(prefix) for prefix in _CAPTION_PREFIXES for line in lines)


def _caption_overlap(prev: str, curr: str, prefix: str) -> bool:
    snippet = prefix.strip()
    if not snippet:
        return False
    first_line = _first_non_empty_line(curr).strip()
    if not first_line or not first_line.lower().startswith(snippet.lower()):
        return False
    return _contains_caption_line(prev)


def _overlap_len(prev_lower: str, curr_lower: str) -> int:
    length = min(len(prev_lower), len(curr_lower))
    return next(
        (i for i in range(length, 0, -1) if prev_lower.endswith(curr_lower[:i])),
        0,
    )


_SENTENCE_TERMINATORS = ".?!"
_CLOSING_PUNCTUATION = "\"')]}"


def _has_sentence_ending(text: str) -> bool:
    stripped = text.rstrip()
    trimmed = stripped.rstrip(_CLOSING_PUNCTUATION)
    return bool(trimmed) and trimmed[-1] in _SENTENCE_TERMINATORS


def _prefix_contained_len(haystack: str, needle: str) -> int:
    length = len(needle)

    def _match(index: int) -> bool:
        segment = needle[:index]
        if not segment.strip():
            return False
        if index < length and not needle[index].isspace():
            return False
        if not _has_sentence_ending(segment):
            return False
        position = haystack.find(segment)
        if position == -1:
            return False
        preceding = haystack[position - 1] if position else ""
        return not preceding.isalnum()

    return next((idx for idx in range(length, 0, -1) if _match(idx)), 0)


def _looks_like_caption_label(text: str) -> bool:
    normalized = text.strip().lower()
    return bool(normalized and _CAPTION_LABEL_RE.match(normalized))


def _trim_overlap(prev: str, curr: str) -> str:
    """Remove duplicated prefix from ``curr`` that already exists in ``prev``."""

    prev_lower, curr_lower = prev.lower(), curr.lower()
    if _contains(prev_lower, curr_lower):
        return ""
    overlap = _overlap_len(prev_lower, curr_lower)
    if overlap and overlap < len(curr) * 0.9:
        prefix = curr[:overlap]
        if _caption_overlap(prev, curr, prefix):
            return curr
        prev_index = len(prev) - overlap
        prev_char = prev[prev_index - 1] if prev_index > 0 else ""
        next_non_space = next((ch for ch in curr[overlap:] if not ch.isspace()), "")
        stripped_prefix = prefix.strip()
        words = re.findall(r"\b\w+\b", stripped_prefix)
        single_title = len(words) == 1 and words[0][0].isupper() and words[0][1:].islower()
        if _looks_like_caption_label(stripped_prefix):
            return curr
        if prev_char.isalnum():
            return curr
        if single_title and (next_non_space.islower() or next_non_space.isdigit()):
            return curr
        return curr[overlap:].lstrip()
    prefix = curr_lower.split("\n\n", 1)[0]
    return curr[len(prefix) :].lstrip() if _contains(prev_lower, prefix) else curr


def _starts_mid_sentence(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and re.match(r"^[\"'(]*[A-Z0-9]", stripped) is None


_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*")


def _steal_sentence_prefix(prev: str, fragment: str, limit: int | None) -> tuple[str, str] | None:
    """Move the leading sentence from ``fragment`` onto ``prev`` when possible."""

    stripped = fragment.lstrip()
    if not stripped:
        return None

    offset = len(fragment) - len(stripped)
    for match in _SENTENCE_END_RE.finditer(stripped):
        end = match.end()
        if end < len(stripped) and not stripped[end].isspace():
            continue
        prefix = fragment[: offset + end]
        remainder = fragment[offset + end :]
        candidate = f"{prev.rstrip()} {prefix.strip()}".strip()
        if limit is not None and len(candidate) > limit:
            return None
        return candidate, remainder.lstrip()

    candidate = f"{prev.rstrip()} {stripped}".strip()
    if limit is not None and len(candidate) > limit:
        return None
    return candidate, ""


def _merge_text(
    prev: str,
    curr: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> str:
    last = _last_non_empty_line(prev)
    first = _first_non_empty_line(curr)
    cond = _is_list_line(last, strategy=strategy) and _is_list_line(
        first,
        strategy=strategy,
    )
    sep = "\n" if cond else "\n\n"
    return f"{prev.rstrip()}{sep}{curr}".strip()


def _merge_sentence_pieces(
    pieces: Iterable[str],
    limit: int | None = None,
) -> list[str]:
    def step(acc: list[str], piece: str) -> list[str]:
        if acc and _starts_mid_sentence(piece):
            merged_prev = acc[-1]
            remainder = piece
            while remainder:
                result = _steal_sentence_prefix(merged_prev, remainder, limit)
                if result is None:
                    break
                merged_prev, remainder = result
                if remainder:
                    remainder = remainder.lstrip()
                else:
                    return [*acc[:-1], merged_prev]
                if not _starts_mid_sentence(remainder):
                    break
            if merged_prev is not acc[-1]:
                acc = [*acc[:-1], merged_prev]
                piece = remainder
            elif limit is None or len(acc[-1]) + 1 + len(piece) <= limit:
                merged = f"{acc[-1].rstrip()} {piece}".strip()
                return [*acc[:-1], merged]
        return [*acc, piece]

    return reduce(step, pieces, [])


def _merge_if_fragment(
    acc: list[dict[str, Any]],
    acc_text: str,
    acc_norm: str,
    item: dict[str, Any],
    text: str,
    text_norm: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Merge ``text`` into ``acc`` if it begins mid-sentence."""

    if _starts_mid_sentence(text) and acc:
        prev = acc[-1]
        merged_text = f"{prev['text'].rstrip()} {text}".strip()
        merged_item = {**prev, "text": merged_text}
        merged_acc = f"{acc_text.rstrip()} {text}".strip()
        return (
            [*acc[:-1], merged_item],
            merged_acc,
            acc_norm + text_norm,
        )
    new_text = _merge_text(acc_text, text, strategy=strategy) if acc_text else text
    return (
        [*acc, {**item, "text": text}],
        new_text,
        acc_norm + text_norm,
    )


def _should_merge(prev_text: str, curr_text: str, min_words: int) -> bool:
    prev_words = _word_count(prev_text)
    curr_words = _word_count(curr_text)
    prev_coherent = _coherent(prev_text)
    curr_coherent = _coherent(curr_text)
    return any(
        (
            prev_words < min_words,
            (curr_words < min_words and not curr_coherent),
            not prev_coherent,
            _starts_mid_sentence(curr_text),
        )
    )


def _merge_items(
    acc: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> list[dict[str, Any]]:
    text = item["text"]
    if acc:
        prev = acc[-1]
        text = _trim_overlap(prev["text"], text)
        if _should_merge(prev["text"], text, _min_words()):
            merged = _merge_text(prev["text"], text, strategy=strategy)
            return [*acc[:-1], {**prev, "text": merged}]
    return [*acc, {**item, "text": text}]


def _coalesce(
    items: Iterable[dict[str, Any]],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> list[dict[str, Any]]:
    """Normalize item boundaries, trimming overlap and merging fragments."""

    cleaned = [{**i, "text": (i.get("text") or "").strip()} for i in items]
    cleaned = [i for i in cleaned if i["text"]]
    merged = reduce(
        lambda acc, item: _merge_items(acc, item, strategy=strategy),
        cleaned,
        cast(list[dict[str, Any]], []),
    )
    return merged


def _very_short_threshold() -> int:
    """Return threshold below which items should always be merged forward."""
    return _config().get_very_short_threshold()


def _starts_with_orphan_bullet(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    """Return True if text starts with a single bullet item (orphaned list fragment).

    An orphan bullet is when text begins with a single bullet point that appears
    to be a fragment of a list from a previous chunk. Complete single-item lists
    are NOT considered orphans, even if they lack terminal punctuation.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    first_line = lines[0].strip()
    heuristics = _resolve_bullet_strategy(strategy)

    # Check if first line is a bullet
    is_bullet = heuristics.starts_with_bullet(first_line)
    is_number = heuristics.starts_with_number(first_line)
    if not (is_bullet or is_number):
        return False

    # If there's only one line, check if it looks like a complete item
    if len(lines) == 1:
        # A coherent item or numbered item with enough words is NOT orphaned
        if _coherent(first_line):
            return False
        # Numbered items with sufficient words are likely complete even without punctuation
        # e.g., "2. Second item continues with sufficient words but lacks punctuation"
        return not (is_number and _word_count(first_line) >= 6)

    # If second line is NOT a bullet, first bullet is orphaned
    # (unless the first line looks complete)
    second_line = lines[1].strip()
    is_list_line = heuristics.starts_with_bullet(second_line) or heuristics.starts_with_number(
        second_line
    )
    if is_list_line:
        return False

    # First line is a bullet, but following content is not a list
    # Check if the first line alone looks complete
    if _coherent(first_line):
        return False
    return not (is_number and _word_count(first_line) >= 6)


def _max_merge_chars() -> int:
    """Return maximum character count for merged chunks."""
    return _config().max_merge_chars


def _merge_very_short_forward(
    items: list[dict[str, Any]],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> list[dict[str, Any]]:
    """Merge items below the very-short threshold into the following item.

    This handles cases like orphaned headings or single-sentence fragments
    that should logically belong with the next semantic unit. Also prevents
    chunks from starting with a single orphaned bullet item.

    Coherent items (proper sentence start and ending) are preserved even if
    short, as they represent complete semantic units.

    Merging stops when the result would exceed _max_merge_chars() to prevent
    creating overly large multi-topic chunks.
    """
    if not items:
        return items

    threshold = _very_short_threshold()
    max_merge = _max_merge_chars()
    # Minimum words for a coherent block to stand alone
    coherent_min = max(15, threshold // 2)
    result: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    pending_reason: str = ""

    for item in items:
        text = item.get("text", "")
        words = _word_count(text)
        chars = len(text)

        if pending is not None:
            pending_text = pending.get("text", "")
            merged_chars = len(pending_text) + len(text) + 2  # +2 for separator

            # Only merge if result won't be too large
            if merged_chars <= max_merge:
                _log.debug(
                    "merge_very_short: forward merge (%s), %d words + %d words, %d chars",
                    pending_reason,
                    _word_count(pending_text),
                    words,
                    merged_chars,
                )
                merged_text = _merge_text(pending_text, text, strategy=strategy)
                item = {**pending, "text": merged_text}
                pending = None
                pending_reason = ""
                # Recalculate after merge
                text = item.get("text", "")
                words = _word_count(text)
                chars = len(text)
            else:
                # Pending is too short but merging would exceed limit
                _log.debug(
                    "merge_very_short: skip merge (size limit), %d + %d > %d chars",
                    len(pending_text),
                    len(text),
                    max_merge,
                )
                # Keep pending as-is and continue
                result.append(pending)
                pending = None
                pending_reason = ""

        # Check if this item should be held for forward merge:
        # 1. Too short (< threshold words), unless coherent with enough words
        # 2. Starts with an orphaned bullet item
        is_short = words < threshold
        is_coherent_block = _coherent(text) and words >= coherent_min
        has_orphan_bullet = _starts_with_orphan_bullet(text, strategy=strategy)

        # Don't hold items that are already large enough
        is_large_enough = chars >= max_merge * 0.5  # At least half the max size

        # Coherent blocks with sufficient words are preserved
        if is_short and not is_coherent_block and not is_large_enough:
            pending = item
            pending_reason = "short"
        elif has_orphan_bullet and not is_large_enough:
            # Orphaned bullet - merge forward for list coherence
            pending = item
            pending_reason = "orphan_bullet"
        else:
            result.append(item)

    # Handle trailing short item - merge backward if possible
    if pending is not None:
        if result:
            prev = result[-1]
            prev_text = prev.get("text", "")
            pending_text = pending.get("text", "")
            merged_chars = len(prev_text) + len(pending_text) + 2

            if merged_chars <= max_merge:
                _log.debug(
                    "merge_very_short: backward merge trailing (%s), %d + %d chars",
                    pending_reason,
                    len(prev_text),
                    len(pending_text),
                )
                merged_text = _merge_text(prev_text, pending_text, strategy=strategy)
                result[-1] = {**prev, "text": merged_text}
            else:
                # Can't merge backward, keep as separate chunk
                result.append(pending)
        else:
            result.append(pending)

    return result


def _strip_superscripts(item: dict[str, Any]) -> dict[str, Any]:
    spans = tuple(item.get("_footnote_spans") or ())
    text = item.get("text", "")
    limit = len(text)
    normalized = tuple(
        (max(0, min(limit, start)), max(0, min(limit, end)))
        for start, end in sorted(spans, key=lambda pair: pair[0])
        if start < end
    )
    trimmed = {k: v for k, v in item.items() if k != "_footnote_spans"}
    if not text or not normalized:
        return {**trimmed, "text": text}
    starts = tuple(start for start, _ in normalized)
    ends = tuple(end for _, end in normalized)
    gaps = (0, *ends)
    pieces = (text[prev_end:curr_start] for prev_end, curr_start in zip(gaps, starts, strict=False))
    tail = text[ends[-1] :]
    stripped = "".join(pieces) + tail
    return {**trimmed, "text": stripped}


def _sanitize_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_strip_superscripts(item) for item in items]


def _dedupe(
    items: Iterable[dict[str, Any]],
    *,
    log: list[str] | None = None,
    strategy: BulletHeuristicStrategy | None = None,
) -> list[dict[str, Any]]:
    """Remove items whose text already appears in prior items.

    When ``log`` is provided, dropped duplicate snippets are appended to it for
    debug inspection. The function itself remains pure; callers decide whether
    to record diagnostics.
    """

    def step(
        state: tuple[list[dict[str, Any]], str, str], item: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, str]:
        acc, acc_text, acc_norm = state
        text = item["text"]
        text_norm = _normalize(text)
        if _contains(acc_norm, text_norm):
            if log is not None:
                log.append(text)
            return state
        overlap = _overlap_len(acc_norm, text_norm) or _prefix_contained_len(
            acc_norm,
            text_norm,
        )
        if overlap:
            prefix = text[:overlap]
            if _looks_like_caption_label(prefix):
                overlap = 0
            if overlap:
                if log is not None:
                    log.append(prefix)
                text = text[overlap:].lstrip()
                if not text:
                    return state
                text_norm = _normalize(text)
        return _merge_if_fragment(
            acc,
            acc_text,
            acc_norm,
            item,
            text,
            text_norm,
            strategy=strategy,
        )

    initial: tuple[list[dict[str, Any]], str, str] = ([], "", "")
    return reduce(step, items, initial)[0]


def _flag_potential_duplicates(
    items: Iterable[dict[str, Any]], *, min_words: int = 10
) -> list[str]:
    """Return sentences appearing more than once after dedupe."""

    seen: set[str] = set()
    flagged: list[str] = []
    for sent in (s for item in items for s in re.split(r"(?<=[.!?])\s+", item["text"])):
        words = re.findall(r"\w+", sent.lower())
        if len(words) < min_words:
            continue
        key = " ".join(words)
        if key in seen:
            flagged.append(sent.strip())
        else:
            seen.add(key)
    return flagged


def _rows_from_item(
    item: dict[str, Any],
    *,
    strategy: BulletHeuristicStrategy | None = None,
    preserve: bool = False,
) -> list[Row]:
    meta_key = _metadata_key()
    max_chars = _max_chars()
    target_chars = _target_chunk_chars()
    meta: dict[str, Any] = item.get("meta") or {}
    chunk_id = meta.get("chunk_id")
    if chunk_id:
        meta = {**meta, "chunk_id": _compat_chunk_id(chunk_id)}
    base_meta = {meta_key: meta} if meta else {}
    overhead = len(json.dumps({"text": "", **base_meta}, ensure_ascii=False)) - 2

    # If preserving chunks (e.g. custom chunk_size set), suppress the opinionated
    # RAG target limit and only enforce the hard safety limit (max_chars).
    if preserve:
        target_chars = max_chars

    # Use target size for splitting, but ensure we don't exceed hard limit
    split_limit = min(target_chars, max_chars - overhead)
    avail = max(split_limit, 0)
    if avail <= 0:
        return []

    heuristics = _resolve_bullet_strategy(strategy)

    def predicate(line: str) -> bool:
        return _is_list_line(line, strategy=heuristics)

    pieces = [
        piece
        for piece in _merge_sentence_pieces(
            _split(item.get("text", ""), avail, is_list_line=predicate),
            avail,
        )
        if piece.strip()
    ]

    def build(idx_piece: tuple[int, str]) -> Row:
        idx, piece = idx_piece
        piece = piece.lstrip()
        if meta and len(pieces) > 1:
            meta_part = {meta_key: {**meta, "chunk_part": idx}}
        else:
            meta_part = base_meta
        row = {"text": piece, **meta_part}

        while len(json.dumps(row, ensure_ascii=False)) > max_chars:
            current_len = len(json.dumps(row, ensure_ascii=False))
            allowed = avail - (current_len - max_chars)
            allowed = min(allowed, len(piece) - 1)
            if allowed <= 0:
                return {"text": "", **meta_part}
            piece = _truncate_chunk(piece[:allowed], allowed)
            row = {"text": piece, **meta_part}
        return row

    return [row for row in (build(x) for x in enumerate(pieces)) if row["text"].strip()]


def _enrich_rows_with_context(rows: list[Row]) -> list[Row]:
    enriched: list[Row] = []
    prev_text: str | None = None
    meta_key = _metadata_key()
    for row in rows:
        text = row.get("text", "")
        lead = text.lstrip()
        meta = row.get(meta_key, {})
        chunk_part = int(meta.get("chunk_part", 0))
        if (
            prev_text
            and lead
            and _starts_with_continuation(lead)
            and (chunk_part > 0 or not prev_text.rstrip().endswith(":"))
        ):
            context = _last_sentence(prev_text)
            if context and not lead.startswith(context):
                merged = f"{context} {text}".strip()
                if len(merged) <= _max_chars():
                    enriched.append({**row, "text": merged})
                    prev_text = merged
                    continue
        enriched.append(row)
        prev_text = text
    return enriched


def _explicit_small_chunks(meta: dict[str, Any] | None) -> bool:
    """Return True if user explicitly requested small chunks via chunk_size."""
    opts = ((meta or {}).get("options") or {}).get("split_semantic", {})
    chunk_size = opts.get("chunk_size")
    # If chunk_size is explicitly set to a small value, user wants small chunks
    return chunk_size is not None and chunk_size < 50


def _preserve_chunks(meta: dict[str, Any] | None) -> bool:
    opts = ((meta or {}).get("options") or {}).get("split_semantic", {})
    chunk_size = opts.get("chunk_size")
    overlap = opts.get("overlap")
    return chunk_size is not None or (isinstance(overlap, int | float) and overlap > 0)


def _min_row_words() -> int:
    """Minimum word count for a standalone row."""
    return _config().get_min_row_words()


def _critical_short_threshold() -> int:
    """Minimum word count below which rows MUST be merged."""
    return _config().critical_short_words


def _merge_short_rows(rows: list[Row]) -> list[Row]:
    """Merge rows below the minimum word threshold with neighbors.

    This is the final pass that catches any short fragments that:
    - Came from items that couldn't be merged due to size limits
    - Were produced by _split/_rows_from_item splitting
    - Are single items with no neighbors

    Short rows are merged with the following row if possible,
    otherwise with the preceding row.

    Rows below the critical threshold (5 words) are ALWAYS merged,
    even if it exceeds the soft size limit.
    """
    if not rows:
        return rows

    # First pass: merge rows with incomplete lists with their continuations
    # This ensures list introductions stay with their items
    rows = _merge_incomplete_lists(rows)

    min_words = _min_row_words()
    critical_threshold = _critical_short_threshold()
    max_chars = _max_merge_chars()
    result: list[Row] = []
    pending: Row | None = None

    for row in rows:
        text = row.get("text", "")
        words = _word_count(text)
        chars = len(text)

        if pending is not None:
            pending_text = pending.get("text", "")
            pending_words = _word_count(pending_text)
            merged_chars = len(pending_text) + chars + 2

            # ALWAYS merge if pending is critically short (< 5 words)
            # Otherwise, only merge if it fits within the soft limit
            is_critical = pending_words < critical_threshold
            should_merge = is_critical or merged_chars <= max_chars

            if should_merge:
                merge_reason = "critical_short" if is_critical else "short"
                _log.debug(
                    "merge_short_rows: forward merge (%s), %d words + %d words, %d chars",
                    merge_reason,
                    pending_words,
                    words,
                    merged_chars,
                )
                merged_text = f"{pending_text.rstrip()}\n\n{text}".strip()
                row = {**row, "text": merged_text}
                pending = None
                # Recalculate
                text = row.get("text", "")
                words = _word_count(text)
                chars = len(text)
            else:
                # Can't merge, keep pending as-is
                _log.debug(
                    "merge_short_rows: skip merge (size limit), %d + %d > %d chars",
                    len(pending_text),
                    chars,
                    max_chars,
                )
                result.append(pending)
                pending = None

        # Check if row is too short to stand alone
        is_short = words < min_words
        # Coherent rows (proper sentence) can stand alone even if short
        is_coherent = _coherent(text) and words >= 8

        if is_short and not is_coherent:
            pending = row
        else:
            result.append(row)

    # Handle trailing short row
    if pending is not None:
        if result:
            prev = result[-1]
            prev_text = prev.get("text", "")
            pending_text = pending.get("text", "")
            pending_words = _word_count(pending_text)
            merged_chars = len(prev_text) + len(pending_text) + 2

            # ALWAYS merge critically short trailing items
            is_critical = pending_words < critical_threshold
            should_merge = is_critical or merged_chars <= max_chars

            if should_merge:
                merge_reason = "critical_short" if is_critical else "short"
                _log.debug(
                    "merge_short_rows: backward merge trailing (%s), %d + %d chars",
                    merge_reason,
                    len(prev_text),
                    len(pending_text),
                )
                merged_text = f"{prev_text.rstrip()}\n\n{pending_text}".strip()
                result[-1] = {**prev, "text": merged_text}
            else:
                # Can't merge, keep as separate row
                _log.debug(
                    "merge_short_rows: keep trailing separate (size limit), %d + %d > %d",
                    len(prev_text),
                    len(pending_text),
                    max_chars,
                )
                result.append(pending)
        else:
            result.append(pending)

    return result


def _rows(
    doc: Doc,
    *,
    preserve: bool = False,
    explicit_small: bool = False,
    strategy: BulletHeuristicStrategy | None = None,
) -> list[Row]:
    cfg = _config()
    debug_log: list[str] | None = [] if (not preserve and cfg.dedup_debug) else None
    items = _sanitize_items(doc.get("items", []))
    heuristics = _resolve_bullet_strategy(strategy)

    # Merge very short items forward unless user explicitly requested small chunks.
    # This handles orphaned headings and single-sentence fragments while respecting
    # explicit --chunk-size configurations.
    if not explicit_small:
        items = _merge_very_short_forward(list(items), strategy=heuristics)

    processed = (
        items
        if preserve
        else _dedupe(
            _coalesce(items, strategy=heuristics),
            log=debug_log,
            strategy=heuristics,
        )
    )
    if debug_log is not None:
        _log.warning("dedupe dropped %d duplicates", len(debug_log))
        for dup in debug_log:
            _log.warning("dedupe dropped: %s", dup[:80])
        for dup in _flag_potential_duplicates(processed):
            _log.warning("possible duplicate retained: %s", dup[:80])
    rows = [
        r for i in processed for r in _rows_from_item(i, strategy=heuristics, preserve=preserve)
    ]
    # Final pass: merge any rows that are still too short after splitting
    if not explicit_small:
        rows = _merge_short_rows(rows)
    if _max_chars() < 8000:
        return rows
    return _enrich_rows_with_context(rows)


def _update_meta(meta: dict[str, Any] | None, count: int) -> dict[str, Any]:
    base = dict(meta or {})
    base.setdefault("metrics", {}).setdefault("emit_jsonl", {})["rows"] = count
    return base


class _EmitJsonlPass:
    name = "emit_jsonl"
    input_type = dict
    output_type = list
    bullet_strategy: BulletHeuristicStrategy | None = default_bullet_strategy()

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload if isinstance(a.payload, dict) else {}
        preserve = _preserve_chunks(a.meta)
        explicit_small = _explicit_small_chunks(a.meta)
        strategy = _resolve_bullet_strategy(self.bullet_strategy)
        rows = (
            _rows(doc, preserve=preserve, explicit_small=explicit_small, strategy=strategy)
            if doc.get("type") == "chunks"
            else []
        )
        meta = _update_meta(a.meta, len(rows))
        return Artifact(payload=rows, meta=meta)


emit_jsonl = register(_EmitJsonlPass())
