"""Sentence fusion helpers extracted from :mod:`split_semantic`."""

from __future__ import annotations

import re
from collections.abc import Iterable
from math import ceil
from functools import reduce
from numbers import Real
from typing import NamedTuple

SOFT_LIMIT = 8_000

_AVERAGE_CHARS_PER_TOKEN = 5.0

_WORD_START = re.compile(r"[\w']+")
_SENTENCE_BOUNDARY = re.compile(r"[.?!][\"')\]]*\s+")
_ENDS_SENTENCE = re.compile(r"[.?!][\"')\]]*\s*$")
_CAPTION_PREFIXES = ("Figure", "Table", "Exhibit")
_CAPTION_RE = re.compile(rf"^(?:{'|'.join(_CAPTION_PREFIXES)})\s+\d[\w.-]*\.\s+", re.IGNORECASE)
_CAPTION_ANYWHERE_RE = re.compile(
    rf"(?:{'|'.join(_CAPTION_PREFIXES)})\s+\d[\w.-]*\.\s+",
    re.IGNORECASE,
)
_CAPTION_BREAK_STARTERS = (
    "the",
    "this",
    "that",
    "these",
    "those",
    "we",
    "it",
    "they",
    "there",
    "their",
    "our",
    "his",
    "her",
    "its",
    "however",
    "but",
    "so",
    "yet",
    "although",
    "though",
    "once",
    "while",
    "when",
    "because",
)
_LEADING_CONTINUATIONS = frozenset(
    token.lower()
    for token in (
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


def _leading_token(text: str) -> str:
    match = _WORD_START.match(text)
    return match.group(0).lower() if match else ""


def _is_continuation_lead(text: str) -> bool:
    return _leading_token(text) in _LEADING_CONTINUATIONS


def _compute_limit(chunk_size: int | None, overlap: int, min_chunk_size: int | None) -> int | None:
    if chunk_size is None or chunk_size <= 0:
        return None
    normalized = int(chunk_size)
    declared_min = (
        int(min_chunk_size)
        if min_chunk_size is not None
        else normalized // 10
    )
    effective_min = min(max(declared_min, 0), normalized)
    fallback = max(normalized - max(overlap, 0), 0)
    return max(fallback, effective_min)


class _MergeBudget(NamedTuple):
    """Effective budget for combining sentence fragments."""

    limit: int | None
    hard_limit: int | None
    word_total: int
    dense_total: int
    effective_total: int


def _token_load(
    words: Iterable[str],
    *,
    average_chars_per_token: float = _AVERAGE_CHARS_PER_TOKEN,
) -> tuple[tuple[str, ...], int, int]:
    tokens = tuple(words)
    char_total = sum(len(token) for token in tokens)
    density = average_chars_per_token if average_chars_per_token > 0 else _AVERAGE_CHARS_PER_TOKEN
    dense_total = int(ceil(char_total / density)) if char_total else 0
    return tokens, len(tokens), dense_total


def _effective_token_count(
    words: Iterable[str],
    *,
    average_chars_per_token: float = _AVERAGE_CHARS_PER_TOKEN,
) -> int:
    _, word_count, dense_total = _token_load(
        words, average_chars_per_token=average_chars_per_token
    )
    return max(word_count, dense_total)


def _derive_merge_budget(
    previous: Iterable[str],
    current: Iterable[str],
    *,
    chunk_size: int | None,
    overlap: int,
    min_chunk_size: int | None = None,
    average_chars_per_token: float = _AVERAGE_CHARS_PER_TOKEN,
) -> _MergeBudget:
    """Derive the merge ceiling using both word and character density data.

    ``chunk_size`` and ``overlap`` originate from CLI overrides.  The helper
    synthesises a deterministic ceiling so whitespace-free fragments can still
    exceed the configured budget and trigger a split.
    """

    allowed_overlap = max(overlap, 0)
    limit = _compute_limit(chunk_size, allowed_overlap, min_chunk_size)
    hard_limit = (
        int(chunk_size)
        if isinstance(chunk_size, Real) and chunk_size > 0
        else None
    )
    prev_words, prev_count, prev_dense = _token_load(
        previous, average_chars_per_token=average_chars_per_token
    )
    next_words, next_count, next_dense = _token_load(
        current, average_chars_per_token=average_chars_per_token
    )
    word_total = prev_count + next_count
    dense_total = prev_dense + next_dense
    effective_total = max(word_total, dense_total)
    return _MergeBudget(limit, hard_limit, word_total, dense_total, effective_total)


def _looks_like_caption(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped and _CAPTION_RE.match(stripped))


def _capitalize_lead(text: str) -> str:
    for idx, char in enumerate(text):
        if char.isalpha():
            return f"{text[:idx]}{char.upper()}{text[idx + 1:]}"
    return text


def _split_caption_body(text: str) -> tuple[str, ...]:
    stripped = text.strip()
    if not stripped:
        return (text,)
    match = _CAPTION_RE.match(stripped)
    if not match:
        return (stripped,)
    label = match.group(0).strip()
    body = stripped[match.end() :].strip()
    if not body:
        return (stripped,)
    words = body.split()
    candidates = (
        idx
        for idx, word in enumerate(words)
        if idx >= 4
        and (token := re.sub(r"[^A-Za-z]", "", word).lower())
        and token in _CAPTION_BREAK_STARTERS
        and word[0].islower()
        and len(words) - idx >= 4
    )
    for idx in candidates:
        caption_words = words[:idx]
        remainder_words = words[idx:]
        caption_text = " ".join(caption_words).strip()
        remainder_text = " ".join(remainder_words).strip()
        if not (caption_text and remainder_text):
            continue
        caption = f"{label} {caption_text}".strip()
        if caption and caption[-1] not in ".!?":
            caption = f"{caption}."
        remainder = _capitalize_lead(remainder_text)
        return (caption, remainder)
    return (stripped,)


def _split_caption_chunk(chunk: str) -> tuple[str, ...]:
    stripped = chunk.strip()
    if not stripped:
        return (chunk,)
    match = _CAPTION_ANYWHERE_RE.search(stripped)
    if not match:
        return (chunk,)
    leading = stripped[: match.start()].strip()
    trailing = stripped[match.start() :]
    parts: list[str] = []
    if leading:
        parts.append(leading)
    parts.extend(_split_caption_body(trailing))
    return tuple(part for part in parts if part)


def _expand_caption_chunks(chunks: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        part
        for chunk in chunks
        for part in _split_caption_chunk(chunk)
    )


def _last_sentence(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    segments = [seg.strip() for seg in _SENTENCE_BOUNDARY.split(stripped) if seg.strip()]
    return segments[-1] if segments else stripped


def _merge_sentence_fragments(
    chunks: Iterable[str],
    *,
    max_words: int = 80,
    chunk_size: int | None = None,
    overlap: int = 0,
    min_chunk_size: int | None = None,
) -> list[str]:
    allowed_overlap = max(overlap, 0)
    normalized_chunks = _expand_caption_chunks(chunks)
    has_sentence_boundary = any(
        _ENDS_SENTENCE.search(chunk.strip()) for chunk in normalized_chunks
    )
    base_limit = _compute_limit(chunk_size, allowed_overlap, min_chunk_size)
    hard_limit = (
        int(chunk_size)
        if isinstance(chunk_size, Real) and chunk_size > 0
        else None
    )

    def _target_limit(budget: _MergeBudget) -> int | None:
        return budget.limit if budget.limit is not None else budget.hard_limit

    def _should_merge(
        previous: str,
        current: str,
        prev_words: tuple[str, ...],
        current_words: tuple[str, ...],
        budget: _MergeBudget,
    ) -> bool:
        if not previous:
            return False
        lead = current.lstrip()
        if not lead:
            return False
        continuation_lead = _is_continuation_lead(lead)
        prev_ends_sentence = bool(_ENDS_SENTENCE.search(previous.rstrip()))
        if prev_ends_sentence and not continuation_lead:
            return False
        first_word = current_words[0] if current_words else ""
        if prev_words and prev_words[-1] == first_word:
            return False
        limit = _target_limit(budget)
        if limit is not None and budget.effective_total > limit:
            soft_limit = budget.limit is not None and budget.hard_limit is not None
            allowed_overflow = (
                (soft_limit and budget.effective_total <= budget.hard_limit)
                or (has_sentence_boundary and not prev_ends_sentence)
            )
            if not allowed_overflow:
                return False
        head = lead[0]
        continuation_chars = ",.;:)]\"'"
        if not (continuation_lead or head.islower() or head in continuation_chars):
            return False
        combined = len(previous) + 1 + len(current)
        if len(previous) >= SOFT_LIMIT or len(current) >= SOFT_LIMIT:
            return False
        return combined <= SOFT_LIMIT

    def _actual_overlap(
        prev_words: tuple[str, ...],
        current_words: tuple[str, ...],
    ) -> int:
        if not allowed_overlap or not prev_words or not current_words:
            return 0
        window = min(allowed_overlap, len(prev_words), len(current_words))
        return window if window and prev_words[-window:] == current_words[:window] else 0

    def _dedupe_overlap(
        prev_words: tuple[str, ...],
        words: tuple[str, ...],
        *,
        trim: bool = True,
    ) -> tuple[str, tuple[str, ...]]:
        if not words:
            return "", words
        if not trim:
            return " ".join(words), words
        overlap_words = _actual_overlap(prev_words, words)
        trimmed_words = words[overlap_words:] if overlap_words else words
        return " ".join(trimmed_words), trimmed_words

    def _append(
        acc: list[tuple[str, tuple[str, ...]]],
        text: str,
        words: tuple[str, ...],
    ) -> list[tuple[str, tuple[str, ...]]]:
        return [*acc, (text, words)]

    def _merge(
        acc: list[tuple[str, tuple[str, ...]]],
        chunk: str,
    ) -> list[tuple[str, tuple[str, ...]]]:
        words = tuple(chunk.split())
        is_caption = _looks_like_caption(chunk)
        if not words:
            return acc

        if not acc:
            return _append(acc, chunk, words)

        prev_text, prev_words = acc[-1]
        trimmed_text, trimmed_words = _dedupe_overlap(
            prev_words,
            words,
            trim=not is_caption,
        )
        if not trimmed_words:
            return _append(acc, chunk, words)

        budget = _derive_merge_budget(
            prev_words,
            trimmed_words,
            chunk_size=chunk_size,
            overlap=allowed_overlap,
            min_chunk_size=min_chunk_size,
        )
        prev_ends_sentence = bool(_ENDS_SENTENCE.search(prev_text.rstrip()))

        if not _should_merge(
            prev_text,
            trimmed_text,
            prev_words,
            trimmed_words,
            budget,
        ):
            if trimmed_words != words:
                return _append(acc, trimmed_text, trimmed_words)
            return _append(acc, chunk, words)

        target_limit = _target_limit(budget)
        exceeds_limit = target_limit is not None and budget.effective_total > target_limit
        if exceeds_limit:
            soft_limit = budget.limit is not None and budget.hard_limit is not None
            if soft_limit and budget.effective_total <= budget.hard_limit:
                exceeds_limit = False
        if exceeds_limit and has_sentence_boundary and not prev_ends_sentence:
            exceeds_limit = False
        if exceeds_limit:
            adjusted = (
                _rebalance_overflow(prev_text, prev_words, trimmed_text, budget)
                if target_limit is not None
                else None
            )
            if adjusted is not None:
                prev_text, prev_words, trimmed_text = adjusted
                trimmed_words = tuple(trimmed_text.split())
                if trimmed_words:
                    acc = [*acc[:-1], (prev_text, prev_words)]
                    budget = _derive_merge_budget(
                        prev_words,
                        trimmed_words,
                        chunk_size=chunk_size,
                        overlap=allowed_overlap,
                        min_chunk_size=min_chunk_size,
                    )
                    target_limit = _target_limit(budget)
                    exceeds_limit = target_limit is not None and budget.effective_total > target_limit
                    if exceeds_limit:
                        soft_limit = budget.limit is not None and budget.hard_limit is not None
                        if soft_limit and budget.effective_total <= budget.hard_limit:
                            exceeds_limit = False
                    prev_ends_sentence = bool(_ENDS_SENTENCE.search(prev_text.rstrip()))
                    if exceeds_limit and has_sentence_boundary and not prev_ends_sentence:
                        exceeds_limit = False
            if not trimmed_words:
                return acc
            if exceeds_limit:
                return _append(acc, trimmed_text, trimmed_words)

        merged_text = f"{prev_text} {trimmed_text}".strip()
        merged_words = (*prev_words, *trimmed_words)
        return [*acc[:-1], (merged_text, merged_words)]

    merged: list[tuple[str, tuple[str, ...]]] = reduce(
        _merge,
        normalized_chunks,
        [],
    )

    stitch_limit = hard_limit if hard_limit is not None else base_limit
    return _stitch_continuation_heads([text for text, _ in merged], stitch_limit)


def _stitch_continuation_heads(chunks: list[str], limit: int | None) -> list[str]:
    def _consume(acc: list[str], chunk: str) -> list[str]:
        if not chunk:
            return acc
        if not acc:
            return [*acc, chunk]

        prev = acc[-1]
        prev_words = tuple(prev.split())
        chunk_words = tuple(chunk.split())
        remaining = chunk
        changed = False

        while True:
            lead = remaining.lstrip()
            if not lead or not _is_continuation_lead(lead):
                break
            boundary = next(
                (
                    match.end()
                    for match in _SENTENCE_BOUNDARY.finditer(remaining)
                    if match.end() < len(remaining)
                ),
                None,
            )
            if boundary is None:
                break
            head = remaining[:boundary].strip()
            tail = remaining[boundary:].lstrip()
            if not head or not tail:
                break
            head_words = tuple(head.split())
            if limit is not None and len(prev_words) + len(head_words) > limit:
                break
            prev = f"{prev} {head}".strip()
            prev_words = tuple(prev.split())
            remaining = tail
            changed = True

        if not changed:
            if _is_continuation_lead(chunk.lstrip()):
                if limit is not None and len(prev_words) + len(chunk_words) > limit:
                    return [*acc, chunk]
                context = _last_sentence(prev)
                if context and not chunk.lstrip().startswith(context):
                    prefixed = f"{context} {chunk}".strip()
                    return [*acc, prefixed]
            return [*acc, chunk]
        if not remaining:
            return [*acc[:-1], prev]
        return [*acc[:-1], prev, remaining]

    return reduce(_consume, chunks, [])


def _rebalance_overflow(
    prev_text: str,
    prev_words: tuple[str, ...],
    next_text: str,
    budget: _MergeBudget,
) -> tuple[str, tuple[str, ...], str] | None:
    limit = budget.limit if budget.limit is not None else budget.hard_limit
    if limit is None:
        return None

    updated_prev = prev_text
    updated_prev_words = prev_words
    updated_next = next_text
    changed = False

    if len(updated_prev_words) > limit:
        boundaries = [
            match.end()
            for match in _SENTENCE_BOUNDARY.finditer(updated_prev)
            if match.end() < len(updated_prev)
        ]
        for pos in reversed(boundaries):
            head = updated_prev[:pos].rstrip()
            tail = updated_prev[pos:].lstrip()
            if not head or not tail:
                continue
            head_words = tuple(head.split())
            if len(head_words) > limit:
                continue
            merged = _merge_tail_with_next(tail, updated_next)
            merged_words = tuple(merged.split())
            if not merged_words:
                continue
            updated_prev = head
            updated_prev_words = head_words
            updated_next = merged
            changed = True
            break
        else:
            return None

    lead = updated_next.lstrip()
    if lead and _is_continuation_lead(lead):
        boundaries = [
            match.end()
            for match in _SENTENCE_BOUNDARY.finditer(updated_next)
            if match.end() < len(updated_next)
        ]
        if boundaries:
            pos = boundaries[0]
            head = updated_next[:pos].strip()
            tail = updated_next[pos:].lstrip()
            if head and tail:
                head_words = tuple(head.split())
                if len(updated_prev_words) + len(head_words) <= limit:
                    updated_prev = f"{updated_prev} {head}".strip()
                    updated_prev_words = tuple(updated_prev.split())
                    updated_next = tail
                    changed = True

    return (updated_prev, updated_prev_words, updated_next) if changed else None


def _merge_tail_with_next(tail: str, current: str) -> str:
    core = tail.rstrip()
    trailing = tail[len(core) :]
    if not core:
        return current.lstrip()
    current_core = current.lstrip()
    if not current_core:
        return core
    max_overlap = min(len(core), len(current_core))
    overlap = next(
        (size for size in range(max_overlap, 0, -1) if core.endswith(current_core[:size])),
        0,
    )
    remainder = current_core[overlap:].lstrip()
    if not remainder:
        return core
    glue = trailing if trailing else ("" if core.endswith((" ", "\n", "\t")) else " ")
    return f"{core}{glue}{remainder}"
