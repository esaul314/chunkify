"""Sentence fusion helpers extracted from :mod:`split_semantic`."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import reduce

SOFT_LIMIT = 8_000

_WORD_START = re.compile(r"[\w']+")
_SENTENCE_BOUNDARY = re.compile(r"[.?!][\"')\]]*\s+")
_ENDS_SENTENCE = re.compile(r"[.?!][\"')\]]*\s*$")
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
    derived_min = min_chunk_size if min_chunk_size is not None else chunk_size // 10
    if derived_min > chunk_size // 2:
        return None
    return max(int(chunk_size) - max(overlap, 0), 0)


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
    limit = _compute_limit(chunk_size, allowed_overlap, min_chunk_size)

    def _should_merge(previous: str, current: str, prev_words: list[str]) -> bool:
        if not previous:
            return False
        lead = current.lstrip()
        if not lead:
            return False
        continuation_lead = _is_continuation_lead(lead)
        if _ENDS_SENTENCE.search(previous.rstrip()) and not continuation_lead:
            return False
        current_words = current.split()
        first_word = current_words[0] if current_words else ""
        if prev_words and prev_words[-1] == first_word:
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
    ) -> tuple[str, tuple[str, ...]]:
        if not words:
            return "", words
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
        if not words:
            return acc

        if not acc:
            return _append(acc, chunk, words)

        prev_text, prev_words = acc[-1]
        trimmed_text, trimmed_words = _dedupe_overlap(prev_words, words)
        if not trimmed_words:
            return _append(acc, chunk, words)

        if not _should_merge(prev_text, trimmed_text, list(prev_words)):
            return _append(acc, chunk, words)

        if limit is not None and len(prev_words) + len(trimmed_words) > limit:
            adjusted = _rebalance_overflow(prev_text, prev_words, trimmed_text, limit)
            if adjusted is not None:
                prev_text, prev_words, trimmed_text = adjusted
                trimmed_words = tuple(trimmed_text.split())
                if trimmed_words:
                    acc = [*acc[:-1], (prev_text, prev_words)]
            if not trimmed_words:
                return _append(acc, chunk, words)
            return _append(acc, trimmed_text, trimmed_words)

        merged_text = f"{prev_text} {trimmed_text}".strip()
        merged_words = (*prev_words, *trimmed_words)
        return [*acc[:-1], (merged_text, merged_words)]

    merged: list[tuple[str, tuple[str, ...]]] = reduce(
        _merge,
        chunks,
        [],
    )

    return _stitch_continuation_heads([text for text, _ in merged], limit)


def _stitch_continuation_heads(chunks: list[str], limit: int | None) -> list[str]:
    def _consume(acc: list[str], chunk: str) -> list[str]:
        if not chunk:
            return acc
        if not acc:
            return [*acc, chunk]

        prev = acc[-1]
        prev_words = tuple(prev.split())
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
    limit: int | None,
) -> tuple[str, tuple[str, ...], str] | None:
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
