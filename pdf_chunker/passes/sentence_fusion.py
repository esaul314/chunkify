"""Sentence fusion helpers extracted from :mod:`split_semantic`."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from functools import reduce
from numbers import Real
from typing import NamedTuple

SOFT_LIMIT = 8_000

_AVERAGE_CHARS_PER_TOKEN = 5.0

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
    normalized = int(chunk_size)
    declared_min = int(min_chunk_size) if min_chunk_size is not None else normalized // 10
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


class _MergeDecision(NamedTuple):
    """Sentence merge outcome including overflow allowances."""

    allowed: bool
    allow_overflow: bool
    dense_fragments: bool


@dataclass(frozen=True)
class _BudgetView:
    """Summarise merge budget tolerance for the current fragment pair."""

    budget: _MergeBudget
    target_limit: int | None
    dense_fragments: bool
    sentence_completion_pending: bool

    @property
    def overflow_required(self) -> bool:
        limit = self.target_limit
        return limit is not None and self.budget.effective_total > limit

    @property
    def dense_overflow_safe(self) -> bool:
        hard = self.budget.hard_limit
        return (
            self.sentence_completion_pending
            and hard is not None
            and self.budget.word_total <= hard
        )

    @property
    def within_hard_limit(self) -> bool:
        hard = self.budget.hard_limit
        if hard is None:
            return True
        if self.budget.effective_total <= hard:
            return True
        return self.dense_overflow_safe

    def allow_overflow(self) -> bool:
        if not self.overflow_required:
            return False
        if not self.within_hard_limit:
            return False
        if self.dense_fragments and not (
            self.sentence_completion_pending
            or (
                self.budget.hard_limit is not None
                and self.budget.effective_total <= self.budget.hard_limit
            )
        ):
            return False
        return True

    def violates(self) -> bool:
        if not self.within_hard_limit:
            return True
        limit = self.budget.limit
        if limit is None or self.budget.effective_total <= limit:
            return False
        if self.allow_overflow():
            return False
        hard = self.budget.hard_limit
        if hard is not None and self.budget.effective_total <= hard:
            return False
        return not self.dense_fragments


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
    hard_limit = int(chunk_size) if isinstance(chunk_size, Real) and chunk_size > 0 else None
    density = average_chars_per_token if average_chars_per_token > 0 else _AVERAGE_CHARS_PER_TOKEN

    def _load(words: tuple[str, ...]) -> tuple[int, int]:
        word_count = len(words)
        char_total = sum(len(token) for token in words)
        dense_total = int(ceil(char_total / density)) if char_total else 0
        return word_count, dense_total

    prev_words = tuple(previous)
    next_words = tuple(current)
    word_loads = tuple(_load(words) for words in (prev_words, next_words))
    word_total = sum(count for count, _ in word_loads)
    dense_total = sum(dense for _, dense in word_loads)
    effective_total = max(word_total, dense_total)
    return _MergeBudget(limit, hard_limit, word_total, dense_total, effective_total)


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
    hard_limit = int(chunk_size) if isinstance(chunk_size, Real) and chunk_size > 0 else None
    target_limit = limit if limit is not None else hard_limit

    def _target_limit_from(budget: _MergeBudget) -> int | None:
        return budget.limit if budget.limit is not None else budget.hard_limit

    def _is_dense_fragment(
        prev_words: tuple[str, ...], current_words: tuple[str, ...]
    ) -> bool:
        return len(prev_words) <= 1 or len(current_words) <= 1

    def _violates_budget(
        budget: _MergeBudget,
        *,
        dense_fragments: bool,
    ) -> bool:
        hard_cap_exceeded = (
            budget.hard_limit is not None and budget.effective_total > budget.hard_limit
        )
        if hard_cap_exceeded:
            return True

        if budget.limit is None or budget.effective_total <= budget.limit:
            return False

        soft_cap_overflow_within_hard_cap = (
            budget.hard_limit is not None and budget.effective_total <= budget.hard_limit
        )
        if soft_cap_overflow_within_hard_cap:
            return False

        return not dense_fragments

    def _assess_merge(
        previous: str,
        current: str,
        prev_words: tuple[str, ...],
        current_words: tuple[str, ...],
        budget: _MergeBudget,
    ) -> _MergeDecision:
        if not previous:
            return _MergeDecision(False, False, False)
        lead = current.lstrip()
        if not lead:
            return _MergeDecision(False, False, False)
        continuation_lead = _is_continuation_lead(lead)
        if _ENDS_SENTENCE.search(previous.rstrip()) and not continuation_lead:
            return _MergeDecision(False, False, False)
        first_word = current_words[0] if current_words else ""
        if prev_words and prev_words[-1] == first_word:
            return _MergeDecision(False, False, False)
        dense_fragments = _is_dense_fragment(prev_words, current_words)
        previous_ends_sentence = bool(_ENDS_SENTENCE.search(previous.rstrip()))
        current_completes_sentence = bool(
            _ENDS_SENTENCE.search(current.rstrip())
        )
        view = _BudgetView(
            budget=budget,
            target_limit=target_limit,
            dense_fragments=dense_fragments,
            sentence_completion_pending=(
                current_completes_sentence and not previous_ends_sentence
            ),
        )
        allow_overflow = view.allow_overflow()
        budget_violation = view.violates()
        if budget_violation and not allow_overflow:
            return _MergeDecision(False, False, dense_fragments)
        if target_limit is not None and budget.effective_total > target_limit:
            if budget.hard_limit is None or budget.effective_total > budget.hard_limit:
                if dense_fragments and not allow_overflow:
                    return _MergeDecision(False, False, dense_fragments)
        head = lead[0]
        continuation_chars = ",.;:)]\"'"
        if not (continuation_lead or head.islower() or head in continuation_chars):
            return _MergeDecision(False, allow_overflow, dense_fragments)
        combined = len(previous) + 1 + len(current)
        if len(previous) >= SOFT_LIMIT or len(current) >= SOFT_LIMIT:
            return _MergeDecision(False, allow_overflow, dense_fragments)
        return _MergeDecision(combined <= SOFT_LIMIT, allow_overflow, dense_fragments)

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

        budget = _derive_merge_budget(
            prev_words,
            trimmed_words,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )
        decision = _assess_merge(
            prev_text,
            trimmed_text,
            prev_words,
            trimmed_words,
            budget,
        )
        if not decision.allowed:
            if trimmed_words != words:
                return _append(acc, trimmed_text, trimmed_words)
            return _append(acc, chunk, words)

        dense_fragments = decision.dense_fragments
        exceeds_limit = _violates_budget(budget, dense_fragments=dense_fragments)
        if exceeds_limit and not decision.allow_overflow:
            adjusted = (
                _rebalance_overflow(prev_text, prev_words, trimmed_text, _target_limit_from(budget))
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
                        overlap=overlap,
                        min_chunk_size=min_chunk_size,
                    )
                    dense_fragments = _is_dense_fragment(prev_words, trimmed_words)
                    exceeds_limit = _violates_budget(
                        budget, dense_fragments=dense_fragments
                    )
            if not trimmed_words:
                return acc
            if exceeds_limit:
                return _append(acc, trimmed_text, trimmed_words)

        merged_text = f"{prev_text} {trimmed_text}".strip()
        merged_words = (*prev_words, *trimmed_words)
        return [*acc[:-1], (merged_text, merged_words)]

    merged: list[tuple[str, tuple[str, ...]]] = reduce(
        _merge,
        chunks,
        [],
    )

    stitch_limit = hard_limit if hard_limit is not None else target_limit
    texts = [text for text, _ in merged]
    coalesced = _coalesce_sentence_runs(texts)
    return _stitch_continuation_heads(coalesced, stitch_limit)


def _coalesce_sentence_runs(chunks: Iterable[str]) -> list[str]:
    def _combine(parts: tuple[str, ...]) -> str:
        return " ".join(segment.strip() for segment in parts if segment.strip()).strip()

    def _flush(acc: list[str], buffer: tuple[str, ...]) -> list[str]:
        if not buffer:
            return acc
        if len(buffer) == 1:
            return [*acc, buffer[0]]
        has_sentence_end = any(
            _ENDS_SENTENCE.search(part.rstrip()) for part in buffer if part
        )
        if not has_sentence_end:
            return [*acc, *buffer]
        combined = _combine(buffer)
        return [*acc, combined] if combined else acc

    def _consume(state: tuple[list[str], tuple[str, ...]], chunk: str) -> tuple[list[str], tuple[str, ...]]:
        acc, buffer = state
        updated_buffer = (*buffer, chunk)
        if _ENDS_SENTENCE.search(chunk.rstrip()):
            return _flush(acc, updated_buffer), ()
        return acc, updated_buffer

    merged, pending = reduce(
        _consume,
        chunks,
        ([], ()),
    )
    return _flush(merged, pending)


def _stitch_continuation_heads(chunks: list[str], limit: int | None) -> list[str]:
    def _split_sentence_tail(text: str) -> tuple[str, str] | None:
        boundaries = tuple(
            match.end()
            for match in _SENTENCE_BOUNDARY.finditer(text)
            if match.end() < len(text)
        )
        if not boundaries:
            return None
        pivot = boundaries[-1]
        head = text[:pivot].rstrip()
        tail = text[pivot:].lstrip()
        if not head or not tail:
            return None
        return head, tail

    def _merge_tail_with_chunk(tail: str, chunk: str) -> str:
        tail_words = tuple(tail.split())
        chunk_words = tuple(chunk.split())
        if not tail_words:
            return chunk.strip()
        if not chunk_words:
            return tail.strip()
        overlap = next(
            (
                size
                for size in range(
                    min(len(tail_words), len(chunk_words)), 0, -1
                )
                if tail_words[-size:] == chunk_words[:size]
            ),
            0,
        )
        prefix_words = tail_words[:-overlap] if overlap else tail_words
        segments = (
            " ".join(prefix_words).strip(),
            chunk.strip(),
        )
        return " ".join(part for part in segments if part)

    def _redistribute_sentence_tail(
        previous: str, current: str, limit: int | None
    ) -> tuple[str, str] | None:
        split = _split_sentence_tail(previous)
        if split is None:
            return None
        head, tail = split
        head_words = tuple(head.split())
        if limit is not None and len(head_words) > limit:
            return None
        next_chunk = _merge_tail_with_chunk(tail, current)
        return head, next_chunk

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

        if limit is not None:
            total = len(prev_words) + len(chunk_words)
            if total > limit:
                redistributed = (
                    _redistribute_sentence_tail(prev, chunk, limit)
                    if not _ENDS_SENTENCE.search(prev.rstrip())
                    else None
                )
                if redistributed is not None:
                    prev, chunk = redistributed
                    prev_words = tuple(prev.split())
                    chunk_words = tuple(chunk.split())
                    acc = [*acc[:-1], prev]
                    remaining = chunk
                    total = len(prev_words) + len(chunk_words)
                if total > limit:
                    return [*acc, chunk]

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
