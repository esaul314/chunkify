"""Split ``page_blocks`` into canonical ``chunks``.

This pass wraps the legacy :mod:`pdf_chunker.splitter` semantic chunker
while keeping a pure function boundary. When the splitter cannot be
imported, each block becomes a single chunk. Chunks carry page and source
metadata so downstream passes can enrich and emit JSONL rows.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import partial, reduce
from itertools import accumulate, chain
from math import ceil
from typing import Any, TypedDict, cast

from pdf_chunker.framework import Artifact, Pass, register
from pdf_chunker.interactive import (
    ListContinuationCache,
    ListContinuationCallback,
    ListContinuationConfig,
    _candidate_continues_list_item,
    _looks_like_list_item,
)
from pdf_chunker.passes.chunk_options import (
    SplitMetrics,
    SplitOptions,
    derive_min_chunk_size,
)
from pdf_chunker.passes.chunk_pipeline import (
    attach_headings as pipeline_attach_headings,
)
from pdf_chunker.passes.chunk_pipeline import (
    chunk_records as pipeline_chunk_records,
)
from pdf_chunker.passes.chunk_pipeline import (
    iter_blocks as pipeline_iter_blocks,
)
from pdf_chunker.passes.chunk_pipeline import (
    merge_adjacent_blocks as pipeline_merge_adjacent_blocks,
)
from pdf_chunker.passes.sentence_fusion import (
    _AVERAGE_CHARS_PER_TOKEN,
    _ENDS_SENTENCE,
    SOFT_LIMIT,
    _is_continuation_lead,
    _last_sentence,
    _merge_sentence_fragments,
)
from pdf_chunker.passes.split_modules.inline_headings import (
    promote_inline_heading as _promote_inline_heading,
)
from pdf_chunker.passes.split_modules.inline_headings import (
    split_inline_heading_records as _split_inline_heading_records,
)
from pdf_chunker.passes.split_modules.lists import (
    colon_bullet_boundary as _colon_bullet_boundary,
)
from pdf_chunker.passes.split_modules.lists import (
    record_is_list_like as _record_is_list_like,
)
from pdf_chunker.passes.split_modules.lists import (
    starts_list_like as _starts_list_like,
)
from pdf_chunker.passes.split_modules.overlap import (
    restore_overlap_words as _restore_overlap_words,
)
from pdf_chunker.passes.split_modules.overlap import (
    trim_boundary_overlap as _trim_boundary_overlap,
)
from pdf_chunker.passes.split_modules.segments import (
    _collapse_records,
)
from pdf_chunker.passes.split_modules.stitching import (
    is_heading as _is_heading,
)
from pdf_chunker.passes.split_modules.stitching import (
    stitch_block_continuations as _stitch_block_continuations,
)
from pdf_chunker.passes.split_semantic_lists import (
    _append_caption,
    _apply_envelope,
    _BlockEnvelope,
    _contains_caption_line,
    _has_caption,
    _looks_like_caption,
    _mark_caption,
    _merge_styled_list_records,
    _resolve_envelope,
)
from pdf_chunker.passes.split_semantic_metadata import (
    Chunk,
    _chunk_meta,
    _merge_heading_texts,
    _meta_is_list,
    build_chunk,
    build_chunk_with_meta,
)
from pdf_chunker.passes.transform_log import TransformationLog
from pdf_chunker.strategies.bullets import (
    BulletHeuristicStrategy,
    default_bullet_strategy,
)

Doc = dict[str, Any]
Block = dict[str, Any]
Record = tuple[int, Block, str]
SplitFn = Callable[[str], list[str]]
MetricFn = Callable[[], dict[str, int | bool]]

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\S+")


def _resolve_bullet_strategy(
    strategy: BulletHeuristicStrategy | None,
) -> BulletHeuristicStrategy:
    return strategy or default_bullet_strategy()


def _warn_stitching_issue(message: str, *, page: int | None = None) -> None:
    if not message:
        return
    detail = f"{message} (page={page})" if page is not None else message
    logger.warning("split_semantic: %s", detail)


# ---------------------------------------------------------------------------
# Inline heading functions delegated to split_modules.inline_headings
# ---------------------------------------------------------------------------
# The following functions are now imported from pdf_chunker.passes.split_modules.inline_headings:
#   _split_inline_heading, _split_inline_heading_records, _promote_inline_heading


def _segment_char_limit(chunk_size: int | None) -> int:
    """Return a soft-segmentation character limit for ``chunk_size``."""

    if chunk_size is None or chunk_size <= 0:
        return SOFT_LIMIT
    estimated = int(ceil(chunk_size * _AVERAGE_CHARS_PER_TOKEN))
    return max(SOFT_LIMIT, max(1, estimated))


def _soft_segments(
    text: str,
    *,
    max_chars: int | None = None,
    max_words: int | None = None,
) -> list[str]:
    """Split ``text`` while honouring character and word ceilings."""

    limit = SOFT_LIMIT if max_chars is None or max_chars <= 0 else min(max_chars, SOFT_LIMIT)
    word_limit = max_words if max_words is not None and max_words > 0 else None

    def _split(chunk: str) -> Iterator[str]:
        trimmed = chunk.strip()
        if not trimmed:
            return

        if word_limit is not None:
            tokens = tuple(_TOKEN_PATTERN.finditer(trimmed))
            if len(tokens) > word_limit:
                split_index = tokens[word_limit - 1].end()
                head = trimmed[:split_index]
                tail = trimmed[split_index:]
                if head:
                    yield from _split(head)
                if tail:
                    yield from _split(tail)
                return

        if len(trimmed) <= limit:
            yield trimmed
            return

        pivot = trimmed.rfind(" ", 0, limit)
        boundary = pivot if pivot != -1 else limit
        head = trimmed[:boundary]
        tail = trimmed[boundary:]
        if head:
            yield from _split(head)
        if tail:
            yield from _split(tail)

    return list(_split(text))


# ---------------------------------------------------------------------------
# Overlap functions delegated to split_modules.overlap
# ---------------------------------------------------------------------------
# The following functions are now imported from pdf_chunker.passes.split_modules.overlap:
#   _restore_overlap_words, _restore_chunk_overlap, _missing_overlap_prefix,
#   _prepend_words, _is_overlap_token, _trim_sentence_prefix, _split_words,
#   _overlap_window, _overlap_text, _should_trim_overlap, _trim_tokens,
#   _trim_boundary_overlap


# ---------------------------------------------------------------------------
# Stitching functions delegated to split_modules.stitching
# ---------------------------------------------------------------------------
# The following functions are now imported from pdf_chunker.passes.split_modules.stitching:
#   _stitch_block_continuations, _merge_record_block, _with_chunk_index, _is_heading


# ---------------------------------------------------------------------------
# List functions delegated to split_modules.lists
# ---------------------------------------------------------------------------
# The following functions are now imported from pdf_chunker.passes.split_modules.lists:
#   _starts_list_like, _record_is_list_like, _colon_bullet_boundary,
#   _first_list_number, _last_list_number


# ---------------------------------------------------------------------------
# Footer functions delegated to split_modules.footers
# ---------------------------------------------------------------------------
# The following functions are now imported from pdf_chunker.passes.split_modules.footers:
#   _resolve_footer_suffix, _record_trailing_footer_lines, _record_is_footer_candidate,
#   _strip_footer_suffix, _strip_footer_suffixes


# ---------------------------------------------------------------------------
# Segment emission functions delegated to split_modules.segments
# ---------------------------------------------------------------------------
# The following functions are now imported from pdf_chunker.passes.split_modules.segments:
#   _effective_counts, _should_emit_list_boundary, _segment_totals, _resolved_limit,
#   _hard_limit, _overlap_value, _emit_buffer_segments, _merged_segment_record,
#   _emit_individual_records, _emit_segment_records, _segment_allows_list_overflow,
#   _segment_is_colon_list, _maybe_merge_dense_page, _CollapseEmitter,
#   _page_or_footer_boundary, _allow_cross_page_list, _buffer_last_list_number,
#   _record_starts_numbered_item, _allow_list_overflow, _allow_colon_list_overflow,
#   _text_has_number_two, _buffer_has_number_two, _projected_overflow,
#   _collapse_step, _collapse_records


def _list_tail_split_index(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> int | None:
    heuristics = _resolve_bullet_strategy(strategy)
    """Return the index where a list block transitions into narrative text."""

    parts = text.splitlines(keepends=True)
    if len(parts) <= 1:
        return None
    offsets = tuple(accumulate(len(part) for part in parts))
    candidates = zip(range(1, len(parts)), parts[1:], offsets[:-1], strict=False)
    for idx, part, offset in candidates:
        stripped = part.lstrip()
        if not stripped:
            continue
        if heuristics.starts_with_bullet(stripped) or heuristics.starts_with_number(stripped):
            continue
        indent = len(part) - len(stripped)
        if indent:
            continue
        prefix = "".join(parts[:idx]).rstrip()
        if not prefix or prefix[-1] not in ".?!":
            continue
        if _is_continuation_lead(stripped) or stripped[0].islower():
            continue
        return offset
    return None


def _split_list_record(
    record: tuple[int, Block, str],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[tuple[int, Block, str], ...]:
    page, block, text = record
    split_at = _list_tail_split_index(text, strategy=strategy)
    if split_at is None:
        return (record,)
    head_text = text[:split_at].rstrip()
    tail_text = text[split_at:].lstrip("\n")
    if not head_text or not tail_text:
        return (record,)
    head_block = {**block, "text": head_text}
    tail_envelope = _resolve_envelope((block,), default_list_kind=None)
    non_list_envelope = _BlockEnvelope(tail_envelope.block_type, None)
    tail_block = _apply_envelope(block, tail_text, non_list_envelope)
    return ((page, head_block, head_text), (page, tail_block, tail_text))


def _split_colon_bullet_segments(
    buffer: Iterable[tuple[int, Block, str]],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[tuple[tuple[int, Block, str], ...], ...]:
    """Return ``buffer`` sliced so colon-prefixed bullets anchor a new segment."""

    def _append_segment(
        acc: tuple[tuple[tuple[int, Block, str], ...], ...],
        record: tuple[int, Block, str],
    ) -> tuple[tuple[tuple[int, Block, str], ...], ...]:
        if not acc:
            return ((record,),)
        previous = acc[-1]
        prev_text = previous[-1][2]
        if _colon_bullet_boundary(
            prev_text,
            record[1],
            record[2],
            strategy=strategy,
        ):
            prefix = previous[:-1]
            colon_record = previous[-1]
            head = acc[:-1]
            head = (*head, prefix) if prefix else head
            return (*head, (colon_record, record))
        if _record_is_list_like(previous[-1], strategy=strategy) and not _record_is_list_like(
            record,
            strategy=strategy,
        ):
            return (*acc, (record,))
        return (*acc[:-1], previous + (record,))

    return reduce(_append_segment, buffer, tuple())


def _expand_segment_records(
    segment: tuple[tuple[int, Block, str], ...],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[tuple[int, Block, str], ...]:
    expanded = tuple(
        chain.from_iterable(_split_list_record(record, strategy=strategy) for record in segment)
    )
    return expanded if expanded else segment


def _segment_offsets(segments: tuple[tuple[tuple[int, Block, str], ...], ...]) -> tuple[int, ...]:
    if not segments:
        return tuple()
    counts = accumulate(len(segment) for segment in segments[:-1])
    return tuple(chain((0,), counts))


def _enumerate_segments(
    segments: tuple[tuple[tuple[int, Block, str], ...], ...],
) -> tuple[tuple[int, tuple[tuple[int, Block, str], ...]], ...]:
    offsets = _segment_offsets(segments)
    return tuple(zip(offsets, segments, strict=False))


def _collapse_numbered_list_spacing(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> str:
    """Collapse blank lines between numbered items while preserving others."""

    if "\n\n" not in text:
        return text
    lines = tuple(text.splitlines())
    if len(lines) <= 2:
        return text

    heuristics = _resolve_bullet_strategy(strategy)

    def _is_numbered(line: str) -> bool:
        stripped = line.lstrip()
        return bool(stripped) and heuristics.starts_with_number(stripped)

    def _keep(index_line: tuple[int, str]) -> bool:
        index, line = index_line
        if line.strip():
            return True
        if index == 0 or index == len(lines) - 1:
            return True
        prev_line, next_line = lines[index - 1], lines[index + 1]
        return not (_is_numbered(prev_line) and _is_numbered(next_line))

    filtered = tuple(line for index, line in enumerate(lines) if _keep((index, line)))
    return "\n".join(filtered)


def _normalize_numbered_list_text(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> str:
    """Normalize numbered list spacing without altering other content."""

    return _collapse_numbered_list_spacing(text, strategy=strategy) if text else text


def _join_record_texts(
    records: Iterable[tuple[int, Block, str]],
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> str:
    joined = "\n\n".join(part.strip() for _, _, part in records if part.strip()).strip()
    return _normalize_numbered_list_text(joined, strategy=strategy) if joined else joined


def _apply_overlap_within_segment(
    segment: tuple[tuple[int, Block, str], ...], overlap: int
) -> tuple[tuple[int, Block, str], ...]:
    """Apply boundary trimming within ``segment`` before joining blocks."""
    if overlap <= 0 or len(segment) <= 1:
        return segment

    def _merge(
        acc: tuple[tuple[int, Block, str], ...],
        record: tuple[int, Block, str],
    ) -> tuple[tuple[int, Block, str], ...]:
        prev_text = acc[-1][2]
        trimmed = _trim_boundary_overlap(prev_text, record[2], overlap)
        updated = (record[0], record[1], trimmed)
        return acc + (updated,)

    return reduce(_merge, segment[1:], (segment[0],))


# _segment_totals imported from split_modules.segments
# _resolved_limit imported from split_modules.segments
# _hard_limit imported from split_modules.segments
# _overlap_value imported from split_modules.segments
# _emit_buffer_segments imported from split_modules.segments
# _merged_segment_record imported from split_modules.segments
# _emit_individual_records imported from split_modules.segments
# _emit_segment_records imported from split_modules.segments
# _segment_allows_list_overflow imported from split_modules.segments
# _segment_is_colon_list imported from split_modules.segments
# _maybe_merge_dense_page imported from split_modules.segments
# _CollapseEmitter imported from split_modules.segments
# _page_or_footer_boundary imported from split_modules.segments


_NUMBERED_ANYWHERE_RE = re.compile(r"(?:^|\n|\s)(\d+)[.)]\s+")
_NUMBERED_TWO_RE = re.compile(r"(?:^|\n|\s)2[.)]\s+")


# Note: _first_list_number and _last_list_number are now imported from split_modules.lists
# _allow_cross_page_list imported from split_modules.segments
# _buffer_last_list_number imported from split_modules.segments
# _record_starts_numbered_item imported from split_modules.segments
# _allow_list_overflow imported from split_modules.segments
# _allow_colon_list_overflow imported from split_modules.segments
# _text_has_number_two imported from split_modules.segments
# _buffer_has_number_two imported from split_modules.segments
# _projected_overflow imported from split_modules.segments
# _collapse_step imported from split_modules.segments
# _collapse_records imported from split_modules.segments


class _OverrideOpts(TypedDict, total=False):
    chunk_size: int
    overlap: int
    generate_metadata: bool


def _get_split_fn(
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> tuple[SplitFn, MetricFn]:
    """Return a semantic splitter enforcing size limits and collecting metrics."""

    soft_hits = 0
    char_limit = _segment_char_limit(chunk_size)

    try:
        from pdf_chunker.splitter import semantic_chunker

        semantic = partial(
            semantic_chunker,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

        def split(text: str) -> list[str]:
            """Split ``text`` while guarding against truncation."""

            nonlocal soft_hits
            pieces = semantic(text)
            merged = pieces if sum(len(p.split()) for p in pieces) >= len(text.split()) else [text]

            def _soften(segment: str) -> list[str]:
                nonlocal soft_hits
                splits = _soft_segments(
                    segment,
                    max_chars=char_limit,
                    max_words=chunk_size,
                )
                if len(splits) > 1:
                    soft_hits += 1
                return splits

            raw = [softened for chunk in merged for softened in _soften(chunk)]
            final = _merge_sentence_fragments(
                raw,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min_chunk_size,
            )
            soft_hits += sum(len(c) > SOFT_LIMIT for c in final)
            return _restore_overlap_words(final, overlap)

    except Exception:  # pragma: no cover - safety fallback

        def split(text: str) -> list[str]:
            nonlocal soft_hits
            raw = _soft_segments(
                text,
                max_chars=char_limit,
                max_words=chunk_size,
            )
            final = _merge_sentence_fragments(
                raw,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min_chunk_size,
            )
            soft_hits += sum(len(seg) > SOFT_LIMIT for seg in final)
            return _restore_overlap_words(final, overlap)

    def metrics() -> dict[str, int]:
        return {"soft_limit_hits": soft_hits}

    return split, metrics


def _block_text(block: Block) -> str:
    return block.get("text", "")


# Note: _starts_list_like is now imported from split_modules.lists


def _should_break_after_colon(
    prev_text: str,
    block: Block,
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    if not prev_text.rstrip().endswith(":"):
        return False
    lead = text.lstrip()
    if not lead:
        return False
    head = lead[0]
    return (
        head.isupper()
        or head.isdigit()
        or _starts_list_like(
            block,
            text,
            strategy=strategy,
        )
    )


def _merge_blocks(
    acc: list[tuple[int, Block, str]],
    cur: tuple[int, Block, str],
    *,
    strategy: BulletHeuristicStrategy | None = None,
    list_continuation_config: ListContinuationConfig | None = None,
    list_continuation_cache: ListContinuationCache | None = None,
) -> list[tuple[int, Block, str]]:
    heuristics = _resolve_bullet_strategy(strategy)
    page, block, text = cur
    block = _promote_inline_heading(block, text)
    cur = (page, block, text)
    if not acc:
        return acc + [cur]
    prev_page, prev_block, prev_text = acc[-1]
    if prev_page != page:
        return acc + [cur]
    if block is prev_block and _is_heading(block) and _looks_like_caption(prev_text):
        merged = " ".join(part for part in (prev_text, text) if part).strip()
        acc[-1] = (prev_page, prev_block, merged)
        return acc
    if _looks_like_caption(text):
        if _has_caption(prev_block) or _contains_caption_line(prev_text):
            return acc + [cur]
        acc[-1] = (prev_page, _mark_caption(prev_block), _append_caption(prev_text, text))
        return acc
    if _is_heading(prev_block) or _is_heading(block):
        return acc + [cur]
    if _starts_list_like(block, text, strategy=heuristics):
        return acc + [cur]
    if _should_break_after_colon(prev_text, block, text, strategy=heuristics):
        return acc + [cur]

    # Check for list continuation - previous is a list item, current might continue it
    if _looks_like_list_item(prev_text) and not _looks_like_list_item(text):
        should_merge, confidence, reason = _candidate_continues_list_item(
            prev_text,
            text,
            confidence_threshold=0.5 if list_continuation_config else 0.7,
        )
        # Interactive confirmation for uncertain cases
        if list_continuation_config is not None and list_continuation_config.callback is not None:
            # Check if already cached
            cached = None
            if list_continuation_cache is not None:
                cached = list_continuation_cache.get(prev_text, text)
            if cached is not None:
                should_merge = cached
            elif 0.4 <= confidence <= 0.85:
                # Ask user for confirmation in uncertain cases
                ctx = {
                    "heuristic_confidence": confidence,
                    "heuristic_reason": reason,
                    "page": page,
                }
                should_merge = list_continuation_config.callback(prev_text, text, page, ctx)
                if list_continuation_cache is not None:
                    list_continuation_cache.set(prev_text, text, should_merge)
        if should_merge:
            acc[-1] = (prev_page, prev_block, f"{prev_text} {text}".strip())
            return acc

    lead = text.lstrip()
    continuation_chars = ",.;:)]\"'"
    prev_ends_sentence = _ENDS_SENTENCE.search(prev_text.rstrip())
    if lead and (
        _is_continuation_lead(lead)
        or (not prev_ends_sentence and (lead[0].islower() or lead[0] in continuation_chars))
    ):
        acc[-1] = (prev_page, prev_block, f"{prev_text} {text}".strip())
        return acc
    return acc + [cur]


def _block_texts(
    doc: Doc,
    split_fn: SplitFn,
    *,
    strategy: BulletHeuristicStrategy | None = None,
    list_continuation_config: ListContinuationConfig | None = None,
    list_continuation_cache: ListContinuationCache | None = None,
) -> Iterator[tuple[int, Block, str]]:
    """Yield ``(page, block, text)`` triples after merging sentence fragments."""

    merged = pipeline_merge_adjacent_blocks(
        pipeline_iter_blocks(doc),
        text_of=_block_text,
        fold=partial(
            _merge_blocks,
            strategy=strategy,
            list_continuation_config=list_continuation_config,
            list_continuation_cache=list_continuation_cache,
        ),
        split_fn=split_fn,
    )
    return _split_inline_heading_records(merged)


# _is_heading is imported from pdf_chunker.passes.split_modules.stitching


def _chunk_items(
    doc: Doc,
    split_fn: SplitFn,
    generate_metadata: bool = True,
    *,
    options: SplitOptions | None = None,
    strategy: BulletHeuristicStrategy | None = None,
    list_continuation_config: ListContinuationConfig | None = None,
    list_continuation_cache: ListContinuationCache | None = None,
    transform_log: TransformationLog | None = None,
) -> Iterator[Chunk]:
    """Yield chunk records from ``doc`` using ``split_fn``."""

    filename = doc.get("source_path")
    limit = options.compute_limit() if options is not None else None
    heuristics = _resolve_bullet_strategy(strategy)
    stages: tuple[Callable[[Iterable[Record]], Iterable[Record]], ...] = (
        partial(
            pipeline_attach_headings,
            is_heading=_is_heading,
            merge_block_text=partial(_merge_heading_texts, strategy=heuristics),
        ),
        _merge_styled_list_records,
        partial(
            _stitch_block_continuations,
            limit=limit,
            strategy=heuristics,
            transform_log=transform_log,
        ),
    )
    records: Iterable[Record] = _block_texts(
        doc,
        split_fn,
        strategy=heuristics,
        list_continuation_config=list_continuation_config,
        list_continuation_cache=list_continuation_cache,
    )
    for stage in stages:
        records = stage(records)
    collapsed = _collapse_records(records, options, limit, strategy=heuristics)
    builder = partial(build_chunk_with_meta, filename=filename, bullet_strategy=heuristics)
    return pipeline_chunk_records(
        collapsed,
        generate_metadata=generate_metadata,
        build_plain=build_chunk,
        build_with_meta=builder,
    )


def _inject_continuation_context(
    items: Iterable[Chunk], limit: int | None, overlap: int
) -> Iterator[Chunk]:
    prev_text: str | None = None
    prev_meta: Mapping[str, Any] | None = None
    for item in items:
        original = item.get("text", "")
        current_meta = _chunk_meta(item)
        current_is_list = _meta_is_list(current_meta)
        can_trim = prev_text is not None and not current_is_list and not _meta_is_list(prev_meta)
        trimmed = (
            _trim_boundary_overlap(cast(str, prev_text), original, overlap)
            if can_trim
            else original
        )
        text = trimmed
        was_trimmed = text != original
        if was_trimmed:
            item = {**item, "text": text}
        lead = text.lstrip()
        if (
            prev_text is None
            or overlap <= 0
            or not lead
            or was_trimmed
            or current_is_list
            or _meta_is_list(prev_meta)
            or not _is_continuation_lead(lead)
        ):
            prev_text = text
            prev_meta = current_meta
            yield item
            continue
        context = _last_sentence(prev_text)
        if not context or lead.startswith(context):
            prev_text = text
            prev_meta = current_meta
            yield item
            continue
        combined = f"{context} {text}".strip()
        prev_text = combined
        prev_meta = current_meta
        yield {**item, "text": combined}


@dataclass
class _SplitSemanticPass:
    name: str = field(default="split_semantic", init=False)
    input_type: type = field(
        default=dict, init=False
    )  # expects {"type": "page_blocks"}  # noqa: E501
    output_type: type = field(
        default=dict, init=False
    )  # returns {"type": "chunks", "items": [...]}
    chunk_size: int = 400
    overlap: int = 50
    min_chunk_size: int | None = None
    generate_metadata: bool = True
    bullet_strategy: BulletHeuristicStrategy | None = field(
        default_factory=default_bullet_strategy,
    )
    interactive_lists: bool = False
    list_continuation_callback: ListContinuationCallback | None = None

    def __post_init__(self) -> None:
        self.min_chunk_size = derive_min_chunk_size(
            self.chunk_size, self.min_chunk_size
        )  # noqa: E501

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload
        if not isinstance(doc, dict) or doc.get("type") != "page_blocks":
            return a
        options = SplitOptions.from_base(
            self.chunk_size, self.overlap, self.min_chunk_size
        ).with_meta(a.meta)
        split_fn, metric_fn = _get_split_fn(
            options.chunk_size, options.overlap, options.min_chunk_size
        )
        limit = options.compute_limit()
        strategy = _resolve_bullet_strategy(self.bullet_strategy)

        # Build list continuation config from options
        list_config: ListContinuationConfig | None = None
        list_cache: ListContinuationCache | None = None
        runtime_interactive = a.meta.get("interactive_lists", False) if a.meta else False
        if self.interactive_lists or runtime_interactive or self.list_continuation_callback:
            callback = self.list_continuation_callback
            if callback is None and (self.interactive_lists or runtime_interactive):
                # Import here to avoid circular dependency at module level
                from pdf_chunker.interactive import make_cli_list_continuation_prompt

                callback = make_cli_list_continuation_prompt()
            list_config = ListContinuationConfig(callback=callback)
            list_cache = ListContinuationCache()

        chunk_records = _chunk_items(
            doc,
            split_fn,
            self.generate_metadata,
            options=options,
            strategy=strategy,
            list_continuation_config=list_config,
            list_continuation_cache=list_cache,
        )
        items = list(
            _inject_continuation_context(
                chunk_records, limit, options.overlap if options else self.overlap
            )
        )
        meta = SplitMetrics(len(items), metric_fn()).apply(a.meta)
        return Artifact(payload={"type": "chunks", "items": items}, meta=meta)


DEFAULT_SPLITTER = _SplitSemanticPass()


def make_splitter(**opts: Any) -> _SplitSemanticPass:
    """Return a configured ``split_semantic`` pass from ``opts``."""
    opts_map: _OverrideOpts = {
        "chunk_size": int(opts.get("chunk_size", DEFAULT_SPLITTER.chunk_size)),
        "overlap": int(opts.get("overlap", DEFAULT_SPLITTER.overlap)),
        "generate_metadata": bool(
            opts.get("generate_metadata", DEFAULT_SPLITTER.generate_metadata)
        ),
    }
    base = replace(DEFAULT_SPLITTER, **opts_map)
    if "chunk_size" in opts and "min_chunk_size" not in opts:
        base = replace(base, min_chunk_size=None)
    strategy = opts.get("bullet_strategy")
    if isinstance(strategy, BulletHeuristicStrategy):
        base = replace(base, bullet_strategy=strategy)
    # Handle interactive list continuation
    if opts.get("interactive_lists"):
        base = replace(base, interactive_lists=True)
    callback = opts.get("list_continuation_callback")
    if callback is not None:
        base = replace(base, list_continuation_callback=callback)
    base.__post_init__()
    return base


split_semantic: Pass = register(make_splitter())

# fmt: on
