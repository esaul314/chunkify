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
from pdf_chunker.list_detection import starts_with_bullet, starts_with_number
from pdf_chunker.page_artifacts import (
    _bullet_body,
    _drop_trailing_bullet_footers,
    _footer_bullet_signals,
    _header_invites_footer,
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
from pdf_chunker.passes.split_semantic_inline import (
    _body_styles,
    _heading_styles,
    _leading_heading_candidate,
    _next_non_whitespace,
    _span_bounds,
    _span_style,
    _trimmed_segment,
)
from pdf_chunker.passes.split_semantic_lists import (
    _STYLED_LIST_KIND,
    _append_caption,
    _apply_envelope,
    _block_list_kind,
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

Doc = dict[str, Any]
Block = dict[str, Any]
Record = tuple[int, Block, str]
SplitFn = Callable[[str], list[str]]
MetricFn = Callable[[], dict[str, int | bool]]

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\S+")


def _warn_stitching_issue(message: str, *, page: int | None = None) -> None:
    if not message:
        return
    detail = f"{message} (page={page})" if page is not None else message
    logger.warning("split_semantic: %s", detail)


def _split_inline_heading(block: Block, text: str) -> tuple[Block, Block] | None:
    if not text or block.get("type") not in {None, "paragraph"}:
        return None
    styles = tuple(block.get("inline_styles") or ())
    if not styles:
        return None
    candidate = _leading_heading_candidate(text, styles)
    if candidate is None:
        return None
    _, end = candidate
    prefix = text[:end]
    suffix = text[end:]
    heading_text, heading_lead_trim = _trimmed_segment(prefix)
    body_text = suffix.lstrip()
    if not heading_text or not body_text:
        return None
    if len(heading_text.split()) > 6:
        return None
    if len(body_text.split()) < 5:
        return None
    trailer = _next_non_whitespace(text, end)
    if trailer is None or trailer in ",;:-–—" or trailer.islower():
        return None
    body_lead_trim = len(suffix) - len(body_text)
    text_length = len(text)
    heading_limit = len(heading_text)
    body_offset = end + body_lead_trim
    body_limit = len(body_text)
    heading_styles = _heading_styles(
        styles,
        text_length,
        end,
        heading_lead_trim,
        heading_limit,
    )
    body_styles = _body_styles(styles, text_length, body_offset, body_limit)
    base = {key: value for key, value in dict(block).items() if key != "text"}
    base.pop("inline_styles", None)
    heading_block = {
        **{k: v for k, v in base.items() if k != "list_kind"},
        "type": "heading",
        "text": heading_text,
    }
    if heading_styles:
        heading_block["inline_styles"] = heading_styles
    body_block = {
        **base,
        "type": "list_item",
        "list_kind": base.get("list_kind") or _STYLED_LIST_KIND,
        "text": body_text,
    }
    if body_styles:
        body_block["inline_styles"] = body_styles
    return heading_block, body_block


def _split_inline_heading_records(
    records: Iterable[tuple[int, Block, str]],
) -> Iterator[tuple[int, Block, str]]:
    for page, block, text in records:
        split = _split_inline_heading(block, text)
        if not split:
            yield page, block, text
            continue
        heading_block, body_block = split
        yield page, heading_block, heading_block.get("text", "")
        yield page, body_block, body_block.get("text", "")


def _segment_char_limit(chunk_size: int | None) -> int:
    """Return a soft-segmentation character limit for ``chunk_size``."""

    if chunk_size is None or chunk_size <= 0:
        return SOFT_LIMIT
    estimated = int(ceil(chunk_size * _AVERAGE_CHARS_PER_TOKEN))
    return min(SOFT_LIMIT, max(1, estimated))


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


def _restore_overlap_words(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return chunks
    restored: list[str] = []
    previous_words: tuple[str, ...] = ()
    previous_text = ""

    for chunk in chunks:
        updated = chunk
        words = tuple(chunk.split())
        prefilled = False

        if previous_words:
            window = min(overlap, len(previous_words))
            if window:
                overlap_words = tuple(previous_words[-window:])
                match_limit = min(window, len(words))
                matched = next(
                    (
                        size
                        for size in range(match_limit, 0, -1)
                        if tuple(overlap_words[-size:]) == tuple(words[:size])
                    ),
                    0,
                )
                missing_count = window - matched
                if missing_count > 0 and overlap_words:
                    prefix = " ".join(overlap_words[:missing_count])
                    if prefix:
                        glue = " " if updated and not updated[0].isspace() else ""
                        updated = f"{prefix}{glue}{updated}"
                        prefilled = True
                if not prefilled:
                    updated = _trim_sentence_prefix(previous_text, updated)

        restored.append(updated)
        previous_words = tuple(updated.split())
        previous_text = updated

    return restored


def _trim_sentence_prefix(previous_text: str, text: str) -> str:
    if not previous_text or not text:
        return text
    sentence = _last_sentence(previous_text)
    if not sentence:
        return text
    candidate = sentence.strip()
    if not candidate or candidate[-1] not in {".", "?", "!"}:
        return text
    if re.search(r"Chapter\s+\d+", candidate):
        return text
    if not text.startswith(candidate):
        return text
    remainder = text[len(candidate) :]
    if not remainder or not remainder.strip():
        return text
    match = re.search(
        r"((?:Chapter|Section|Part)\s+[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?)\.?$", candidate
    )  # noqa: E501
    preserved = match.group(0) if match else ""
    leading_space = remainder.startswith(" ")
    gap = remainder[1:] if leading_space else remainder
    if preserved:
        if gap.startswith("\n"):
            return f"{preserved}{gap}"
        spacer = "" if not gap or gap[0].isspace() else " "
        return f"{preserved}{spacer}{gap}"
    return gap


def _split_words(text: str) -> tuple[str, ...]:
    return tuple(text.split())


def _overlap_window(prev_words: tuple[str, ...], current_words: tuple[str, ...], limit: int) -> int:
    match_limit = min(limit, len(prev_words), len(current_words))
    return next(
        (size for size in range(match_limit, 0, -1) if prev_words[-size:] == current_words[:size]),
        0,
    )


def _overlap_text(words: tuple[str, ...], size: int) -> str:
    return " ".join(words[:size]).strip()


def _should_trim_overlap(segment: str) -> bool:
    return bool(segment) and segment[-1] in {".", "?", "!"}


def _trim_tokens(text: str, count: int) -> str:
    matches = list(_TOKEN_PATTERN.finditer(text))
    if len(matches) >= count:
        cut = matches[count - 1].end()
        return text[cut:].lstrip(" ")
    return ""


def _trim_boundary_overlap(prev_text: str, text: str, overlap: int) -> str:
    if overlap <= 0 or not prev_text or not text:
        return text
    previous_words = _split_words(prev_text)
    current_words = _split_words(text)
    window = min(overlap, len(previous_words))
    if not window:
        return text
    matched = _overlap_window(previous_words, current_words, window)
    if not matched or len(current_words) <= matched:
        return text
    if _looks_like_caption(text):
        return text
    overlap_segment = _overlap_text(current_words, matched)
    if not _should_trim_overlap(overlap_segment):
        return text
    return _trim_tokens(text, matched)


def _promote_inline_heading(block: Block, text: str) -> Block:
    """Return ``block`` promoted to a heading when inline styles indicate one."""

    if block.get("type") == "heading":
        return block

    styles = tuple(block.get("inline_styles") or ())
    if not styles:
        return block

    length = len(text)

    def _covers_entire(style: Any) -> bool:
        bounds = _span_bounds(style, length)
        if bounds is None:
            return False
        start, end = bounds
        return start == 0 and end >= length

    def _is_heading_style(style: Any) -> bool:
        flavor = _span_style(style).lower()
        return flavor in {"bold", "italic", "small_caps", "caps", "uppercase"}

    word_limit = len(tuple(token for token in text.split() if token))
    if word_limit > 12:
        return block

    if any(_covers_entire(style) and _is_heading_style(style) for style in styles):
        return {**block, "type": "heading"}

    return block


def _stitch_block_continuations(
    seq: Iterable[tuple[int, Block, str]], limit: int | None
) -> list[tuple[int, Block, str]]:
    def _consume(
        acc: list[tuple[int, Block, str]],
        cur: tuple[int, Block, str],
    ) -> list[tuple[int, Block, str]]:
        page, block, text = cur
        if not acc:
            return [*acc, cur]
        if _starts_list_like(block, text):
            return [*acc, cur]
        lead = text.lstrip()
        if not lead or not _is_continuation_lead(lead):
            return [*acc, cur]
        context = _last_sentence(acc[-1][2])
        if not context or text.lstrip().startswith(context):
            return [*acc, cur]
        context_words = tuple(context.split())
        text_words = tuple(text.split())
        if limit is not None and len(text_words) + len(context_words) > limit:
            _warn_stitching_issue(
                "continuation context skipped due to chunk limit",
                page=acc[-1][0],
            )
            merged = f"{acc[-1][2]} {text}".strip()
            return [*acc[:-1], (acc[-1][0], acc[-1][1], merged)]
        enriched = f"{context} {text}".strip()
        return [*acc, (page, block, enriched)]

    return reduce(_consume, seq, [])


def _merge_record_block(records: list[tuple[int, Block, str]], text: str) -> Block:
    blocks = tuple(block for _, block, _ in records)
    envelope = _resolve_envelope(blocks)
    first = blocks[0] if blocks else {}
    return _apply_envelope(first, text, envelope)


def _with_chunk_index(block: Block, index: int) -> Block:
    return {**block, "_chunk_start_index": index}


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


def _colon_bullet_boundary(prev_text: str, block: Block, text: str) -> bool:
    return prev_text.rstrip().endswith(":") and _starts_list_like(block, text)


def _record_is_list_like(record: tuple[int, Block, str]) -> bool:
    _, block, text = record
    return _starts_list_like(block, text)


def _previous_non_empty_line(lines: tuple[str, ...]) -> str:
    return next((line for line in reversed(lines) if line.strip()), "")


def _footer_context_allows(previous_line: str, trailing_count: int) -> bool:
    return any(
        (
            _footer_bullet_signals("", previous_line),
            _header_invites_footer(previous_line, trailing_count),
        )
    )


def _footer_line_is_artifact(line: str, previous_line: str) -> bool:
    body = _bullet_body(line)
    return not body or _footer_bullet_signals(body, previous_line)


def _resolve_footer_suffix(lines: tuple[str, ...]) -> tuple[str, ...]:
    pruned = tuple(_drop_trailing_bullet_footers(list(lines)))
    if len(pruned) == len(lines):
        return tuple()
    suffix = lines[len(pruned) :]
    if not suffix:
        return tuple()
    previous_line = _previous_non_empty_line(pruned)
    if not _footer_context_allows(previous_line, len(suffix)):
        return tuple()
    if not all(_footer_line_is_artifact(line, previous_line) for line in suffix):
        return tuple()
    return suffix


def _record_trailing_footer_lines(record: tuple[int, Block, str]) -> tuple[str, ...]:
    """Return trailing bullet lines that heuristically resemble footers."""

    _, block, text = record
    if not _starts_list_like(block, text):
        return tuple()
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not lines:
        return tuple()
    suffix = _resolve_footer_suffix(lines)
    bullet_like = tuple(
        line for line in suffix if starts_with_bullet(line) or starts_with_number(line)
    )
    return bullet_like if bullet_like == suffix else tuple()


def _record_is_footer_candidate(record: tuple[int, Block, str]) -> bool:
    return bool(_record_trailing_footer_lines(record))


def _trim_footer_suffix(text: str, suffix: tuple[str, ...]) -> str:
    """Return ``text`` with trailing ``suffix`` bullet lines removed."""

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


def _strip_footer_suffix(record: tuple[int, Block, str]) -> tuple[int, Block, str] | None:
    """Return ``record`` without footer bullets or ``None`` if empty."""

    page, block, text = record
    suffix = _record_trailing_footer_lines(record)
    if not suffix:
        return record
    trimmed = _trim_footer_suffix(text, suffix)
    if trimmed == text:
        return record
    if not trimmed.strip():
        return None
    updated_block: Block = block
    if isinstance(block, Mapping):
        updated = dict(block)
        updated["text"] = trimmed
        updated_block = cast(Block, updated)
    return page, updated_block, trimmed


def _is_footer_artifact_record(
    previous: tuple[int, Block, str],
    current: tuple[int, Block, str],
) -> bool:
    """Return ``True`` when ``current`` resembles a stray footer list."""

    prev_page, prev_block, prev_text = previous
    page, block, text = current
    if page != prev_page:
        return False
    stripped_lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not stripped_lines or len(stripped_lines) > 2:
        return False
    if not all(starts_with_bullet(line) or starts_with_number(line) for line in stripped_lines):
        return False
    word_total = sum(len(line.split()) for line in stripped_lines)
    if word_total > 20:
        return False
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
    return not _starts_list_like(prev_block, prev_text)


def _strip_footer_suffixes(
    records: Iterable[tuple[int, Block, str]],
) -> tuple[tuple[int, Block, str], ...]:
    """Remove footer suffix bullets from ``records``."""

    cleaned: list[tuple[int, Block, str]] = []
    for record in records:
        trimmed = _strip_footer_suffix(record)
        if trimmed is None:
            continue
        if cleaned and _is_footer_artifact_record(cleaned[-1], trimmed):
            continue
        cleaned.append(trimmed)
    return tuple(cleaned)


def _should_emit_list_boundary(previous: tuple[int, Block, str], block: Block, text: str) -> bool:
    _, prev_block, prev_text = previous
    if prev_text.rstrip().endswith(":"):
        return False
    if _starts_list_like(prev_block, prev_text):
        return False
    return not _record_is_footer_candidate(previous)


def _list_tail_split_index(text: str) -> int | None:
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
        if starts_with_bullet(stripped) or starts_with_number(stripped):
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


def _split_list_record(record: tuple[int, Block, str]) -> tuple[tuple[int, Block, str], ...]:
    page, block, text = record
    split_at = _list_tail_split_index(text)
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
        if _colon_bullet_boundary(prev_text, record[1], record[2]):
            prefix = previous[:-1]
            colon_record = previous[-1]
            head = acc[:-1]
            head = (*head, prefix) if prefix else head
            return (*head, (colon_record, record))
        if _record_is_list_like(previous[-1]) and not _record_is_list_like(record):
            return (*acc, (record,))
        return (*acc[:-1], previous + (record,))

    return reduce(_append_segment, buffer, tuple())


def _expand_segment_records(
    segment: tuple[tuple[int, Block, str], ...],
) -> tuple[tuple[int, Block, str], ...]:
    expanded = tuple(chain.from_iterable(_split_list_record(record) for record in segment))
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


def _collapse_numbered_list_spacing(text: str) -> str:
    """Collapse blank lines between numbered items while preserving others."""

    if "\n\n" not in text:
        return text
    lines = tuple(text.splitlines())
    if len(lines) <= 2:
        return text

    def _is_numbered(line: str) -> bool:
        stripped = line.lstrip()
        return bool(stripped) and starts_with_number(stripped)

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


def _normalize_numbered_list_text(text: str) -> str:
    """Normalize numbered list spacing without altering other content."""

    return _collapse_numbered_list_spacing(text) if text else text


def _join_record_texts(records: Iterable[tuple[int, Block, str]]) -> str:
    joined = "\n\n".join(part.strip() for _, _, part in records if part.strip()).strip()
    return _normalize_numbered_list_text(joined) if joined else joined


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


def _segment_totals(segment: tuple[tuple[int, Block, str], ...]) -> tuple[int, int, int]:
    totals = tuple(_effective_counts(text) for _, _, text in segment)
    words = sum(count for count, _, _ in totals)
    dense = sum(count for _, count, _ in totals)
    return words, dense, max(words, dense)


def _resolved_limit(options: SplitOptions | None, limit: int | None) -> int | None:
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


def _hard_limit(options: SplitOptions | None, resolved_limit: int | None) -> int | None:
    if options is not None and options.chunk_size > 0:
        return options.chunk_size
    return resolved_limit


def _overlap_value(options: SplitOptions | None) -> int:
    return options.overlap if options is not None else 0


def _emit_buffer_segments(
    buffer: tuple[tuple[int, Block, str], ...],
    *,
    start_index: int,
    overlap: int,
    resolved_limit: int | None,
    hard_limit: int | None,
    overflow: bool,
) -> tuple[tuple[int, Block, str], ...]:
    if not buffer:
        return tuple()
    segments = _split_colon_bullet_segments(buffer) or (buffer,)
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
        )
    )


def _merged_segment_record(
    segment: tuple[tuple[int, Block, str], ...],
    *,
    start_index: int,
    overlap: int,
) -> tuple[int, Block, str] | None:
    trimmed = _apply_overlap_within_segment(segment, overlap)
    if not trimmed:
        return None
    joined = _join_record_texts(trimmed)
    if not joined or len(joined) > SOFT_LIMIT:
        return None
    merged = _merge_record_block(list(trimmed), joined)
    first_page = trimmed[0][0]
    return first_page, _with_chunk_index(merged, start_index), joined


def _emit_individual_records(
    segment: tuple[tuple[int, Block, str], ...], start_index: int
) -> tuple[tuple[int, Block, str], ...]:
    return tuple(
        (
            page,
            _with_chunk_index(block, start_index + offset),
            _normalize_numbered_list_text(text),
        )
        for offset, (page, block, text) in enumerate(segment)
    )


def _emit_segment_records(
    segment: tuple[tuple[int, Block, str], ...],
    *,
    start_index: int,
    overlap: int,
    resolved_limit: int | None,
    hard_limit: int | None,
    overflow: bool,
) -> tuple[tuple[int, Block, str], ...]:
    """Emit ``segment`` as merged or individual records respecting limits."""
    if not segment:
        return tuple()
    segment = _expand_segment_records(segment)
    words, dense, effective = _segment_totals(segment)
    exceeds_soft = resolved_limit is not None and effective > resolved_limit
    exceeds_hard = hard_limit is not None and effective > hard_limit
    overflow_active = overflow and (exceeds_soft or exceeds_hard)
    if overflow_active or len(segment) == 1:
        return _emit_individual_records(segment, start_index)
    if exceeds_soft or exceeds_hard:
        return _emit_individual_records(segment, start_index)
    merged = _merged_segment_record(
        segment,
        start_index=start_index,
        overlap=overlap,
    )
    if merged is None:
        return _emit_individual_records(segment, start_index)
    return (merged,)


def _maybe_merge_dense_page(
    records: Iterable[tuple[int, Block, str]],
    options: SplitOptions | None,
    limit: int | None,
) -> tuple[tuple[int, Block, str], ...]:
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
    merged_text = _join_record_texts(sequence)
    if not merged_text:
        return sequence
    merged_block = _merge_record_block(list(sequence), merged_text)
    return ((sequence[0][0], merged_block, merged_text),)


@dataclass(frozen=True)
class _CollapseEmitter:
    resolved_limit: int | None
    hard_limit: int | None
    overlap: int
    buffer: tuple[Record, ...] = tuple()
    running_words: int = 0
    running_dense: int = 0
    start_index: int | None = None
    outputs: tuple[Record, ...] = tuple()

    def append(
        self,
        idx: int,
        record: Record,
        counts: tuple[int, int, int],
    ) -> _CollapseEmitter:
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

    def flush(self, *, overflow: bool = False) -> _CollapseEmitter:
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
        )
        return replace(
            self,
            buffer=tuple(),
            running_words=0,
            running_dense=0,
            start_index=None,
            outputs=self.outputs + produced,
        )

    def emit_single(self, idx: int, record: Record) -> _CollapseEmitter:
        page, block, text = record
        entry: Record = (page, _with_chunk_index(block, idx), text)
        return replace(self, outputs=self.outputs + (entry,))


def _page_or_footer_boundary(buffer: tuple[Record, ...], record: Record) -> bool:
    if not buffer:
        return False
    prev_page, _, _ = buffer[-1]
    page, _, _ = record
    if prev_page != page:
        return True
    prev_is_footer = _record_is_footer_candidate(buffer[-1])
    current_is_footer = _record_is_footer_candidate(record)
    return (current_is_footer != prev_is_footer) and (current_is_footer or prev_is_footer)


def _projected_overflow(
    state: _CollapseEmitter,
    block: Block,
    text: str,
    counts: tuple[int, int, int],
) -> str | None:
    words = state.running_words + counts[0]
    dense = state.running_dense + counts[1]
    effective = max(words, dense)
    exceeds_soft = state.resolved_limit is not None and effective > state.resolved_limit
    exceeds_hard = state.hard_limit is not None and effective > state.hard_limit
    if exceeds_hard:
        last_text = state.buffer[-1][2].rstrip()
        if _ENDS_SENTENCE.search(last_text) and not _starts_list_like(block, text):
            return "flush"
        return "overflow"
    if exceeds_soft:
        return "overflow"
    return None


def _collapse_step(state: _CollapseEmitter, item: tuple[int, Record]) -> _CollapseEmitter:
    idx, record = item
    _, block, text = record
    counts = _effective_counts(text)
    _, _, effective = counts
    if _page_or_footer_boundary(state.buffer, record):
        state = state.flush()
    if (state.resolved_limit is not None and effective > state.resolved_limit) or (
        state.hard_limit is not None and effective > state.hard_limit
    ):
        return state.flush().emit_single(idx, record)
    if (
        state.buffer
        and _starts_list_like(block, text)
        and _should_emit_list_boundary(state.buffer[-1], block, text)
    ):
        state = state.flush()
    if state.buffer:
        overflow_action = _projected_overflow(state, block, text, counts)
        if overflow_action is not None:
            state = state.flush(overflow=overflow_action == "overflow")
    return state.append(idx, record, counts)


def _collapse_records(
    records: Iterable[tuple[int, Block, str]],
    options: SplitOptions | None = None,
    limit: int | None = None,
) -> Iterator[tuple[int, Block, str]]:
    seq = tuple(_strip_footer_suffixes(tuple(records)))
    if not seq:
        return
    resolved_limit = _resolved_limit(options, limit)
    if resolved_limit is None:
        yield from (
            (page, _with_chunk_index(block, idx), text)
            for idx, (page, block, text) in enumerate(seq)
        )
        return

    hard_limit = _hard_limit(options, resolved_limit)
    overlap = _overlap_value(options)
    initial = _CollapseEmitter(
        resolved_limit=resolved_limit,
        hard_limit=hard_limit,
        overlap=overlap,
    )
    final_state = reduce(_collapse_step, enumerate(seq), initial).flush()
    merged_outputs = _maybe_merge_dense_page(final_state.outputs, options, limit)
    yield from (
        (page, _with_chunk_index(block, idx), text)
        for idx, (page, block, text) in enumerate(merged_outputs)
    )


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


def _starts_list_like(block: Block, text: str) -> bool:
    kind = _block_list_kind(block)
    if kind:
        return True
    if block.get("type") == "list_item":
        return True
    stripped = text.lstrip()
    return bool(stripped) and (starts_with_bullet(stripped) or starts_with_number(stripped))


def _should_break_after_colon(prev_text: str, block: Block, text: str) -> bool:
    if not prev_text.rstrip().endswith(":"):
        return False
    lead = text.lstrip()
    if not lead:
        return False
    head = lead[0]
    return head.isupper() or head.isdigit() or _starts_list_like(block, text)


def _merge_blocks(
    acc: list[tuple[int, Block, str]],
    cur: tuple[int, Block, str],
) -> list[tuple[int, Block, str]]:
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
    if _starts_list_like(block, text):
        return acc + [cur]
    if _should_break_after_colon(prev_text, block, text):
        return acc + [cur]
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


def _block_texts(doc: Doc, split_fn: SplitFn) -> Iterator[tuple[int, Block, str]]:
    """Yield ``(page, block, text)`` triples after merging sentence fragments."""

    merged = pipeline_merge_adjacent_blocks(
        pipeline_iter_blocks(doc),
        text_of=_block_text,
        fold=_merge_blocks,
        split_fn=split_fn,
    )
    return _split_inline_heading_records(merged)


def _is_heading(block: Block) -> bool:
    """Return ``True`` when ``block`` represents a heading."""

    return block.get("type") == "heading"


def _chunk_items(
    doc: Doc,
    split_fn: SplitFn,
    generate_metadata: bool = True,
    *,
    options: SplitOptions | None = None,
) -> Iterator[Chunk]:
    """Yield chunk records from ``doc`` using ``split_fn``."""

    filename = doc.get("source_path")
    limit = options.compute_limit() if options is not None else None
    stages: tuple[Callable[[Iterable[Record]], Iterable[Record]], ...] = (
        partial(
            pipeline_attach_headings,
            is_heading=_is_heading,
            merge_block_text=_merge_heading_texts,
        ),
        _merge_styled_list_records,
        partial(_stitch_block_continuations, limit=limit),
    )
    records: Iterable[Record] = _block_texts(doc, split_fn)
    for stage in stages:
        records = stage(records)
    collapsed = _collapse_records(records, options, limit)
    builder = partial(build_chunk_with_meta, filename=filename)
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
        chunk_records = _chunk_items(
            doc,
            split_fn,
            self.generate_metadata,
            options=options,
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
    base.__post_init__()
    return base


split_semantic: Pass = register(make_splitter())

# fmt: on
