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
from pdf_chunker.inline_styles import InlineStyleSpan
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
from pdf_chunker.text_cleaning import STOPWORDS
from pdf_chunker.utils import _build_metadata

logger = logging.getLogger(__name__)

_STOPWORD_TITLES = frozenset(word.title() for word in STOPWORDS)
_FOOTNOTE_TAILS = {"", ".", ",", ";", ":"}
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
    r"(?:\d+(?:[-–—.]\d+)*[a-z]?|[ivxlcdm]+(?:[-–—.][ivxlcdm]+)*[a-z]?)",
    re.IGNORECASE,
)
_CAPTION_FLAG = "_caption_attached"
_TOKEN_PATTERN = re.compile(r"\S+")
_HEADING_STYLE_FLAVORS = frozenset({"bold", "italic", "small_caps", "caps", "uppercase"})
_STYLED_LIST_KIND = "styled"

# fmt: off


def _span_attr(span: Any, name: str, default: Any = None) -> Any:
    if isinstance(span, Mapping):
        return span.get(name, default)
    return getattr(span, name, default)


def _span_bounds(
    span: Any, limit: int
) -> tuple[int, int] | None:
    try:
        start_raw = _span_attr(span, "start")
        end_raw = _span_attr(span, "end", start_raw)
        if start_raw is None or end_raw is None:
            return None
        start = max(0, min(limit, int(start_raw)))
        end = max(start, min(limit, int(end_raw)))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def _span_style(span: Any) -> str:
    style = _span_attr(span, "style", "")
    return str(style or "")


def _span_attrs(span: Any) -> Mapping[str, Any] | None:
    attrs = _span_attr(span, "attrs")
    return attrs if isinstance(attrs, Mapping) else None


def _inline_list_kinds(block: Mapping[str, Any]) -> tuple[str, ...]:
    styles = tuple(block.get("inline_styles") or ())
    return tuple(
        attrs["list_kind"]
        for attrs in (_span_attrs(span) for span in styles)
        if isinstance(attrs, Mapping)
        and isinstance(attrs.get("list_kind"), str)
        and attrs["list_kind"]
    )


def _block_list_kind(block: Block) -> str | None:
    if not isinstance(block, Mapping):
        return None
    declared = block.get("list_kind")
    if isinstance(declared, str) and declared:
        return declared
    inline = _inline_list_kinds(block)
    return next(iter(inline), None)


def _warn_stitching_issue(message: str, *, page: int | None = None) -> None:
    if not message:
        return
    detail = f"{message} (page={page})" if page is not None else message
    logger.warning("split_semantic: %s", detail)


def _chunk_meta(chunk: Chunk) -> Mapping[str, Any]:
    meta = chunk.get("meta")
    if isinstance(meta, Mapping):
        return meta
    legacy = chunk.get("metadata")
    return legacy if isinstance(legacy, Mapping) else {}


def _meta_list_kind(meta: Mapping[str, Any] | None) -> str | None:
    if not isinstance(meta, Mapping):
        return None
    kind = meta.get("list_kind")
    return kind if isinstance(kind, str) and kind else None


def _meta_is_list(meta: Mapping[str, Any] | None) -> bool:
    if not isinstance(meta, Mapping):
        return False
    block_type = meta.get("block_type")
    if block_type == "list_item":
        return True
    return block_type in {None, ""} and _meta_list_kind(meta) is not None


def _chunk_is_list(chunk: Chunk) -> bool:
    return _meta_is_list(_chunk_meta(chunk))


def _is_heading_style_span(span: Any) -> bool:
    return _span_style(span).lower() in _HEADING_STYLE_FLAVORS


def _remap_span(
    span: Any, start: int, end: int, limit: int
) -> InlineStyleSpan | dict[str, Any] | None:
    bounded_start = max(0, min(start, limit))
    bounded_end = max(bounded_start, min(end, limit))
    if bounded_end <= bounded_start:
        return None
    if isinstance(span, InlineStyleSpan):
        return replace(span, start=bounded_start, end=bounded_end)
    remapped: dict[str, Any] = {
        "start": bounded_start,
        "end": bounded_end,
        "style": _span_style(span),
    }
    confidence = _span_attr(span, "confidence")
    if confidence is not None:
        remapped["confidence"] = confidence
    attrs = _span_attrs(span)
    if attrs:
        remapped["attrs"] = attrs
    return remapped


def _leading_heading_candidate(text: str, styles: Iterable[Any]) -> tuple[int, int] | None:
    if not text:
        return None
    length = len(text)
    leading_ws = len(text) - len(text.lstrip())
    candidates = [
        bounds
        for span in styles
        if _is_heading_style_span(span)
        and (bounds := _span_bounds(span, length)) is not None
        and bounds[0] <= leading_ws
        and bounds[1] < length
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[1])


def _next_non_whitespace(text: str, index: int) -> str | None:
    for char in text[index:]:
        if not char.isspace():
            return char
    return None


def _trimmed_segment(segment: str) -> tuple[str, int]:
    stripped_lead = segment.lstrip()
    lead_trim = len(segment) - len(stripped_lead)
    trimmed = stripped_lead.rstrip()
    return trimmed, lead_trim


def _heading_styles(
    styles: Iterable[Any],
    text_length: int,
    cutoff: int,
    lead_trim: int,
    heading_limit: int,
) -> tuple[InlineStyleSpan | dict[str, Any], ...]:
    return tuple(
        filter(
            None,
            (
                _remap_span(
                    span,
                    max(bounds[0], 0) - lead_trim,
                    min(bounds[1], cutoff) - lead_trim,
                    heading_limit,
                )
                for span in styles
                if (bounds := _span_bounds(span, text_length)) is not None
                and bounds[0] < cutoff
            ),
        )
    )


def _body_styles(
    styles: Iterable[Any],
    text_length: int,
    offset: int,
    body_limit: int,
) -> tuple[InlineStyleSpan | dict[str, Any], ...]:
    return tuple(
        filter(
            None,
            (
                _remap_span(
                    span,
                    max(bounds[0], offset) - offset,
                    max(bounds[1], offset) - offset,
                    body_limit,
                )
                for span in styles
                if (bounds := _span_bounds(span, text_length)) is not None
                and bounds[1] > offset
            ),
        )
    )


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
    records: Iterable[tuple[int, Block, str]]
) -> Iterator[tuple[int, Block, str]]:
    for page, block, text in records:
        split = _split_inline_heading(block, text)
        if not split:
            yield page, block, text
            continue
        heading_block, body_block = split
        yield page, heading_block, heading_block.get("text", "")
        yield page, body_block, body_block.get("text", "")


def _merge_styled_list_text(first: str, second: str) -> str:
    lead = first.rstrip()
    tail = second.lstrip()
    if not lead:
        return tail
    if not tail:
        return lead
    return f"{lead}\n\n{tail}"


def _normalize_sequence(value: Any) -> tuple[Any, ...]:
    if not value:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _chain_sequences(*values: Any) -> tuple[Any, ...]:
    return tuple(chain.from_iterable(_normalize_sequence(value) for value in values))


def _without_keys(mapping: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    drop = frozenset(keys)
    return {k: v for k, v in mapping.items() if k not in drop}


def _with_optional_tuple(
    mapping: Mapping[str, Any], key: str, values: tuple[Any, ...]
) -> dict[str, Any]:
    if values:
        return {**mapping, key: values}
    return {k: v for k, v in mapping.items() if k != key}


@dataclass(frozen=True)
class _BlockEnvelope:
    block_type: str
    list_kind: str | None = None


def _coalesce_list_kind(blocks: Iterable[Block]) -> str | None:
    kinds = {block.get("list_kind") for block in blocks if block.get("list_kind")}
    return next(iter(kinds)) if len(kinds) == 1 else None


def _resolve_envelope(
    blocks: Iterable[Block], *, default_list_kind: str | None = None
) -> _BlockEnvelope:
    sequence = tuple(blocks)
    block_type = _coalesce_block_type(sequence)
    kind = _coalesce_list_kind(sequence) or default_list_kind
    return _BlockEnvelope(block_type, kind)


def _apply_envelope(
    base: Mapping[str, Any], text: str, envelope: _BlockEnvelope
) -> Block:
    payload = {
        **_without_keys(base, {"text", "list_kind"}),
        "text": text,
        "type": envelope.block_type,
    }
    return {**payload, "list_kind": envelope.list_kind} if envelope.list_kind else payload


def _merge_styled_list_block(primary: Block, secondary: Block) -> Block:
    merged_text = _merge_styled_list_text(
        str(primary.get("text", "")), str(secondary.get("text", ""))
    )
    envelope = _resolve_envelope(
        (primary, secondary), default_list_kind=_STYLED_LIST_KIND
    )
    merged = _apply_envelope(primary, merged_text, envelope)
    inline_styles = _chain_sequences(
        primary.get("inline_styles"), secondary.get("inline_styles")
    )
    source_blocks = _chain_sequences(
        primary.get("source_blocks"), secondary.get("source_blocks")
    )
    without_bbox = _without_keys(merged, {"bbox"})
    with_styles = _with_optional_tuple(without_bbox, "inline_styles", inline_styles)
    return _with_optional_tuple(with_styles, "source_blocks", source_blocks)


def _merge_styled_list_records(
    records: Iterable[tuple[int, Block, str]]
) -> Iterator[tuple[int, Block, str]]:
    pending: tuple[int, Block, str] | None = None
    for page, block, text in records:
        if block.get("list_kind") == _STYLED_LIST_KIND:
            block_copy = dict(block)
            if pending is None:
                pending = (page, block_copy, text)
                continue
            pending_page, pending_block, pending_text = pending
            merged_block = _merge_styled_list_block(pending_block, block_copy)
            merged_text = _merge_styled_list_text(pending_text, text)
            pending = (min(pending_page, page), merged_block, merged_text)
            continue
        if pending is not None:
            yield pending
            pending = None
        yield page, block, text
    if pending is not None:
        yield pending


def _collect_superscripts(
    block: Block, text: str
) -> tuple[list[dict[str, str]], tuple[tuple[int, int], ...]]:
    if not text:
        return [], ()
    limit = len(text)

    def _normalize(span: Any) -> tuple[dict[str, str], tuple[int, int]] | None:
        if _span_style(span) != "superscript":
            return None
        bounds = _span_bounds(span, limit)
        if bounds is None:
            return None
        start, end = bounds
        raw = text[start:end]
        snippet = raw.strip()
        if not snippet or text[end:].strip() not in _FOOTNOTE_TAILS:
            return None
        attrs = _span_attrs(span)
        note_id = attrs.get("note_id") if attrs else None
        focus = raw.find(snippet)
        span_start = start + (focus if focus >= 0 else 0)
        span_end = span_start + len(snippet)
        public = {"text": snippet, **({"note_id": note_id} if note_id else {})}
        return public, (span_start, span_end)

    entries = tuple(
        entry
        for entry in (
            _normalize(span)
            for span in tuple(block.get("inline_styles") or ())
        )
        if entry
    )
    anchors = [public for public, _ in entries]
    spans = tuple(span for _, span in entries if span[0] < span[1])
    return anchors, spans


def _segment_char_limit(chunk_size: int | None) -> int:
    """Return a soft-segmentation character limit for ``chunk_size``."""

    if chunk_size is None or chunk_size <= 0:
        return SOFT_LIMIT
    estimated = int(ceil(chunk_size * _AVERAGE_CHARS_PER_TOKEN))
    return min(SOFT_LIMIT, max(1, estimated))


def _soft_segments(
    text: str, *, max_chars: int | None = None
) -> list[str]:
    """Split ``text`` into segments of at most ``max_chars`` characters."""

    limit = SOFT_LIMIT if max_chars is None or max_chars <= 0 else min(max_chars, SOFT_LIMIT)

    def _split(chunk: str) -> Iterator[str]:
        if len(chunk) <= limit:
            trimmed = chunk.strip()
            if trimmed:
                yield trimmed
            return
        cut = chunk.rfind(" ", 0, limit)
        head = chunk[: cut if cut != -1 else limit].strip()
        tail = chunk[len(head) :].lstrip()
        if head:
            yield head
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
    match = re.search(r"((?:Chapter|Section|Part)\s+[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?)\.?$", candidate)  # noqa: E501
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


def _overlap_window(
    prev_words: tuple[str, ...], current_words: tuple[str, ...], limit: int
) -> int:
    match_limit = min(limit, len(prev_words), len(current_words))
    return next(
        (
            size
            for size in range(match_limit, 0, -1)
            if prev_words[-size:] == current_words[:size]
        ),
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

    if any(
        _covers_entire(style) and _is_heading_style(style) for style in styles
    ):
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


def _coalesce_block_type(blocks: Iterable[Block]) -> str:
    """Return the merged block ``type`` for ``blocks``."""

    types = tuple(
        block.get("type")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type")
    )
    candidates = tuple(t for t in types if t != "heading")
    if not candidates:
        return "paragraph"
    if all(t == "list_item" for t in candidates):
        return "list_item"
    unique = frozenset(candidates)
    if len(unique) == 1:
        return cast(str, candidates[0])
    if "list_item" in unique:
        return "paragraph"
    return cast(str, candidates[0])


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
        line
        for line in suffix
        if starts_with_bullet(line) or starts_with_number(line)
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


def _strip_footer_suffix(
    record: tuple[int, Block, str]
) -> tuple[int, Block, str] | None:
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
    stripped_lines = tuple(
        line.strip() for line in text.splitlines() if line.strip()
    )
    if not stripped_lines or len(stripped_lines) > 2:
        return False
    if not all(
        starts_with_bullet(line) or starts_with_number(line)
        for line in stripped_lines
    ):
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
    records: Iterable[tuple[int, Block, str]]
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


def _should_emit_list_boundary(
    previous: tuple[int, Block, str], block: Block, text: str
) -> bool:
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


def _split_list_record(
    record: tuple[int, Block, str]
) -> tuple[tuple[int, Block, str], ...]:
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
    buffer: Iterable[tuple[int, Block, str]]
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
    segment: tuple[tuple[int, Block, str], ...]
) -> tuple[tuple[int, Block, str], ...]:
    expanded = tuple(chain.from_iterable(_split_list_record(record) for record in segment))
    return expanded if expanded else segment


def _segment_offsets(
    segments: tuple[tuple[tuple[int, Block, str], ...], ...]
) -> tuple[int, ...]:
    if not segments:
        return tuple()
    counts = accumulate(len(segment) for segment in segments[:-1])
    return tuple(chain((0,), counts))


def _enumerate_segments(
    segments: tuple[tuple[tuple[int, Block, str], ...], ...]
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


def _collapse_records(
    records: Iterable[tuple[int, Block, str]],
    options: SplitOptions | None = None,
    limit: int | None = None,
) -> Iterator[tuple[int, Block, str]]:
    seq = list(records)
    seq = list(_strip_footer_suffixes(seq))
    if not seq:
        return
    resolved_limit = _resolved_limit(options, limit)
    if resolved_limit is None:
        for idx, (page, block, text) in enumerate(seq):
            yield page, _with_chunk_index(block, idx), text
        return

    hard_limit = _hard_limit(options, resolved_limit)
    overlap = _overlap_value(options)
    buffer: list[tuple[int, Block, str]] = []
    running_words = 0
    running_dense = 0
    start_index: int | None = None
    outputs: list[tuple[int, Block, str]] = []

    def emit(
        *, overflow: bool = False
    ) -> None:
        nonlocal buffer, running_words, running_dense, start_index
        if not buffer:
            return
        first_index = start_index if start_index is not None else 0
        produced = _emit_buffer_segments(
            tuple(buffer),
            start_index=first_index,
            overlap=overlap,
            resolved_limit=resolved_limit,
            hard_limit=hard_limit,
            overflow=overflow,
        )
        outputs.extend(produced)
        buffer, running_words, running_dense, start_index = [], 0, 0, None

    for idx, record in enumerate(seq):
        page, block, text = record
        is_footer = _record_is_footer_candidate(record)
        if buffer:
            prev_page = buffer[-1][0]
            prev_is_footer = _record_is_footer_candidate(buffer[-1])
            if prev_page != page or (is_footer != prev_is_footer and (is_footer or prev_is_footer)):
                emit()
        word_count, dense_count, effective_count = _effective_counts(text)
        if (resolved_limit is not None and effective_count > resolved_limit) or (
            hard_limit is not None and effective_count > hard_limit
        ):
            emit()
            outputs.append((page, _with_chunk_index(block, idx), text))
            continue
        if (
            buffer
            and _starts_list_like(block, text)
            and _should_emit_list_boundary(buffer[-1], block, text)
        ):
            emit()
        if buffer:
            projected_words = running_words + word_count
            projected_dense = running_dense + dense_count
            projected_effective = max(projected_words, projected_dense)
            exceeds_soft = (
                resolved_limit is not None and projected_effective > resolved_limit
            )
            exceeds_hard = (
                hard_limit is not None and projected_effective > hard_limit
            )
            if exceeds_hard:
                last_text = buffer[-1][2].rstrip()
                if _ENDS_SENTENCE.search(last_text) and not _starts_list_like(block, text):
                    emit()
                else:
                    emit(overflow=True)
            elif exceeds_soft:
                emit(overflow=True)
        if not buffer:
            start_index = idx
            running_words, running_dense = word_count, dense_count
        else:
            running_words += word_count
            running_dense += dense_count
        buffer.append((page, block, text))

    emit()
    merged_outputs = _maybe_merge_dense_page(outputs, options, limit)
    for idx, (page, block, text) in enumerate(merged_outputs):
        yield page, _with_chunk_index(block, idx), text


Doc = dict[str, Any]
Block = dict[str, Any]
Chunk = dict[str, Any]
SplitFn = Callable[[str], list[str]]
MetricFn = Callable[[], dict[str, int | bool]]


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
                splits = _soft_segments(segment, max_chars=char_limit)
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
            raw = _soft_segments(text, max_chars=char_limit)
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


def _looks_like_caption(text: str) -> bool:
    stripped = text.lstrip()
    lower = stripped.lower()
    prefix = next(
        (candidate for candidate in _CAPTION_PREFIXES if lower.startswith(candidate)),
        None,
    )
    if not prefix:
        return False
    remainder = stripped[len(prefix) :].lstrip()
    if not remainder:
        return False
    label_match = _CAPTION_LABEL_RE.match(remainder)
    if not label_match:
        return False
    tail = remainder[label_match.end() :].lstrip()
    if not tail:
        return False
    head = tail[0]
    if head in '.:()"“”':
        return True
    if head in "–—":
        return True
    if head == "-":
        if len(tail) == 1:
            return True
        return not tail[1].isalnum()
    return False


def _contains_caption_line(text: str) -> bool:
    return any(_looks_like_caption(line) for line in text.splitlines())


def _has_caption(block: Block) -> bool:
    return isinstance(block, Mapping) and bool(dict(block).get(_CAPTION_FLAG))


def _mark_caption(block: Block) -> Block:
    if not isinstance(block, Mapping):
        return block
    data = dict(block)
    data[_CAPTION_FLAG] = True
    return data


def _append_caption(prev_text: str, caption: str) -> str:
    head = prev_text.rstrip()
    tail = caption.strip()
    if not head:
        return tail
    return "\n\n".join(filter(None, (head, tail)))


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


def _leading_list_kind(text: str) -> str | None:
    """Return list kind inferred from the first non-empty line of ``text``."""

    lines = (line.lstrip() for line in text.splitlines())
    first = next((line for line in lines if line), "")
    if starts_with_bullet(first):
        return "bullet"
    if starts_with_number(first):
        return "numbered"
    return None


def _infer_list_kind(text: str) -> str | None:
    """Return list kind when any line resembles a bullet or numbered item."""

    if starts_with_bullet(text):
        return "bullet"
    if starts_with_number(text):
        return "numbered"
    lines = tuple(line.lstrip() for line in text.splitlines())
    if any(starts_with_bullet(line) for line in lines):
        return "bullet"
    if any(starts_with_number(line) for line in lines):
        return "numbered"
    return None


def _list_line_ratio(text: str) -> tuple[int, int]:
    """Return count of list-like lines vs total non-empty lines."""

    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not lines:
        return 0, 0
    list_lines = sum(
        1
        for line in lines
        if starts_with_bullet(line) or starts_with_number(line)
    )
    return list_lines, len(lines)


def _tag_list(block: Block) -> Block:
    """Return ``block`` with list metadata inferred when appropriate."""

    text = block.get("text", "")
    block_type = block.get("type")
    existing_kind = block.get("list_kind")

    if block_type == "list_item":
        if existing_kind:
            return block
        inferred = _infer_list_kind(text)
        return {**block, "list_kind": inferred} if inferred else block

    leading_kind = _leading_list_kind(text)
    if not leading_kind:
        inferred = _infer_list_kind(text)
        if not inferred:
            return block
        list_lines, total_lines = _list_line_ratio(text)
        if total_lines and (list_lines * 2) >= total_lines:
            return {**block, "type": "list_item", "list_kind": inferred}
        return block

    return {**block, "type": "list_item", "list_kind": leading_kind}


def _normalize_bullet_tail(tail: str) -> str:
    if not tail:
        return ""
    head, *rest = tail.split(" ", 1)
    normalized = head.lower() if head in _STOPWORD_TITLES else head
    return f"{normalized} {rest[0]}".strip() if rest and rest[0] else normalized


def _normalized_heading_lines(headings: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        stripped
        for heading in headings
        if heading and (stripped := heading.strip())
    )


def _heading_body_separator(heading_block: str) -> str:
    """Return the separator inserted between heading text and body."""

    # LoRA fine-tuning and RAG retrieval benefit from compact, predictable
    # whitespace so heading tokens stay adjacent to their descriptive body
    # text without introducing gratuitous padding. A single newline keeps the
    # visual separation while avoiding the blank line that previously doubled
    # the token cost.
    return "\n"


def _merge_heading_texts(headings: Iterable[str], body: str) -> str:
    normalized_headings = _normalized_heading_lines(headings)
    if any(starts_with_bullet(h.lstrip()) for h in normalized_headings):
        lead = " ".join(h.rstrip() for h in normalized_headings).rstrip()
        tail = _normalize_bullet_tail(body.lstrip()) if body else ""
        return f"{lead} {tail}".strip()

    heading_block = "\n".join(normalized_headings)
    body_text = body.strip() if body else ""

    if not heading_block:
        return body_text
    if not body_text:
        return heading_block

    separator = _heading_body_separator(heading_block)
    return f"{heading_block}{separator}{body_text}"


def _with_source(block: Block, page: int, filename: str | None) -> Block:
    """Attach ``filename`` and ``page`` as a ``source`` entry when absent."""

    existing = block.get("source") or {}
    source = {**{"filename": filename, "page": page}, **existing}
    return {**block, "source": {k: v for k, v in source.items() if v is not None}}


def build_chunk(text: str) -> Chunk:
    """Return chunk payload containing only ``text``."""

    return {"text": text}


def build_chunk_with_meta(
    text: str, block: Block, page: int, filename: str | None, index: int
) -> Chunk:
    """Return chunk payload enriched with metadata."""
    annotated = _tag_list(block)
    start_index = annotated.pop("_chunk_start_index", None)
    chunk_index = start_index if isinstance(start_index, int) else index
    metadata = _build_metadata(
        text,
        _with_source(annotated, page, filename),
        chunk_index,
        {},
    )
    anchors, spans = _collect_superscripts(annotated, text)
    if anchors:
        metadata["footnote_anchors"] = anchors
    chunk = {"text": text, "meta": metadata}
    return {**chunk, "_footnote_spans": spans} if spans else chunk


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
    merged = _stitch_block_continuations(
        _merge_styled_list_records(
            pipeline_attach_headings(
                _block_texts(doc, split_fn),
                is_heading=_is_heading,
                merge_block_text=_merge_heading_texts,
            )
        ),
        limit,
    )
    collapsed = _collapse_records(merged, options, limit)
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
        can_trim = (
            prev_text is not None
            and not current_is_list
            and not _meta_is_list(prev_meta)
        )
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
