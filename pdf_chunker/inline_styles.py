from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from operator import attrgetter
from collections.abc import Mapping as MappingABC
from typing import Any, Callable, Iterable, Mapping, Sequence


BoundsTransform = Callable[[int, int], tuple[int, int] | None]


@dataclass(frozen=True)
class InlineStyleSpan:
    """Normalized inline style span aligned to ``Block.text`` indices."""

    start: int
    end: int
    style: str
    confidence: float | None = None
    attrs: Mapping[str, str] | None = None


def clamp_spans(
    spans: Iterable[InlineStyleSpan],
    text_length: int,
) -> tuple[InlineStyleSpan, ...]:
    """Clamp spans to ``[0, text_length]`` and drop zero-length results."""

    return tuple(
        clamped
        for clamped in (
            _clamp_single_span(span, text_length) for span in spans
        )
        if clamped is not None
    )


def _clamp_single_span(
    span: InlineStyleSpan, text_length: int
) -> InlineStyleSpan | None:
    start = min(max(span.start, 0), text_length)
    end = min(max(span.end, 0), text_length)
    if end <= start:
        return None
    if start == span.start and end == span.end:
        return span
    return replace(span, start=start, end=end)


def merge_adjacent_spans(
    spans: Sequence[InlineStyleSpan],
) -> tuple[InlineStyleSpan, ...]:
    """Merge adjacent spans with identical metadata."""

    merged: list[InlineStyleSpan] = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        prev = merged[-1]
        if _spans_share_identity(prev, span) and prev.end == span.start:
            merged[-1] = replace(prev, end=span.end)
        else:
            merged.append(span)
    return tuple(merged)


def sort_and_deduplicate(
    spans: Iterable[InlineStyleSpan],
) -> tuple[InlineStyleSpan, ...]:
    """Sort spans and drop overlaps by retaining the earliest occurrence."""

    sorted_spans = sorted(spans, key=attrgetter("start", "end", "style"))
    deduped: list[InlineStyleSpan] = []
    for span in sorted_spans:
        if deduped and span.start < deduped[-1].end:
            continue
        deduped.append(span)
    return tuple(deduped)


def remap_spans(
    spans: Iterable[InlineStyleSpan],
    transform: BoundsTransform | None,
) -> tuple[InlineStyleSpan, ...]:
    """Remap span bounds through ``transform`` and drop invalid results."""

    if transform is None:
        return tuple(spans)
    return tuple(
        remapped
        for remapped in (
            _remap_single_span(span, transform) for span in spans
        )
        if remapped is not None
    )


def _remap_single_span(
    span: InlineStyleSpan, transform: BoundsTransform
) -> InlineStyleSpan | None:
    mapped = transform(span.start, span.end)
    if mapped is None:
        return None
    new_start, new_end = mapped
    if new_end <= new_start:
        return None
    if new_start == span.start and new_end == span.end:
        return span
    return replace(span, start=new_start, end=new_end)


def build_index_remapper(
    index_map: Sequence[int | None],
) -> BoundsTransform:
    """Create a ``BoundsTransform`` from an index remapping array."""

    def transform(start: int, end: int) -> tuple[int, int] | None:
        try:
            new_start = index_map[start]
            new_end = index_map[end]
        except IndexError:
            return None
        if new_start is None or new_end is None:
            return None
        return new_start, new_end

    return transform


def _coerce_attrs(attrs: Any) -> Mapping[str, str] | None:
    if not isinstance(attrs, MappingABC):
        return None
    return {str(k): str(v) for k, v in attrs.items()}


def _coerce_span(span: Any) -> InlineStyleSpan | None:
    if isinstance(span, InlineStyleSpan):
        return span
    if not isinstance(span, MappingABC):
        return None

    start = span.get("start")
    end = span.get("end")
    style = span.get("style")
    if start is None or end is None or style is None:
        return None

    try:
        start_idx = int(start)
        end_idx = int(end)
    except (TypeError, ValueError):
        return None

    confidence = span.get("confidence")
    try:
        confidence_val = (
            float(confidence)
            if confidence is not None and confidence != ""
            else None
        )
    except (TypeError, ValueError):
        confidence_val = None

    return InlineStyleSpan(
        start=start_idx,
        end=end_idx,
        style=str(style),
        confidence=confidence_val,
        attrs=_coerce_attrs(span.get("attrs")),
    )


def normalize_spans(
    spans: Iterable[InlineStyleSpan | Mapping[str, Any] | Any],
    text_length: int,
    transform: BoundsTransform | None = None,
) -> tuple[InlineStyleSpan, ...]:
    """Apply remapping, clamping, sorting, and merging to spans."""

    materialized = tuple(filter(None, (_coerce_span(span) for span in spans)))
    if not materialized:
        return ()

    remapped = remap_spans(materialized, transform)
    clamped = clamp_spans(remapped, text_length)
    ordered = sort_and_deduplicate(clamped)
    return merge_adjacent_spans(ordered)


def _spans_share_identity(
    left: InlineStyleSpan, right: InlineStyleSpan
) -> bool:
    return (
        left.style == right.style
        and left.confidence == right.confidence
        and left.attrs == right.attrs
    )


def build_index_map(source: str, target: str) -> list[int | None]:
    """Return an index remapping from ``source`` to ``target`` text."""

    matcher = SequenceMatcher(a=source, b=target)
    mapping: list[int | None] = [None] * (len(source) + 1)
    for a_start, b_start, size in matcher.get_matching_blocks():
        for offset in range(size + 1):
            mapping[a_start + offset] = b_start + offset

    last = 0
    for idx, value in enumerate(mapping):
        if value is None:
            mapping[idx] = last
        else:
            last = value

    mapping[-1] = len(target)
    return mapping


def merge_inline_styles(
    first_styles: Sequence[InlineStyleSpan] | None,
    second_styles: Sequence[InlineStyleSpan] | None,
    first_text_len: int,
    separator_len: int,
) -> list[InlineStyleSpan] | None:
    """Merge inline styles from two concatenated text blocks.

    When text A is concatenated with text B via a separator, this function
    adjusts spans from block B to account for the offset and combines them
    with spans from block A.

    Args:
        first_styles: Inline styles from the first block (or None)
        second_styles: Inline styles from the second block (or None)
        first_text_len: Length of the first block's text
        separator_len: Length of the separator (e.g., 1 for " ", 1 for "\\n")

    Returns:
        Combined list of inline styles, or None if both inputs are None
    """
    if not first_styles and not second_styles:
        return None

    result: list[InlineStyleSpan] = []

    # Add first block's styles unchanged
    if first_styles:
        result.extend(first_styles)

    # Add second block's styles with offset adjustment
    if second_styles:
        offset = first_text_len + separator_len
        for span in second_styles:
            result.append(
                replace(span, start=span.start + offset, end=span.end + offset)
            )

    return result if result else None


__all__ = [
    "InlineStyleSpan",
    "BoundsTransform",
    "build_index_map",
    "build_index_remapper",
    "clamp_spans",
    "merge_adjacent_spans",
    "merge_inline_styles",
    "normalize_spans",
    "remap_spans",
    "sort_and_deduplicate",
]
