from __future__ import annotations

from dataclasses import dataclass, replace
from operator import attrgetter
from typing import Callable, Iterable, Mapping, Sequence


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


def normalize_spans(
    spans: Iterable[InlineStyleSpan],
    text_length: int,
    transform: BoundsTransform | None = None,
) -> tuple[InlineStyleSpan, ...]:
    """Apply remapping, clamping, sorting, and merging to spans."""

    remapped = remap_spans(spans, transform)
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


__all__ = [
    "InlineStyleSpan",
    "BoundsTransform",
    "build_index_remapper",
    "clamp_spans",
    "merge_adjacent_spans",
    "normalize_spans",
    "remap_spans",
    "sort_and_deduplicate",
]
