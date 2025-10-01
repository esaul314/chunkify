"""Inline-style span helpers used by :mod:`pdf_chunker.passes.split_semantic`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from pdf_chunker.inline_styles import InlineStyleSpan

_HEADING_STYLE_FLAVORS = frozenset({"bold", "italic", "small_caps", "caps", "uppercase"})


def _span_attr(span: Any, name: str, default: Any = None) -> Any:
    if isinstance(span, Mapping):
        return span.get(name, default)
    return getattr(span, name, default)


def _span_bounds(span: Any, limit: int) -> tuple[int, int] | None:
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
