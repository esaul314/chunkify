from __future__ import annotations

from pdf_chunker.inline_styles import (
    InlineStyleSpan,
    build_index_remapper,
    clamp_spans,
    merge_adjacent_spans,
    normalize_spans,
    remap_spans,
)


def _span(
    start: int,
    end: int,
    style: str = "bold",
    confidence: float | None = None,
    attrs: dict[str, str] | None = None,
) -> InlineStyleSpan:
    return InlineStyleSpan(
        start=start,
        end=end,
        style=style,
        confidence=confidence,
        attrs=attrs,
    )


def test_clamp_spans_drops_zero_length_results() -> None:
    spans = (
        _span(-2, 2, "italic"),
        _span(3, 10, "bold"),
        _span(8, 12, "bold"),
    )

    clamped = clamp_spans(spans, text_length=8)

    assert clamped == (
        _span(0, 2, "italic"),
        _span(3, 8, "bold"),
    )


def test_merge_adjacent_spans_combines_matching_metadata() -> None:
    spans = (
        _span(0, 2, "bold", confidence=1.0),
        _span(2, 4, "bold", confidence=1.0),
        _span(4, 6, "bold", confidence=0.9),
        _span(6, 7, "italic"),
        _span(7, 9, "italic"),
    )

    merged = merge_adjacent_spans(spans)

    assert merged == (
        _span(0, 4, "bold", confidence=1.0),
        _span(4, 6, "bold", confidence=0.9),
        _span(6, 9, "italic"),
    )


def test_normalize_spans_applies_full_pipeline() -> None:
    spans = (
        _span(5, 7, "bold"),
        _span(0, 2, "bold"),
        _span(2, 5, "bold", attrs={"note_id": "1"}),
        _span(7, 9, "bold"),
        _span(9, 11, "bold"),
        _span(4, 6, "italic"),
    )

    normalized = normalize_spans(spans, text_length=10)

    assert normalized == (
        _span(0, 2, "bold"),
        _span(2, 5, "bold", attrs={"note_id": "1"}),
        _span(5, 10, "bold"),
    )


def test_build_index_remapper_handles_deleted_glyphs() -> None:
    spans = (
        _span(1, 3, "bold"),
        _span(3, 5, "italic"),
    )
    remapper = build_index_remapper([0, 1, 2, 2, 3, 4])

    remapped = remap_spans(spans, remapper)

    assert remapped == (
        _span(1, 2, "bold"),
        _span(2, 4, "italic"),
    )
