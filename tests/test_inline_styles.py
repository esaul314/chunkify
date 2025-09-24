from __future__ import annotations

import pytest

from pdf_chunker.inline_styles import (
    InlineStyleSpan,
    build_index_remapper,
    clamp_spans,
    merge_adjacent_spans,
    normalize_spans,
    remap_spans,
)
from pdf_chunker.pdf_blocks import _structured_block


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


class _FakePage:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def get_text(self, kind: str, clip=None):  # noqa: ANN001 - signature mirrors PyMuPDF
        assert kind == "dict"
        return self._payload


def _block_tuple(raw_text: str) -> tuple[int, int, int, int, str]:
    return (0, 0, 100, 20, raw_text)


def test_structured_block_emits_basic_styles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pdf_chunker.pdf_blocks.clean_text", lambda text: text)

    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Bold ",
                                "flags": 16,
                                "font": "Helvetica-Bold",
                                "origin": (0.0, 100.0),
                                "size": 12.0,
                            },
                            {
                                "text": "Italic",
                                "flags": 2,
                                "font": "Helvetica-Oblique",
                                "origin": (40.0, 100.0),
                                "size": 12.0,
                            },
                        ]
                    }
                ],
            }
        ]
    }

    block = _structured_block(_FakePage(payload), _block_tuple("Bold Italic"), 1, "doc.pdf")
    assert block is not None
    assert block.inline_styles == [
        InlineStyleSpan(start=0, end=5, style="bold", confidence=1.0, attrs=None),
        InlineStyleSpan(start=5, end=11, style="italic", confidence=1.0, attrs=None),
    ]


def test_structured_block_detects_superscript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pdf_chunker.pdf_blocks.clean_text", lambda text: text)

    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Text",
                                "flags": 0,
                                "font": "Times-Roman",
                                "origin": (0.0, 100.0),
                                "size": 12.0,
                            },
                            {
                                "text": "1",
                                "flags": 0,
                                "font": "Times-Roman",
                                "origin": (40.0, 96.0),
                                "size": 6.0,
                            },
                        ]
                    }
                ],
            }
        ]
    }

    block = _structured_block(_FakePage(payload), _block_tuple("Text1"), 1, "doc.pdf")
    assert block is not None
    assert block.inline_styles == [
        InlineStyleSpan(start=4, end=5, style="superscript", confidence=1.0, attrs=None)
    ]


def test_structured_block_remaps_offsets_after_cleaning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pdf_chunker.pdf_blocks.clean_text", lambda text: text.replace("-\n", ""))

    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "co-",
                                "flags": 2,
                                "font": "Times-Italic",
                                "origin": (0.0, 100.0),
                                "size": 12.0,
                            },
                            {
                                "text": "\n",
                                "flags": 0,
                                "font": "Times-Roman",
                                "origin": (0.0, 100.0),
                                "size": 12.0,
                            },
                            {
                                "text": "operate",
                                "flags": 2,
                                "font": "Times-Italic",
                                "origin": (20.0, 100.0),
                                "size": 12.0,
                            },
                        ]
                    }
                ],
            }
        ]
    }

    block = _structured_block(_FakePage(payload), _block_tuple("co-\noperate"), 1, "doc.pdf")
    assert block is not None
    assert block.inline_styles == [
        InlineStyleSpan(start=0, end=9, style="italic", confidence=1.0, attrs=None)
    ]
