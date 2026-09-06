from __future__ import annotations

import pdf_chunker.fallbacks as mod
from pdf_chunker.fallbacks import _assess_text_quality, apply_fallbacks
from pdf_chunker.pdf_blocks import Block


def test_quality_metrics_distinguish_good_empty_and_artifact_heavy_text() -> None:
    good = _assess_text_quality(
        "A coherent paragraph contains ordinary words and enough spacing to "
        "represent readable document text."
    )
    empty = _assess_text_quality("\n  \t")
    artifact_heavy = _assess_text_quality("HEADER\nHEADER\nHEADER\n" + "�" * 40)

    assert good["quality_score"] > 0.7
    assert empty["quality_score"] == 0.0
    assert artifact_heavy["quality_score"] < good["quality_score"]
    assert artifact_heavy["repeated_line_ratio"] > 0.0
    assert artifact_heavy["replacement_glyph_ratio"] > 0.0


def test_block_quality_reports_page_coverage_and_missing_pages() -> None:
    report: dict[str, object] = {}
    primary = [Block(text="Readable text.", source={"page": 1})]

    apply_fallbacks(
        primary,
        "dummy.pdf",
        set(),
        expected_pages={1, 2},
        quality_report=report,
    )

    assert report["page_coverage"] == 0.5
    assert report["missing_pages"] == [2]


def test_degraded_primary_selects_better_fallback_and_records_candidates(monkeypatch) -> None:
    primary = [
        Block(text="x" * 240, source={"page": 1}),
        Block(text="x" * 240, source={"page": 2}),
    ]
    fallback = [
        {"text": "A readable page contains normal words and spacing.", "source": {"page": 1}},
        {"text": "Another readable page preserves the document content.", "source": {"page": 2}},
    ]

    monkeypatch.setattr(mod, "_extract_with_pdftotext", lambda *_: fallback)
    monkeypatch.setattr(mod, "_extract_with_pdfminer", lambda *_: [])
    report: dict[str, object] = {}

    result = list(
        apply_fallbacks(
            primary,
            "dummy.pdf",
            set(),
            expected_pages={1, 2},
            quality_report=report,
        )
    )

    assert [block.text for block in result] == [item["text"] for item in fallback]
    assert report["selected_engine"] == "pdftotext"
    assert report["reason"] == "primary_quality_below_threshold"
    assert report["candidate_scores"]["pdftotext"] > report["candidate_scores"]["pymupdf"]


def test_good_primary_skips_fallback_and_records_reason(monkeypatch) -> None:
    primary = [
        Block(
            text="A successful primary extraction has readable words and spacing.",
            source={"page": 1},
        )
    ]

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fallback should not be called for good primary text")

    monkeypatch.setattr("pdf_chunker.fallbacks._extract_with_pdftotext", fail_if_called)
    monkeypatch.setattr("pdf_chunker.fallbacks._extract_with_pdfminer", fail_if_called)
    report: dict[str, object] = {}

    result = list(
        apply_fallbacks(
            primary,
            "dummy.pdf",
            set(),
            expected_pages={1},
            quality_report=report,
        )
    )

    assert result == primary
    assert report["selected_engine"] == "pymupdf"
    assert report["reason"] == "primary_quality_acceptable"
