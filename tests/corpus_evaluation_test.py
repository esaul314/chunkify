from __future__ import annotations

from pathlib import Path

from pdf_chunker.evaluation import evaluate_rows, quality_passed
from scripts.evaluate_corpus import evaluate_manifest, load_manifest


def _row(text: str, chunk_id: str = "chunk") -> dict[str, object]:
    return {
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "source": "fixture.txt",
            "source_file": "fixture.txt",
            "utterance_type": "unclassified",
            "tags": [],
            "language": "en",
            "block_type": "paragraph",
            "readability": {},
            "importance": "medium",
            "page": 1,
            "page_range": "1",
        },
    }


def test_evaluate_rows_reports_overlap_duplicates_and_metadata() -> None:
    rows = [
        _row("Alpha beta gamma", "one"),
        _row("beta gamma delta", "two"),
        _row("beta gamma delta", "three"),
    ]

    metrics = evaluate_rows(
        rows,
        page_count=1,
        sentinels=["Alpha beta"],
        ordered_fragments=["Alpha", "delta"],
    )

    assert metrics["chunks"]["count"] == 3
    assert metrics["chunks"]["duplicate_chunks"] == 1
    assert metrics["chunks"]["overlap"]["total_tokens"] == 5
    assert metrics["metadata"]["completeness"] == 1.0
    assert quality_passed(metrics)


def test_evaluate_rows_exposes_missing_sentinel_and_order() -> None:
    metrics = evaluate_rows(
        [_row("Second phrase")],
        sentinels=["missing phrase"],
        ordered_fragments=["First phrase", "Second phrase"],
    )

    assert not quality_passed(metrics)
    assert not metrics["preservation"]["sentinels"]["passed"]
    assert not metrics["preservation"]["text_order"]["passed"]


def test_manifest_includes_representative_formats_and_review_categories() -> None:
    manifest = Path("docs/corpus_manifest.yaml")
    cases = load_manifest(manifest)
    assert {case["kind"] for case in cases} == {"pdf", "epub", "txt"}
    reviewed = {label for case in cases for label in case.get("reviewed_for", [])}
    assert {"running_headers_footers", "footnotes", "tables", "lists"} <= reviewed


def test_evaluate_manifest_runs_text_fixture(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
version: 1
cases:
  - name: transcript
    path: tests/evaluation/fixtures/sample_transcript.txt
    kind: txt
    spec: pipeline.yaml
    expected_page_count: 1
    sentinels:
      - stable chunk identity
    ordered_fragments:
      - Welcome to the document quality evaluation.
      - The evaluation report records preservation
    """,
        encoding="utf-8",
    )

    report = evaluate_manifest(manifest)

    assert report["summary"] == {
        "cases": 1,
        "passed": 1,
        "failed": 0,
        "fallback_rate": 0.0,
    }
    case = report["cases"][0]
    assert case["metrics"]["metadata"]["completeness"] == 1.0
    assert case["extraction"]["engine"] == "txt"
