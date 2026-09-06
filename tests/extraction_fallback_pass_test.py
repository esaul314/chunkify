from pdf_chunker.framework import Artifact
from pdf_chunker.passes.extraction_fallback import extraction_fallback


def test_extraction_fallback_records_reason_and_score(monkeypatch):
    monkeypatch.setattr(
        "pdf_chunker.fallbacks._assess_text_quality",
        lambda text: {"quality_score": 0.4 if "bad" in text else 0.8},
    )

    artifact = Artifact(
        payload={
            "type": "extraction_candidates",
            "source_path": "dummy",
            "primary": [{"text": "bad", "source": {"page": 1}}],
            "candidates": {
                "pdftotext": [
                    {"text": "hello", "source": {"page": 1}},
                    {"text": "world", "source": {"page": 2}},
                ]
            },
            "expected_pages": {1, 2},
        },
        meta={"fallback_reason": "low_quality"},
    )
    result = extraction_fallback(artifact)

    assert artifact.meta == {"fallback_reason": "low_quality"}
    assert result.payload == {
        "type": "blocks",
        "blocks": [
            {"text": "hello", "source": {"page": 1}},
            {"text": "world", "source": {"page": 2}},
        ],
        "source_path": "dummy",
    }
    metrics = result.meta["metrics"]["extraction_fallback"]
    assert metrics["selected_engine"] == "pdftotext"
    assert metrics["reason"] == "primary_quality_below_threshold"
