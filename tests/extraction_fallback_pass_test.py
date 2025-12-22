from pdf_chunker.framework import Artifact
from pdf_chunker.passes.extraction_fallback import extraction_fallback


def test_extraction_fallback_records_reason_and_score(monkeypatch):
    monkeypatch.setattr(
        "pdf_chunker.fallbacks.execute_fallback_extraction",
        lambda path, fallback_reason=None: [{"text": "hello"}],
    )
    monkeypatch.setattr(
        "pdf_chunker.fallbacks._assess_text_quality",
        lambda text: {"quality_score": 0.5},
    )

    artifact = Artifact(payload={"source_path": "dummy"}, meta={"fallback_reason": "low_quality"})
    result = extraction_fallback(artifact)

    assert artifact.meta == {"fallback_reason": "low_quality"}
    assert result.payload == {
        "type": "blocks",
        "blocks": [{"text": "hello"}],
        "source_path": "dummy",
    }
    metrics = result.meta["metrics"]["extraction_fallback"]
    assert metrics == {"reason": "low_quality", "score": 0.5}
