from pdf_chunker.framework import Artifact
from pdf_chunker.passes.extraction_fallback import extraction_fallback


def test_extraction_fallback_records_metrics(monkeypatch):
    def fake_extract(path: str, reason: str | None):
        return ([{"text": "hello"}], {"reason": reason, "score": 0.5})

    monkeypatch.setattr("pdf_chunker.passes.extraction_fallback._extract", fake_extract)

    artifact = Artifact(payload={"source_path": "dummy"}, meta={"fallback_reason": "low_quality"})
    result = extraction_fallback(artifact)

    assert artifact.meta == {"fallback_reason": "low_quality"}
    metrics = result.meta["metrics"]["extraction_fallback"]
    assert metrics == {"reason": "low_quality", "score": 0.5}
