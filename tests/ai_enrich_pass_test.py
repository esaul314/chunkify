from pdf_chunker.framework import Artifact
from pdf_chunker.passes.ai_enrich import ai_enrich
from pdf_chunker.utils import _enrich_chunk


class _DummyClient:
    def __init__(self) -> None:
        self.calls = 0

    def classify_chunk_utterance(self, text: str, *, tag_configs: dict) -> dict:
        self.calls += 1
        return {"classification": "question", "tags": ["technical"]}


def test_ai_enrich_pass_disabled():
    client = _DummyClient()
    chunks = [{"text": "What is AI?"}]
    meta = {"ai_enrich": {"client": client, "enabled": False}}
    result = ai_enrich(Artifact(payload=chunks, meta=meta))
    assert result.payload == chunks
    assert client.calls == 0
    assert "ai_enrich" not in (result.meta or {}).get("metrics", {})


def test_ai_enrich_pass_enabled_enriches():
    client = _DummyClient()
    chunks = [{"text": "What is AI?"}]
    meta = {
        "ai_enrich": {
            "client": client,
            "enabled": True,
            "tag_configs": {"generic": ["technical"]},
        }
    }
    result = ai_enrich(Artifact(payload=chunks, meta=meta))
    enriched = result.payload[0]
    assert enriched["utterance_type"] == "question"
    assert enriched["tags"] == ["technical"]
    assert enriched["metadata"]["tags"] == ["technical"]
    assert result.meta["metrics"]["ai_enrich"]["chunks"] == 1
    assert client.calls == 1


def test_enrich_chunk_fallback_returns_error() -> None:
    assert _enrich_chunk("hi", False, None)["classification"] == "error"
