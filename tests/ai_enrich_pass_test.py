import pytest

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


def test_ai_enrich_pass_enabled_requires_client(monkeypatch) -> None:
    from pdf_chunker import adapters

    def _raise_import_error(*_args, **_kwargs):
        raise ImportError("litellm missing")

    monkeypatch.setattr(adapters.ai_enrich, "init_llm", _raise_import_error)
    chunks = [{"text": "What is AI?"}]
    meta = {"ai_enrich": {"enabled": True}}
    with pytest.raises(RuntimeError, match="no completion client"):
        ai_enrich(Artifact(payload=chunks, meta=meta))


def test_ai_enrich_pass_enabled_enriches():
    client = _DummyClient()
    chunks = [{"text": "What is AI?", "meta": {"source": "doc.pdf"}}]
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
    assert enriched["meta"]["utterance_type"] == "question"
    assert enriched["metadata"]["tags"] == ["technical"]
    assert result.meta["metrics"]["ai_enrich"]["chunks"] == 1
    assert client.calls == 1


def test_ai_enrich_pass_reads_nested_options():
    client = _DummyClient()
    chunks = [{"text": "What is AI?"}]
    meta = {
        "options": {
            "ai_enrich": {
                "enabled": True,
                "client": client,
                "tag_configs": {"generic": ["technical"]},
            }
        }
    }
    result = ai_enrich(Artifact(payload=chunks, meta=meta))
    enriched = result.payload[0]
    assert enriched["utterance_type"] == "question"
    assert enriched["meta"]["tags"] == ["technical"]
    assert client.calls == 1


def test_ai_enrich_pass_enriches_chunk_containers() -> None:
    client = _DummyClient()
    payload = {"type": "chunks", "items": [{"text": "What is AI?"}]}
    meta = {
        "options": {
            "ai_enrich": {
                "enabled": True,
                "client": client,
                "tag_configs": {"generic": ["technical"]},
            }
        }
    }
    result = ai_enrich(Artifact(payload=payload, meta=meta))
    items = result.payload["items"]
    assert [item.get("utterance_type") for item in items] == ["question"]
    assert [item.get("tags") for item in items] == [["technical"]]
    assert client.calls == 1


def test_enrich_chunk_fallback_returns_default_classification() -> None:
    fallback = _enrich_chunk("hi", False, None)
    assert fallback == {"classification": "unclassified", "tags": []}
