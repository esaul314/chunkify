from pdf_chunker.framework import Artifact
from pdf_chunker.passes.ai_enrich import ai_enrich


def _dummy_completion(_: str) -> str:
    return "Classification: question\nTags: [Technical]"


def test_ai_enrich_pass_adds_tags():
    chunks = [{"text": "What is AI?"}]
    meta = {
        "ai_enrich": {
            "completion_fn": _dummy_completion,
            "tag_configs": {"generic": ["technical"]},
        }
    }
    result = ai_enrich(Artifact(payload=chunks, meta=meta))
    enriched = result.payload[0]
    assert enriched["utterance_type"] == "question"
    assert enriched["tags"] == ["technical"]
    assert enriched["metadata"]["tags"] == ["technical"]
    assert result.meta["metrics"]["ai_enrich"]["chunks"] == 1
