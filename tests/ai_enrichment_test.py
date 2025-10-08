import sys
from typing import Any

# Ensure local package path for pdf_chunker imports
sys.path.insert(0, ".")

from pdf_chunker.adapters.ai_enrich import (
    _load_tag_configs,
    _process_chunk_for_file,
)
from pdf_chunker.config import PipelineSpec
from pdf_chunker.framework import Artifact, run_pipeline
from pdf_chunker.passes.ai_enrich import classify_chunk_utterance


def _dummy_completion(_: str) -> str:
    return "Classification: question\nTags: [Technical, unknown]"


def test_load_tag_configs_deduplicates() -> None:
    configs = _load_tag_configs()
    assert all(len(tags) == len({t for t in tags}) for tags in configs.values())


def test_classify_chunk_utterance_filters_invalid_tags() -> None:
    tag_configs = {"generic": ["technical"]}
    result = classify_chunk_utterance(
        "What is AI?", tag_configs=tag_configs, completion_fn=_dummy_completion
    )
    assert result == {"classification": "question", "tags": ["technical"]}


def test_process_chunk_for_file_populates_tags() -> None:
    chunk = {"text": "What is AI?"}
    tag_configs = {"generic": ["technical"]}
    result = _process_chunk_for_file(
        chunk, tag_configs=tag_configs, completion_fn=_dummy_completion
    )
    assert result["tags"] == ["technical"]
    assert result["metadata"]["tags"] == ["technical"]


class _StubClient:
    def classify_chunk_utterance(
        self, text_chunk: str, *, tag_configs: dict[str, list[str]]
    ) -> dict[str, Any]:
        return {"classification": "question", "tags": ["technical"]}


def test_pipeline_enrichment_with_stub() -> None:
    spec = PipelineSpec(pipeline=["ai_enrich"])
    tag_configs = {"generic": ["technical"]}
    artifact = Artifact(
        payload={"type": "chunks", "items": [{"text": "What is AI?"}]},
        meta={
            "ai_enrich": {
                "enabled": True,
                "client": _StubClient(),
                "tag_configs": tag_configs,
            }
        },
    )
    result = run_pipeline(spec.pipeline, artifact)
    items = result.payload["items"]
    assert all(c.get("utterance_type") == "question" for c in items)
    assert all(c.get("tags") == ["technical"] for c in items)
