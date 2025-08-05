import sys

# Ensure local package path for pdf_chunker imports
sys.path.insert(0, ".")

from pdf_chunker.ai_enrichment import (
    _load_tag_configs,
    classify_chunk_utterance,
    _process_chunk_for_file,
)


def _dummy_completion(_: str) -> str:
    return "Classification: question\nTags: [Technical, unknown]"


def test_load_tag_configs_deduplicates():
    configs = _load_tag_configs()
    assert all(len(tags) == len({t for t in tags}) for tags in configs.values())


def test_classify_chunk_utterance_filters_invalid_tags():
    tag_configs = {"generic": ["technical"]}
    result = classify_chunk_utterance(
        "What is AI?", tag_configs=tag_configs, completion_fn=_dummy_completion
    )
    assert result == {"classification": "question", "tags": ["technical"]}


def test_process_chunk_for_file_populates_tags():
    chunk = {"text": "What is AI?"}
    tag_configs = {"generic": ["technical"]}
    result = _process_chunk_for_file(
        chunk, tag_configs=tag_configs, completion_fn=_dummy_completion
    )
    assert result["tags"] == ["technical"]
    assert result["metadata"]["tags"] == ["technical"]
