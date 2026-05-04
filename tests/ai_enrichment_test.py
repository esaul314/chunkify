import sys
from typing import Any
from types import SimpleNamespace

# Ensure local package path for pdf_chunker imports
sys.path.insert(0, ".")

from pdf_chunker.adapters.ai_enrich import (
    DEFAULT_AI_ENRICH_MAX_TOKENS,
    _load_tag_configs,
    _process_chunk_for_file,
    init_fallback_llm,
    init_llm,
)
from pdf_chunker.config import PipelineSpec
from pdf_chunker.framework import Artifact, run_pipeline
from pdf_chunker.passes.ai_enrich import (
    _effective_tag_configs,
    _resolve_completion,
    _resolve_fallback_completion,
    classify_chunk_utterance,
)


def _dummy_completion(_: str) -> str:
    return "Classification: question\nTags: [Technical, unknown]"


def _multiline_completion(_: str) -> str:
    return """Classification: summary\nTags:\n- Technical\n- methodology\n"""


def test_load_tag_configs_deduplicates() -> None:
    configs = _load_tag_configs()
    assert all(len(tags) == len({t for t in tags}) for tags in configs.values())


def test_classify_chunk_utterance_filters_invalid_tags() -> None:
    tag_configs = {"generic": ["technical"]}
    result = classify_chunk_utterance(
        "This technical overview asks what AI is.",
        tag_configs=tag_configs,
        completion_fn=_dummy_completion,
    )
    assert result == {"classification": "question", "tags": ["technical"]}


def test_effective_tag_configs_drops_content_type_and_utterance_type_tags() -> None:
    tag_configs = {
        "content_type": ["explanation", "summary"],
        "concepts": ["architecture", "explanation", "testing"],
    }
    assert _effective_tag_configs(tag_configs) == {
        "concepts": ["architecture", "testing"],
    }


def test_process_chunk_for_file_populates_tags() -> None:
    chunk = {"text": "This technical overview asks what AI is."}
    tag_configs = {"generic": ["technical"]}
    result = _process_chunk_for_file(
        chunk, tag_configs=tag_configs, completion_fn=_dummy_completion
    )
    assert result["tags"] == ["technical"]
    assert result["metadata"]["tags"] == ["technical"]


def test_classify_chunk_utterance_handles_multiline_tags() -> None:
    tag_configs = {"generic": ["technical", "methodology"]}
    result = classify_chunk_utterance(
        "This technical methodology explains how we should build this.",
        tag_configs=tag_configs,
        completion_fn=_multiline_completion,
    )
    assert result == {
        "classification": "summary",
        "tags": ["technical", "methodology"],
    }

def _category_heading_completion(_: str) -> str:
    return """Classification: explanation
Tags:
- technical_level:
- analysis
"""


def test_classify_chunk_utterance_ignores_category_headings_and_invalid_tags() -> None:
    tag_configs = {"generic": ["tool", "analysis"]}
    result = classify_chunk_utterance(
        "This section provides analysis of the system and compares tradeoffs.",
        tag_configs=tag_configs,
        completion_fn=_category_heading_completion,
    )
    assert result == {
        "classification": "explanation",
        "tags": ["analysis"],
    }





def _invalid_classification_completion(_: str) -> str:
    return "Classification: book\nTags: [software_design, technical_debt]"


def test_classify_chunk_utterance_invalid_classification_falls_back_to_unclassified() -> None:
    tag_configs = {"generic": ["software_design", "technical_debt"]}
    result = classify_chunk_utterance(
        "Front matter for a software design book.",
        tag_configs=tag_configs,
        completion_fn=_invalid_classification_completion,
    )
    assert result == {
        "classification": "unclassified",
        "tags": ["software_design"],
    }


def _ungrounded_tag_completion(_: str) -> str:
    return "Classification: explanation\nTags: [technical_debt, software_design]"


def test_classify_chunk_utterance_drops_ungrounded_model_tags() -> None:
    tag_configs = {"generic": ["software_design", "technical_debt"]}
    result = classify_chunk_utterance(
        "This software design book introduces modular architecture.",
        tag_configs=tag_configs,
        completion_fn=_ungrounded_tag_completion,
    )
    assert result == {
        "classification": "explanation",
        "tags": ["software_design"],
    }


def _empty_tags_completion(_: str) -> str:
    return "Classification: explanation\nTags: []"


def test_classify_chunk_utterance_uses_lexical_fallback_for_empty_tags() -> None:
    tag_configs = {"generic": ["software_design", "technical_debt", "tool"]}
    result = classify_chunk_utterance(
        "This software design chapter explains how to pay down technical debt.",
        tag_configs=tag_configs,
        completion_fn=_empty_tags_completion,
    )
    assert result == {
        "classification": "explanation",
        "tags": ["software_design", "technical_debt"],
    }


def test_classify_chunk_utterance_does_not_guess_single_word_fallback_tags() -> None:
    tag_configs = {"concepts": ["algorithms", "testing", "security"]}
    result = classify_chunk_utterance(
        "Algorithms are central to this chapter, and the introduction compares algorithm design tradeoffs.",
        tag_configs=tag_configs,
        completion_fn=_empty_tags_completion,
    )
    assert result == {
        "classification": "explanation",
        "tags": [],
    }


def _fallback_only_completion(_: str) -> str:
    return "Classification: explanation\nTags: [software_design]"


def test_classify_chunk_utterance_uses_fallback_for_empty_tags() -> None:
    tag_configs = {"generic": ["software_design", "technical_debt"]}
    result = classify_chunk_utterance(
        "This software design guide focuses on modular architecture.",
        tag_configs=tag_configs,
        completion_fn=_empty_tags_completion,
        fallback_completion_fn=_fallback_only_completion,
        fallback_min_chars=10,
    )
    assert result == {
        "classification": "explanation",
        "tags": ["software_design"],
    }


def test_classify_chunk_utterance_skips_fallback_for_short_chunks() -> None:
    tag_configs = {"generic": ["software_design"]}
    result = classify_chunk_utterance(
        "short chunk",
        tag_configs=tag_configs,
        completion_fn=_empty_tags_completion,
        fallback_completion_fn=_fallback_only_completion,
        fallback_min_chars=50,
    )
    assert result == {
        "classification": "explanation",
        "tags": [],
    }


def test_classify_chunk_utterance_preserves_local_result_when_fallback_fails() -> None:
    tag_configs = {"generic": ["javascript", "design_patterns"]}
    result = classify_chunk_utterance(
        "This chapter explains JavaScript design patterns for modular frontend architecture.",
        tag_configs=tag_configs,
        completion_fn=lambda _: "Classification: explanation\nTags: [javascript, design_patterns]",
        fallback_completion_fn=lambda _: (_ for _ in ()).throw(RuntimeError("quota")),
        fallback_min_chars=10,
    )
    assert result == {
        "classification": "explanation",
        "tags": ["javascript", "design_patterns"],
    }


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


def test_classify_chunk_utterance_repairs_unclassified_when_model_can_reclassify() -> None:
    calls: list[str] = []

    def completion(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "Classification: unclassified\nTags: [javascript, design_patterns]"
        return "Classification: explanation"

    tag_configs = {"generic": ["javascript", "design_patterns"]}
    result = classify_chunk_utterance(
        "This chapter explains JavaScript design patterns for modular frontend architecture.",
        tag_configs=tag_configs,
        completion_fn=completion,
    )

    assert result == {"classification": "explanation", "tags": ["javascript", "design_patterns"]}
    assert len(calls) == 2


def test_init_llm_can_disable_no_think_for_local_qwen(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Classification: question\nTags: []"))]
        )

    monkeypatch.setenv("OPENAI_API_KEY", "local-test")
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=fake_completion),
    )

    completion = init_llm(
        model="Qwen/Qwen3-14B-AWQ",
        api_base="http://localhost:8000/v1",
        no_think=False,
    )
    completion("hello")

    assert calls
    assert calls[0]["model"] == "Qwen/Qwen3-14B-AWQ"
    assert calls[0]["api_base"] == "http://localhost:8000/v1"
    assert calls[0]["max_tokens"] == DEFAULT_AI_ENRICH_MAX_TOKENS
    assert calls[0]["messages"][0]["content"] == "hello"


def test_init_llm_uses_local_luminatus_defaults(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Classification: question\nTags: []"))]
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PDF_CHUNKER_AI_ENRICH_MODEL", raising=False)
    monkeypatch.delenv("PDF_CHUNKER_AI_ENRICH_API_BASE", raising=False)
    monkeypatch.delenv("PDF_CHUNKER_AI_ENRICH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=fake_completion),
    )

    completion = init_llm()
    completion("hello")

    assert calls
    assert calls[0]["model"] == "qwen3-8b-local"
    assert calls[0]["api_base"] == "http://localhost:8100/v1"
    assert calls[0]["api_key"] == "sk-local"
    assert calls[0]["custom_llm_provider"] == "openai"
    assert calls[0]["max_tokens"] == 256
    assert calls[0]["messages"][0]["content"].startswith("/no_think\n")


def test_init_fallback_llm_uses_openai_defaults(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Classification: question\nTags: []"))]
        )

    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    monkeypatch.delenv("PDF_CHUNKER_AI_ENRICH_FALLBACK_MODEL", raising=False)
    monkeypatch.delenv("PDF_CHUNKER_AI_ENRICH_FALLBACK_API_BASE", raising=False)
    monkeypatch.delenv("PDF_CHUNKER_AI_ENRICH_FALLBACK_API_KEY", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=fake_completion),
    )

    completion = init_fallback_llm()
    assert completion is not None
    completion("hello")

    assert calls
    assert calls[0]["model"] == "gpt-4.1-nano"
    assert calls[0]["api_key"] == "openai-test"
    assert "api_base" not in calls[0]
    assert calls[0]["max_tokens"] == 128


def test_resolve_fallback_completion_passes_fallback_options(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_fallback_llm(*, api_key=None, model=None, api_base=None, max_tokens=None):
        captured["api_key"] = api_key
        captured["model"] = model
        captured["api_base"] = api_base
        captured["max_tokens"] = max_tokens
        return lambda _: "Classification: question\nTags: []"

    import pdf_chunker.adapters.ai_enrich as adapter

    monkeypatch.setattr(adapter, "init_fallback_llm", fake_init_fallback_llm)
    completion, reason = _resolve_fallback_completion(
        {
            "fallback_api_key": "k",
            "fallback_model": "gpt-4.1-nano",
            "fallback_api_base": None,
            "fallback_max_tokens": 80,
        }
    )

    assert callable(completion)
    assert reason == "init_fallback_llm"
    assert captured == {
        "api_key": "k",
        "model": "gpt-4.1-nano",
        "api_base": None,
        "max_tokens": 80,
    }


def test_resolve_completion_passes_model_and_api_base_options(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_llm(*, api_key=None, model=None, api_base=None, max_tokens=None, no_think=None):
        captured["api_key"] = api_key
        captured["model"] = model
        captured["api_base"] = api_base
        captured["max_tokens"] = max_tokens
        captured["no_think"] = no_think
        return lambda _: "Classification: question\nTags: []"

    import pdf_chunker.adapters.ai_enrich as adapter

    monkeypatch.setattr(adapter, "init_llm", fake_init_llm)
    completion, reason = _resolve_completion(
        {
            "api_key": "k",
            "model": "qwen3-8b-local",
            "api_base": "http://localhost:8100/v1",
            "max_tokens": 300,
            "no_think": False,
        }
    )

    assert callable(completion)
    assert reason == "init_llm"
    assert captured == {
        "api_key": "k",
        "model": "qwen3-8b-local",
        "api_base": "http://localhost:8100/v1",
        "max_tokens": 300,
        "no_think": False,
    }
