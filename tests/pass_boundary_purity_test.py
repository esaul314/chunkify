from __future__ import annotations

import pytest

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _run_passes, assemble_report
from pdf_chunker.framework import Artifact
from pdf_chunker.interactive import (
    make_learned_footer_callback,
    make_learned_list_continuation_callback,
)
from pdf_chunker.learned_patterns import LearnedPattern, LearnedPatterns
from pdf_chunker.passes.ai_enrich import ai_enrich
from pdf_chunker.passes.emit_jsonl import emit_jsonl
from pdf_chunker.passes.extraction_fallback import extraction_fallback
from pdf_chunker.passes.split_semantic import _get_split_fn, split_semantic


class _Client:
    def classify_chunk_utterance(self, text: str, *, tag_configs: dict) -> dict:
        return {"classification": "question", "tags": []}


def test_enrichment_pass_uses_only_injected_client(monkeypatch) -> None:
    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("pass must not construct an enrichment adapter")

    monkeypatch.setattr("pdf_chunker.adapters.ai_enrich.prepare_options", fail_if_constructed)
    result = ai_enrich(
        Artifact(
            payload=[{"text": "What is a boundary?"}],
            meta={
                "ai_enrich": {
                    "enabled": True,
                    "client": _Client(),
                    "tag_configs": {"generic": ["boundary"]},
                }
            },
        )
    )
    assert result.payload[0]["utterance_type"] == "question"


def test_runner_injects_enrichment_adapter_at_boundary() -> None:
    spec = PipelineSpec(
        pipeline=["ai_enrich"],
        options={
            "ai_enrich": {
                "enabled": True,
                "completion_fn": lambda _prompt: "Classification: question\nTags: []",
                "tag_configs": {"generic": ["boundary"]},
            }
        },
    )
    artifact, _ = _run_passes(
        spec,
        Artifact(
            payload={"type": "chunks", "items": [{"text": "What is a boundary?"}]},
            meta={"input": "memory", "metrics": {}},
        ),
    )
    assert artifact.payload["items"][0]["utterance_type"] == "question"


def test_extraction_fallback_pass_consumes_supplied_candidates(monkeypatch) -> None:
    def fail_if_extracted(*_args, **_kwargs):
        raise AssertionError("pass must not execute extraction")

    monkeypatch.setattr("pdf_chunker.fallbacks.execute_fallback_extraction", fail_if_extracted)
    result = extraction_fallback(
        Artifact(
            payload={
                "type": "extraction_candidates",
                "primary": [{"text": "primary", "source": {"page": 1}}],
                "candidates": {},
            }
        )
    )
    assert result.payload["blocks"][0]["text"] == "primary"


def test_learned_callbacks_use_and_record_supplied_state() -> None:
    learned = LearnedPatterns(
        [LearnedPattern(kind="footer", pattern="Known Footer", decision="split")]
    )
    footer = make_learned_footer_callback(
        lambda *_args: (_ for _ in ()).throw(AssertionError("prompt should be skipped")),
        learned,
        teach=True,
    )
    assert footer("Known Footer 12", 12, {}) is True

    list_callback = make_learned_list_continuation_callback(
        lambda *_args: True,
        learned,
        teach=True,
        source_file="memory",
    )
    assert list_callback("1. Start", "continuation", 1, {}) is True
    assert learned.find_match("list_continuation", "1. Start\ncontinuation") is not None


def test_report_assembly_consumes_supplied_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        "pdf_chunker.core_new._dependency_versions",
        lambda: (_ for _ in ()).throw(AssertionError("assembly must not inspect dependencies")),
    )
    monkeypatch.setattr(
        "pdf_chunker.core_new._pass_hashes",
        lambda _passes: (_ for _ in ()).throw(AssertionError("assembly must not read sources")),
    )
    environment = {"sys_version": "test", "passes": {"demo": "hash"}}
    report = assemble_report({}, {"metrics": {}}, [], environment=environment)
    assert report["metrics"]["env"] == environment


def test_splitter_programming_errors_are_not_silently_replaced(monkeypatch) -> None:
    def broken_splitter(*_args, **_kwargs):
        raise RuntimeError("splitter regression")

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", broken_splitter)
    split, _metrics = _get_split_fn(20, 0, 10)
    with pytest.raises(RuntimeError, match="splitter regression"):
        split("a small input")


def test_canonical_emission_preserves_semantic_boundaries() -> None:
    items = [
        {"text": "First semantic chunk.", "meta": {"chunk_id": "x_p1_c0"}},
        {"text": "Second semantic chunk.", "meta": {"chunk_id": "x_p1_c1"}},
    ]
    result = emit_jsonl(
        Artifact(
            payload={"type": "chunks", "items": items},
            meta={"semantic_pipeline": "canonical-v1"},
        )
    )
    assert [row["text"] for row in result.payload] == [item["text"] for item in items]
    assert [row["metadata"]["chunk_id"] for row in result.payload] == ["x_p0_c0", "x_p0_c1"]


def test_semantic_pass_owns_duplicate_removal() -> None:
    artifact = split_semantic(
        Artifact(
            payload={
                "type": "page_blocks",
                "pages": [
                    {
                        "page": 1,
                        "blocks": [{"text": "A repeated sentence."}],
                    },
                    {
                        "page": 2,
                        "blocks": [{"text": "A repeated sentence."}],
                    },
                ],
            }
        )
    )
    assert [item["text"] for item in artifact.payload["items"]] == ["A repeated sentence."]


def test_semantic_pass_marks_canonical_boundaries_and_trace_log() -> None:
    artifact = split_semantic(
        Artifact(
            payload={
                "type": "page_blocks",
                "pages": [
                    {
                        "page": 1,
                        "blocks": [
                            {"text": "The first sentence"},
                            {"text": "continues and ends here."},
                        ],
                    }
                ],
            },
            meta={"trace_transformations": True},
        )
    )
    assert artifact.meta["semantic_pipeline"] == "canonical-v1"
    log = artifact.meta["transformation_log"]
    assert isinstance(log, dict)
    assert any(entry["kind"] == "merged" for entry in log["entries"])
