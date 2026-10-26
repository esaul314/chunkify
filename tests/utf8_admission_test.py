"""Opt-in complete-row UTF-8 admission through the existing enrichment path."""

import copy
import json
from functools import partial

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker.adapters import article_conversion as article_adapter
from pdf_chunker.adapters.article_conversion import (
    execute_bound_conversion,
    prepare_bound_conversion,
)
from pdf_chunker.article_output import validate_article_output
from pdf_chunker.config import PipelineSpec, validate_options
from pdf_chunker.core_new import _run_passes
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.ai_enrich import _admit_chunks_for_jsonl
from pdf_chunker.passes.emit_jsonl import EmitConfig
from tests.verified_markdown_ingress_test import SOURCE, bind_article


class RecordingClient:
    def __init__(self, tags=()):
        self.inputs = []
        self.tags = list(tags)

    def classify_chunk_utterance(self, text, *, tag_configs):
        self.inputs.append(text)
        return {"classification": "question", "tags": list(self.tags)}


def encoded(row):
    return json.dumps(row, ensure_ascii=False).encode("utf-8")


def pipeline(items, *, client, limit, meta=None, metadata_key="metadata"):
    return _run_passes(
        PipelineSpec(
            pipeline=["ai_enrich", "emit_jsonl"],
            options={
                "ai_enrich": {
                    "enabled": True,
                    "client": client,
                    "tag_configs": {"topic": ["界🧭"]},
                },
                "emit_jsonl": {
                    "max_chars": 8000,
                    "max_row_bytes": limit,
                    "metadata_key": metadata_key,
                },
            },
        ),
        Artifact(
            payload={"type": "chunks", "items": items},
            meta={"semantic_pipeline": "canonical-v1", **(meta or {})},
        ),
    )[0]


@given(
    text=st.text(alphabet='界🧭é"\\\n\t A\u0085\u2028\u2029', min_size=1, max_size=300),
    byte_limit=st.integers(min_value=220, max_value=600),
)
def test_admission_is_lossless_pure_deterministic_and_partially_applicable(text, byte_limit):
    chunks = [{"text": text, "meta": {"chunk_id": "source_p1_c0", "source": "界.md"}}]
    original = copy.deepcopy(chunks)
    admit = partial(
        _admit_chunks_for_jsonl,
        max_chars=400,
        max_row_bytes=byte_limit,
        metadata_key="metadata",
        tag_configs={"topic": ["界🧭"]},
    )
    parts = admit(chunks)

    assert chunks == original
    assert admit(chunks) == parts
    assert "".join(part["text"] for part in parts) == text
    for part in parts:
        meta = {
            **part["meta"],
            "utterance_type": "statement_of_fact",
            "tags": ["界🧭"],
            "enrichment_status": "complete",
        }
        row = {"text": part["text"], "metadata": meta}
        assert len(encoded(row)) <= byte_limit
        assert len(encoded(row).decode("utf-8")) <= 400
        assert part["meta"]["chunk_id"] == "source_p1_c0"
        assert part["meta"]["source"] == "界.md"
    if len(parts) > 1:
        assert [part["meta"]["chunk_part"] for part in parts] == list(range(len(parts)))


def test_pipeline_splits_before_classification_and_emits_exact_bounded_text():
    text = '界🧭 "quoted" \\\n\t\u0085\u2028\u2029 ' * 80
    client = RecordingClient(tags=["界🧭"])
    items = [{"text": text, "meta": {"chunk_id": "book_p1_c0", "source": "界.pdf"}}]
    original = copy.deepcopy(items)

    result = pipeline(items, client=client, limit=512)

    assert items == original
    assert len(client.inputs) > 1
    assert sorted(client.inputs) == sorted(row["text"] for row in result.payload)
    assert "".join(row["text"] for row in result.payload) == text
    assert [row["metadata"]["chunk_part"] for row in result.payload] == list(
        range(len(result.payload))
    )
    assert all(len(encoded(row)) <= 512 for row in result.payload)
    assert all(row["metadata"]["tags"] == ["界🧭"] for row in result.payload)
    assert all(row["metadata"]["chunk_id"] == "book_p0_c0" for row in result.payload)


@pytest.mark.parametrize("metadata_key", ["metadata", "article_meta"])
def test_complete_metadata_and_custom_key_count_before_any_classification(metadata_key):
    client = RecordingClient()
    items = [
        {"text": "Fits.", metadata_key: {"source": "one.pdf"}},
        {"text": "Small.", metadata_key: {"source": "界🧭" * 1000}},
    ]
    with pytest.raises(ValueError, match="metadata exceeds JSONL record limit"):
        pipeline(items, client=client, limit=512, metadata_key=metadata_key)
    assert client.inputs == []


@pytest.mark.parametrize("kind", ["code", "table", "link"])
def test_verified_markdown_refuses_repartition_before_classification(kind):
    contexts = {
        "code": "```python\nvalue = '" + "界🧭" * 200 + "'\n```\n",
        "table": "| Key | Value |\n| --- | --- |\n| key | " + "界🧭" * 200 + " |\n",
        "link": "[source](https://example.org/" + "é" * 1000 + ")\n",
    }
    text = contexts[kind]
    items = [{"text": text, "meta": {"chunk_id": "article", "source": "article.md"}}]
    original = copy.deepcopy(items)
    client = RecordingClient()

    with pytest.raises(ValueError, match="verified Markdown context.*review required"):
        pipeline(
            items,
            client=client,
            limit=512,
            meta={"verified_joplin_markdown": text},
        )
    assert client.inputs == []
    assert items == original


def test_actual_enriched_overflow_refuses_even_when_projection_fits():
    client = RecordingClient(tags=["界🧭" * 200])
    items = [{"text": "A small source.", "meta": {"source": "article.pdf"}}]
    original = copy.deepcopy(items)
    with pytest.raises(ValueError, match="enriched JSONL record.*byte"):
        pipeline(items, client=client, limit=512)
    assert client.inputs == ["A small source."]
    assert items == original


@pytest.mark.parametrize("canonical", [False, True])
def test_final_emitter_refuses_byte_overflow_without_enrichment(canonical):
    artifact = Artifact(
        payload={"type": "chunks", "items": [{"text": "界🧭 words " * 50}]},
        meta={"semantic_pipeline": "canonical-v1"} if canonical else {},
    )
    with pytest.raises(ValueError, match="JSONL.*byte"):
        _run_passes(
            PipelineSpec(pipeline=["emit_jsonl"], options={"emit_jsonl": {"max_row_bytes": 1}}),
            artifact,
        )


def test_exact_final_byte_boundary_excludes_lf_and_counts_complete_custom_metadata():
    artifact = Artifact(
        payload={
            "type": "chunks",
            "items": [{"text": "界🧭\u0085\u2028\u2029", "article_meta": {"source": "é.md"}}],
        },
        meta={"semantic_pipeline": "canonical-v1"},
    )
    options = {"metadata_key": "article_meta"}
    spec = PipelineSpec(pipeline=["emit_jsonl"], options={"emit_jsonl": options})
    baseline, _ = _run_passes(spec, artifact)
    limit = len(encoded(baseline.payload[0]))
    assert limit > len(encoded(baseline.payload[0]).decode("utf-8"))
    spec.options["emit_jsonl"]["max_row_bytes"] = limit
    exact, _ = _run_passes(spec, artifact)
    assert exact.payload == baseline.payload
    spec.options["emit_jsonl"]["max_row_bytes"] = limit - 1
    with pytest.raises(ValueError, match="JSONL.*byte"):
        _run_passes(spec, artifact)


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "512"])
def test_byte_limit_requires_positive_integers_at_config_and_direct_pass_boundaries(invalid):
    with pytest.raises(ValueError, match="max_row_bytes"):
        validate_options({"emit_jsonl": {"max_row_bytes": invalid}})
    with pytest.raises(ValueError, match="max_row_bytes"):
        EmitConfig(max_row_bytes=invalid)
    with pytest.raises(ValueError, match="max_row_bytes"):
        _admit_chunks_for_jsonl(
            [{"text": "source"}],
            max_chars=8000,
            max_row_bytes=invalid,
            metadata_key="metadata",
            tag_configs={},
        )


def test_disabled_byte_budget_preserves_existing_character_admission():
    chunks = [{"text": "界🧭" * 200, "meta": {"source": "article.pdf"}}]
    settings = dict(max_chars=512, metadata_key="metadata", tag_configs={})
    assert _admit_chunks_for_jsonl(chunks, **settings) == _admit_chunks_for_jsonl(
        chunks, max_row_bytes=None, **settings
    )
    validate_options({"emit_jsonl": {"max_row_bytes": None}})
    assert EmitConfig().max_row_bytes is None


@pytest.mark.parametrize("suffix", [".md", ".epub"])
def test_bound_article_budget_is_pinned_and_checked_at_execution_and_readback(
    tmp_path, monkeypatch, suffix
):
    monkeypatch.setattr(article_adapter, "collect_article_runtime", lambda: {"fixture": "stable"})
    bound = bind_article(tmp_path / "article", suffix=suffix, text=SOURCE + "\n界🧭\n")
    spec = bound.config()
    baseline = prepare_bound_conversion(bound.binding, spec, corpus="technology")
    output = execute_bound_conversion(bound.binding, prepared=baseline, corpus="technology")
    limit = max(len(line) for line in output.jsonl.split(b"\n")[:-1])

    spec.options["emit_jsonl"]["max_row_bytes"] = limit
    exact = prepare_bound_conversion(bound.binding, spec, corpus="technology")
    assert exact.descriptor.key != baseline.descriptor.key
    assert exact.spec().options["emit_jsonl"]["max_row_bytes"] == limit
    assert (
        execute_bound_conversion(bound.binding, prepared=exact, corpus="technology").jsonl
        == output.jsonl
    )
    assert (
        validate_article_output(
            output.jsonl, descriptor=exact.descriptor, expected_sha256=output.sha256
        ).jsonl
        == output.jsonl
    )

    spec.options["emit_jsonl"]["max_row_bytes"] = limit - 1
    too_small = prepare_bound_conversion(bound.binding, spec, corpus="technology")
    with pytest.raises(ValueError, match="JSONL.*byte"):
        execute_bound_conversion(bound.binding, prepared=too_small, corpus="technology")
    with pytest.raises(ValueError, match="JSONL.*byte"):
        validate_article_output(
            output.jsonl, descriptor=too_small.descriptor, expected_sha256=output.sha256
        )
    assert not list(tmp_path.rglob("*.jsonl"))


def test_existing_part_identity_is_not_overwritten_by_byte_splitting():
    with pytest.raises(ValueError, match="already has chunk_part"):
        _admit_chunks_for_jsonl(
            [{"text": "界🧭" * 100, "meta": {"chunk_part": "upstream"}}],
            max_chars=8000,
            max_row_bytes=256,
            metadata_key="metadata",
            tag_configs={},
        )


def test_fitting_verified_context_is_preserved_through_enrichment():
    text = "  ```text\n界🧭\n```\n\n"
    client = RecordingClient()
    result = pipeline(
        [{"text": text, "meta": {"chunk_id": "article", "source": "article.md"}}],
        client=client,
        limit=1024,
        meta={"verified_joplin_markdown": text},
    )
    assert client.inputs == [text]
    assert result.payload[0]["text"] == text
    assert result.payload[0]["metadata"]["chunk_id"] == "article"
    assert "chunk_part" not in result.payload[0]["metadata"]


def test_exact_projected_byte_boundary_fits_without_repartitioning():
    text = "界🧭" * 50
    metadata = {"source": "article.pdf"}
    projected = {
        **metadata,
        "utterance_type": "statement_of_fact",
        "tags": ["界🧭"],
        "enrichment_status": "complete",
    }
    chunks = [{"text": text, "meta": metadata}]
    limit = len(encoded({"text": text, "metadata": projected}))
    admit = partial(
        _admit_chunks_for_jsonl,
        max_chars=8000,
        metadata_key="metadata",
        tag_configs={"topic": ["界🧭"]},
    )
    assert admit(chunks, max_row_bytes=limit) == chunks
    assert len(admit(chunks, max_row_bytes=limit - 1)) > 1
