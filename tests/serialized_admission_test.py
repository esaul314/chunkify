from __future__ import annotations

import json

import pytest

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _run_passes
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.ai_enrich import _admit_chunks_for_jsonl, _payload_chunks


class _RecordingClient:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def classify_chunk_utterance(self, text: str, *, tag_configs: dict) -> dict:
        self.inputs.append(text)
        return {"classification": "question", "tags": []}


def test_admission_splits_losslessly_by_escaped_serialized_size() -> None:
    source = 'Quote " slash \\\\ caf\u00e9 \U0001f9ed\ncontrol\t. ' * 100
    tags = {"topic": ["bounded-tag", "another-tag"]}
    original_meta = {"chunk_id": "book_p1_c7", "source": "shelf/Book.pdf"}
    limit = 512

    parts = _admit_chunks_for_jsonl(
        [{"text": source, "meta": original_meta}],
        max_chars=limit,
        metadata_key="metadata",
        tag_configs=tags,
    )

    assert len(parts) > 1
    assert "".join(part["text"] for part in parts) == source
    assert [part["meta"]["chunk_part"] for part in parts] == list(range(len(parts)))
    assert all(
        part["meta"]["chunk_id"] == original_meta["chunk_id"]
        and part["meta"]["source"] == original_meta["source"]
        for part in parts
    )
    for part in parts:
        metadata = {
            **part["meta"],
            "utterance_type": "statement_of_fact",
            "tags": ["another-tag", "bounded-tag"],
            "enrichment_status": "complete",
        }
        assert (
            len(
                json.dumps(
                    {"text": part["text"], "metadata": metadata},
                    ensure_ascii=False,
                )
            )
            <= limit
        )


def test_admission_refuses_metadata_that_cannot_fit() -> None:
    with pytest.raises(ValueError, match="metadata exceeds JSONL record limit"):
        _admit_chunks_for_jsonl(
            [{"text": "x", "meta": {"source": "y" * 1000}}],
            max_chars=128,
            metadata_key="metadata",
            tag_configs={},
        )


def test_pipeline_admits_before_enrichment_and_emits_bounded_rows() -> None:
    source = 'A long "quoted" sentence with caf\u00e9 \U0001f9ed and exact spacing.\n' * 80
    client = _RecordingClient()
    limit = 512
    tag_configs = {"topic": ["bounded-tag"]}
    expected_parts = _admit_chunks_for_jsonl(
        [{"text": source, "meta": {"chunk_id": "book_p1_c7", "source": "Book.pdf"}}],
        max_chars=limit,
        metadata_key="metadata",
        tag_configs=tag_configs,
    )
    spec = PipelineSpec(
        pipeline=["ai_enrich", "emit_jsonl"],
        options={
            "ai_enrich": {
                "enabled": True,
                "client": client,
                "tag_configs": tag_configs,
            },
            "emit_jsonl": {"max_chars": limit},
        },
    )
    artifact = Artifact(
        payload={
            "type": "chunks",
            "items": [{"text": source, "meta": {"chunk_id": "book_p1_c7", "source": "Book.pdf"}}],
        },
        meta={"semantic_pipeline": "canonical-v1"},
    )

    result, _ = _run_passes(spec, artifact)

    assert sorted(client.inputs) == sorted(part["text"] for part in expected_parts)
    assert all(len(json.dumps(row, ensure_ascii=False)) <= limit for row in result.payload)
    ordered_rows = sorted(result.payload, key=lambda row: row["metadata"]["chunk_part"])
    assert "".join(row["text"] for row in ordered_rows) == source
    assert [row["metadata"]["chunk_part"] for row in ordered_rows] == list(range(len(ordered_rows)))



def test_split_preserves_emitted_metadata_when_meta_and_metadata_coexist() -> None:
    source = ('Long source sentence with exact spacing.\n' * 100)
    client = _RecordingClient()
    spec = PipelineSpec(
        pipeline=["ai_enrich", "emit_jsonl"],
        options={
            "ai_enrich": {
                "enabled": True,
                "client": client,
                "tag_configs": {"topic": ["bounded-tag"]},
            },
            "emit_jsonl": {"max_chars": 512},
        },
    )
    artifact = Artifact(
        payload={
            "type": "chunks",
            "items": [
                {
                    "text": source,
                    "meta": {"chunk_id": "internal-id", "internal": "kept"},
                    "metadata": {"source": "Book.pdf", "edition": "public"},
                }
            ],
        },
        meta={"semantic_pipeline": "canonical-v1"},
    )

    result, _ = _run_passes(spec, artifact)
    rows = result.payload

    assert len(rows) > 1
    assert [row["metadata"]["chunk_part"] for row in rows] == list(range(len(rows)))
    assert all(row["metadata"]["source"] == "Book.pdf" for row in rows)
    assert all(row["metadata"]["edition"] == "public" for row in rows)
    assert "".join(row["text"] for row in rows) == source
    assert all(len(json.dumps(row, ensure_ascii=False)) <= 512 for row in rows)


def test_admission_refuses_to_overwrite_existing_chunk_part_identity() -> None:
    with pytest.raises(ValueError, match="already has chunk_part metadata"):
        _admit_chunks_for_jsonl(
            [{"text": "x" * 2000, "metadata": {"chunk_part": "upstream"}}],
            max_chars=256,
            metadata_key="metadata",
            tag_configs={},
        )


def test_canonical_emitter_honors_configured_hard_limit() -> None:
    spec = PipelineSpec(
        pipeline=["emit_jsonl"],
        options={"emit_jsonl": {"max_chars": 256}},
    )
    artifact = Artifact(
        payload={"type": "chunks", "items": [{"text": "x" * 300, "meta": {"source": "Book.pdf"}}]},
        meta={"semantic_pipeline": "canonical-v1"},
    )

    with pytest.raises(ValueError, match="semantic chunk exceeds JSONL size limit"):
        _run_passes(spec, artifact)


@pytest.mark.parametrize("wrapped", [False, True])
def test_split_reconstruction_preserves_interleaved_non_chunk_order(wrapped: bool) -> None:
    items = [{"text": "first"}, "unrelated", {"text": "second"}]
    payload = {"type": "chunks", "items": items} if wrapped else items

    chunks, rebuild = _payload_chunks(payload)
    marker = next(key for key in chunks[0] if key.startswith("__pdf_chunker_original_item_index__"))
    enriched = [
        {**chunks[0], "text": "first-a"},
        {**chunks[0], "text": "first-b"},
        {**chunks[1], "text": "second-a"},
    ]

    result = rebuild(enriched)
    result_items = result["items"] if wrapped else result

    assert [item.get("text", item) if isinstance(item, dict) else item for item in result_items] == [
        "first-a", "first-b", "unrelated", "second-a"
    ]
    assert all(marker not in item for item in result_items if isinstance(item, dict))
