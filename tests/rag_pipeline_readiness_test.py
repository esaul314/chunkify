from __future__ import annotations

from pdf_chunker.config import load_spec
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import _rows
from pdf_chunker.passes.split_semantic import make_splitter
from pdf_chunker.adapters.ai_enrich import init_llm


def _observed_overlap(first: list[str], second: list[str]) -> int:
    """Return the number of overlapping words shared by two sequential chunks."""
    limit = min(len(first), len(second))
    return max(
        (size for size in range(limit, -1, -1) if first[-size:] == second[:size]),
        default=0,
    )


def test_rag_spec_overlap_and_metadata(monkeypatch) -> None:
    monkeypatch.setenv("PDF_CHUNKER_JSONL_META_KEY", "metadata")
    spec = load_spec("pipeline_rag.yaml")
    split_opts = spec.options["split_semantic"]
    chunk_size = split_opts["chunk_size"]
    overlap = split_opts["overlap"]

    assert "ai_enrich" in spec.pipeline
    assert spec.options["ai_enrich"]["enabled"] is True

    word_count = chunk_size + overlap + 50
    text = " ".join(f"w{i}" for i in range(word_count))
    doc = {
        "type": "page_blocks",
        "source_path": "rag.pdf",
        "pages": [
            {
                "blocks": [
                    {
                        "text": text,
                        "type": "paragraph",
                        "source": {"filename": "rag.pdf", "page": 3, "page_range": (3, 4)},
                    }
                ]
            }
        ],
    }

    splitter = make_splitter(chunk_size=chunk_size, overlap=overlap, generate_metadata=True)
    artifact = splitter(Artifact(payload=doc))
    items = artifact.payload["items"]

    assert len(items) >= 2
    first_words = items[0]["text"].split()
    second_words = items[1]["text"].split()
    assert _observed_overlap(first_words, second_words) == overlap

    rows = _rows(artifact.payload, preserve=True)
    assert rows
    meta = rows[0]["metadata"]

    assert meta["chunk_id"]
    assert meta["source"] == "rag.pdf"
    assert meta["source_file"] == "rag.pdf"
    assert meta["page"] == 3
    assert meta["page_range"] == "3-4"
    assert "utterance_type" in meta
    assert isinstance(meta["tags"], list)


def test_litellm_preflight(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    completion = init_llm()
    assert callable(completion)
