import re
import sys
import types

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import _dedupe
from pdf_chunker.passes.split_semantic import _SplitSemanticPass


def _doc(text: str) -> dict:
    return {
        "type": "page_blocks",
        "pages": [{"page": 1, "blocks": [{"text": text}]}],
    }


def test_limits_and_metrics() -> None:
    """Chunks obey soft/hard limits and expose metrics."""
    text = "x" * 26_000
    splitter = _SplitSemanticPass(chunk_size=100_000, overlap=0)
    art = splitter(Artifact(payload=_doc(text)))
    chunks = [c["text"] for c in art.payload["items"]]
    metrics = art.meta["metrics"]["split_semantic"]
    assert len(chunks) > 1
    assert all(len(c) <= 8_000 for c in chunks)
    assert metrics["soft_limit_hits"] == 1


def test_parameter_propagation() -> None:
    """Custom chunk sizing parameters propagate to the splitter."""
    words = " ".join(f"w{i}" for i in range(20))
    opts = {
        "options": {
            "split_semantic": {
                "chunk_size": 5,
                "overlap": 1,
                "min_chunk_size": 2,
            }
        }
    }
    art = _SplitSemanticPass()(Artifact(payload=_doc(words), meta=opts))
    texts = [c["text"] for c in art.payload["items"]]
    counts = [len(t.split()) for t in texts]
    assert counts == [5, 5, 5, 5, 4]
    assert texts[1].split()[0] == "w4"


def test_no_chunk_starts_mid_sentence() -> None:
    """Chunks begin at sentence boundaries and never start mid-sentence."""
    end_re = re.compile(r"[.?!][\"')\]]*$")
    long_sentence = " ".join(f"w{i}" for i in range(120)) + "."
    text = f"{long_sentence} Next one."
    art = _SplitSemanticPass(chunk_size=10, overlap=0)(Artifact(payload=_doc(text)))
    chunks = [c["text"] for c in art.payload["items"]]
    assert all(end_re.search(prev.rstrip()) for prev in chunks[:-1])


def test_blocks_merge_into_sentence() -> None:
    """Adjacent blocks merge so chunks don't start mid-sentence."""
    doc = {
        "type": "page_blocks",
        "pages": [
            {
                "page": 1,
                "blocks": [
                    {"text": "Cloud"},
                    {"text": "development envs are new."},
                ],
            }
        ],
    }
    art = _SplitSemanticPass()(Artifact(payload=doc))
    texts = [c["text"] for c in art.payload["items"]]
    assert texts == ["Cloud development envs are new."]


def test_semantic_splitter_merges_conversation(monkeypatch) -> None:
    """Semantic splitting uses conversational merges before word segmentation."""

    base_chunks = [
        "Hello there",
        "and welcome to the platform?",
        "Closing statement.",
    ]
    merged_chunks = [
        "Hello there ⧉ and welcome to the platform?",
        "Closing statement.",
    ]

    def fake_semantic_chunker(
        text: str,
        *,
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
    ) -> list[str]:
        return list(base_chunks)

    def fake_merge(
        chunks: list[str],
        min_chunk_size: int,
    ) -> tuple[list[str], dict[str, int]]:
        assert list(chunks) == base_chunks
        return list(merged_chunks), {"merges_performed": 1}

    def fake_iter_word_chunks(chunk: str, max_words: int) -> list[str]:
        assert chunk in merged_chunks
        return [chunk]

    stub = types.SimpleNamespace(
        semantic_chunker=fake_semantic_chunker,
        merge_conversational_chunks=fake_merge,
        iter_word_chunks=fake_iter_word_chunks,
    )
    monkeypatch.setitem(sys.modules, "pdf_chunker.splitter", stub)

    art = _SplitSemanticPass(chunk_size=1_024, overlap=0)(
        Artifact(payload=_doc("Placeholder text."))
    )
    texts = [c["text"] for c in art.payload["items"]]
    metrics = art.meta["metrics"]["split_semantic"]

    assert texts == merged_chunks
    assert metrics.get("conversational_merges") == 1
    assert all(text and not text.lstrip()[0].islower() for text in texts)


def test_dedupe_preserves_sentence_start() -> None:
    """Dedupe merges fragments so outputs don't start mid-sentence."""
    items = [
        {"text": "Prime numbers are tricky"},
        {"text": "are tricky to reason about."},
    ]
    texts = [r["text"] for r in _dedupe(items)]
    assert texts == ["Prime numbers are tricky to reason about."]
