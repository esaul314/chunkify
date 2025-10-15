from __future__ import annotations

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_semantic import (
    _SplitSemanticPass,
    _segment_char_limit,
)


def _doc(text: str) -> dict:
    return {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [{"page": 1, "blocks": [{"text": text}]}],
    }


def test_enforces_limits_and_structure(monkeypatch) -> None:
    captured: dict[str, tuple[int, int, int]] = {}

    def fake_semantic_chunker(
        text: str, chunk_size: int, overlap: int, *, min_chunk_size: int
    ) -> list[str]:
        captured["args"] = (chunk_size, overlap, min_chunk_size)
        return [text]

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", fake_semantic_chunker)

    long_text = "x" * 30_000
    art = _SplitSemanticPass(chunk_size=123, overlap=7, min_chunk_size=11)(
        Artifact(payload=_doc(long_text))
    )

    chunks = art.payload["items"]
    metrics = art.meta["metrics"]["split_semantic"]
    char_budget = _segment_char_limit(123)

    assert art.payload["type"] == "chunks"
    assert captured["args"] == (123, 7, 11)
    assert chunks[0]["id"] == "0" and chunks[0]["meta"]["page"] == 1
    assert chunks[0]["meta"]["source"] == "src.pdf"
    lengths = [len(chunk["text"]) for chunk in chunks]
    assert lengths[0] == char_budget
    assert all(length <= char_budget for length in lengths)
    assert metrics["soft_limit_hits"] == 1


def test_dense_single_token_chunks_respect_budget() -> None:
    chunk_size = 20
    char_budget = _segment_char_limit(chunk_size)
    long_token = "x" * (char_budget * 3 + 5)
    artifact = _SplitSemanticPass(chunk_size=chunk_size, overlap=0)(
        Artifact(payload=_doc(long_token))
    )
    texts = [chunk["text"] for chunk in artifact.payload["items"]]
    assert texts and len(texts[0]) == char_budget
    assert all(len(text) <= char_budget for text in texts)


def test_forced_budget_split_avoids_overlap_duplication() -> None:
    text = (
        "Instead of thinking of an offering as done the minute they got one customer "
        "successfully onboarded, the teams took a staged approach, starting by onboarding "
        "less critical applications. These applications provided data they could use for "
        "performance tuning and ironing out other bugs. Once they had these improvements "
        "in hand, they used them to gain the trust of the next tranche of more critical "
        "use cases, and so on.\n\nTrust continues when the next phase builds on that "
        "momentum without repeating itself."
    )
    chunk_size = 36
    overlap = 12
    artifact = _SplitSemanticPass(chunk_size=chunk_size, overlap=overlap)(
        Artifact(payload=_doc(text))
    )
    texts = [chunk["text"] for chunk in artifact.payload["items"]]
    target = "the teams took a staged approach"
    containing = [t for t in texts if target in t]
    assert containing, "expected the staged approach paragraph to survive"
    assert all(t.count(target) == 1 for t in containing)
    assert not any("use cases,\n\n the teams took" in t for t in texts)
