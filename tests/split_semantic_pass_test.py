from __future__ import annotations

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_semantic import _SplitSemanticPass


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

    chunk = art.payload["items"][0]
    metrics = art.meta["metrics"]["split_semantic"]

    assert art.payload["type"] == "chunks"
    assert captured["args"] == (123, 7, 11)
    assert chunk["id"] == "0" and chunk["meta"]["page"] == 1
    assert chunk["meta"]["source"] == "src.pdf"
    assert len(chunk["text"]) == 8_000
    assert metrics["soft_limit_hits"] == 1 and metrics["hard_limit_hit"]
