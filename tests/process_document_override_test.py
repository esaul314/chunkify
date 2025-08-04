#!/usr/bin/env python3

from pdf_chunker.core import process_document


def _extractor(fp: str, ex: str | None) -> list[dict]:
    return [
        {"text": "alpha", "source": {"page": 1}},
        {"text": "beta", "source": {"page": 2}},
    ]


def _chunker(
    blocks: list[dict],
    chunk_size: int,
    overlap: int,
    *,
    min_chunk_size: int,
    enable_dialogue_detection: bool,
) -> list[str]:
    return [" ".join(block["text"] for block in blocks)]


def _enricher(docs, blocks, **_):
    return [{"text": docs[0].content, "metadata": {"custom": True}}]


def test_callable_overrides():
    result = process_document(
        "dummy.pdf",
        chunk_size=100,
        overlap=0,
        generate_metadata=False,
        ai_enrichment=False,
        extractor=_extractor,
        chunker=_chunker,
        enricher=_enricher,
    )
    assert result == [{"text": "alpha beta", "metadata": {"custom": True}}]
