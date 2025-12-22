import json
from itertools import count

import pytest

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import emit_jsonl, _max_chars as _jsonl_max_chars
from pdf_chunker.passes.split_semantic import make_splitter, split_semantic


def _observed_overlap(first: list[str], second: list[str]) -> int:
    """Return the number of overlapping words shared by two sequential chunks."""

    limit = min(len(first), len(second))
    return max(
        (size for size in range(limit, -1, -1) if first[-size:] == second[:size]),
        default=0,
    )


@pytest.fixture(scope="module")
def jsonl_max_chars() -> int:
    return _jsonl_max_chars()


@pytest.fixture(scope="module")
def default_split_limits() -> tuple[int, int]:
    splitter = make_splitter()
    return splitter.chunk_size, splitter.overlap


def test_emit_jsonl_splits_and_clamps_rows(jsonl_max_chars: int) -> None:
    texts = (f"A{'a' * n}." for n in count(jsonl_max_chars))
    candidates = (
        (text, emit_jsonl(Artifact(payload={"type": "chunks", "items": [{"text": text}]})).payload)
        for text in texts
    )
    long_text, rows = next(
        (text, result)
        for text, result in candidates
        if len(result) > 1
        and len(json.dumps({"text": text}, ensure_ascii=False)) > jsonl_max_chars
    )
    assert len(long_text) > jsonl_max_chars
    assert len(rows) > 1
    assert all(len(json.dumps(r, ensure_ascii=False)) <= jsonl_max_chars for r in rows)


def test_split_semantic_produces_bounded_chunks(
    default_split_limits: tuple[int, int]
) -> None:
    chunk_size, overlap = default_split_limits
    word_count = chunk_size * 3 + overlap * 2 + 7
    long_text = " ".join(f"w{i}" for i in range(word_count))
    doc = {"type": "page_blocks", "pages": [{"blocks": [{"text": long_text}]}]}
    artifact = Artifact(payload=doc)
    items = split_semantic(artifact).payload["items"]
    chunk_words = [c["text"].split() for c in items]

    assert len(chunk_words) > 1
    assert len(chunk_words[0]) == chunk_size
    assert all(len(words) <= chunk_size for words in chunk_words)
    assert _observed_overlap(chunk_words[0], chunk_words[1]) == overlap
