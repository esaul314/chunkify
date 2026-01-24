import json

import pytest

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import _max_chars as _jsonl_max_chars
from pdf_chunker.passes.emit_jsonl import emit_jsonl
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
    """Test that emit_jsonl splits long text into multiple rows.

    Uses unique content to prevent deduplication/overlap trimming
    from discarding chunks. Avoids numbered patterns that would
    trigger list detection (e.g., "Sentence 0." looks like a list).
    """

    def make_unique_text(length: int) -> str:
        """Generate text with unique words to prevent dedup and list detection."""
        # Use unique words that won't trigger numbered list detection
        # Pattern: "Alpha beta gamma delta. Epsilon zeta eta theta." etc.
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        sentences = []
        i = 0
        while len(" ".join(sentences)) < length:
            # Include unique counter in middle to avoid dedup, but not as list marker
            w = words[i % len(words)]
            sentences.append(f"The {w} value is {i} units.")
            i += 1
        return " ".join(sentences)

    # Generate text longer than max_chars
    text = make_unique_text(jsonl_max_chars * 2)
    json_len = len(json.dumps({"text": text}, ensure_ascii=False))

    # Verify preconditions
    assert json_len > jsonl_max_chars, (
        f"Text JSON should exceed limit: {json_len} vs {jsonl_max_chars}"
    )

    # Run emit_jsonl
    rows = emit_jsonl(Artifact(payload={"type": "chunks", "items": [{"text": text}]})).payload

    # Should split into multiple rows
    assert len(rows) > 1, f"Expected multiple rows, got {len(rows)}"
    # Each row should be within the limit
    assert all(len(json.dumps(r, ensure_ascii=False)) <= jsonl_max_chars for r in rows)


def test_split_semantic_produces_bounded_chunks(default_split_limits: tuple[int, int]) -> None:
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
