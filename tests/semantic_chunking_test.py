from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import _dedupe
from pdf_chunker.passes.sentence_fusion import _compute_limit, _merge_sentence_fragments
from pdf_chunker.passes.split_semantic import _SplitSemanticPass
import re


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


def test_sentence_merge_allows_soft_limit_overflow() -> None:
    """Sentence fusion tolerates soft-limit overflow when hard limit allows it."""
    chunk_size, overlap = 12, 4
    fragments = [
        "Alpha beta gamma delta epsilon zeta",
        "and eta theta iota.",
    ]
    merged = _merge_sentence_fragments(
        fragments,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=2,
    )
    assert merged == ["Alpha beta gamma delta epsilon zeta and eta theta iota."]
    word_count = len(merged[0].split())
    assert chunk_size - overlap < word_count <= chunk_size


def test_compute_limit_handles_small_chunk_override() -> None:
    """Fallback limits keep small chunk overrides from inflating merges."""

    assert _compute_limit(chunk_size=5, overlap=0, min_chunk_size=8) == 5


def test_compute_limit_applies_overlap_margin() -> None:
    """Chunk limits deduct overlap before enforcing minimum capacity."""

    assert _compute_limit(chunk_size=123, overlap=23, min_chunk_size=None) == 100


def test_limit_fallback_dedupes_overlap_tokens() -> None:
    """Fallback chunks trim duplicated overlap tokens."""
    fragments = [
        "alpha beta gamma delta epsilon zeta eta theta",
        "eta theta iota kappa lambda mu",
    ]
    merged = _merge_sentence_fragments(
        fragments,
        chunk_size=10,
        overlap=2,
        min_chunk_size=2,
    )
    assert merged == [
        "alpha beta gamma delta epsilon zeta eta theta",
        "iota kappa lambda mu",
    ]


def test_sentence_merge_respects_small_chunk_capacity() -> None:
    """Sentence fusion honors tiny chunk overrides instead of merging endlessly."""

    fragments = [
        "Alpha beta gamma delta epsilon.",
        "And zeta eta theta iota kappa.",
        "And lambda mu nu xi omicron.",
    ]
    merged = _merge_sentence_fragments(
        fragments,
        chunk_size=5,
        overlap=0,
        min_chunk_size=8,
    )
    assert len(merged) > 1
    assert len(merged[0].split()) <= 5


def test_sentence_merge_large_chunks_respect_hard_cap() -> None:
    """Large chunk configurations still merge up to their hard cap."""

    lead = " ".join(f"w{i}" for i in range(90))
    tail = "and " + " ".join(f"w{i}" for i in range(90, 121)) + "."
    merged = _merge_sentence_fragments(
        (lead, tail),
        chunk_size=123,
        overlap=23,
        min_chunk_size=None,
    )
    expected = f"{lead} {tail}".strip()
    expected_words = len(expected.split())
    assert merged == [expected]
    assert 100 < expected_words <= 123


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


def test_dedupe_preserves_sentence_start() -> None:
    """Dedupe merges fragments so outputs don't start mid-sentence."""
    items = [
        {"text": "Prime numbers are tricky"},
        {"text": "are tricky to reason about."},
    ]
    texts = [r["text"] for r in _dedupe(items)]
    assert texts == ["Prime numbers are tricky to reason about."]
