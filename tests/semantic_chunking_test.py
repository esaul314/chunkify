import math
import re

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import _dedupe
from pdf_chunker.passes.sentence_fusion import (
    _compute_limit,
    _derive_merge_budget,
    _merge_sentence_fragments,
    _stitch_continuation_heads,
)
from pdf_chunker.passes.split_semantic import (
    _SplitSemanticPass,
    _inject_continuation_context,
    _starts_list_like,
    _stitch_block_continuations,
)


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


def test_stitch_forces_boundary_when_limit_blocks_context() -> None:
    """Continuation stitching pushes trailing fragments into the next chunk."""

    fragments = [
        "Intro sentence. And this fragment trails without punctuation",
        "And now the continuation completes the idea.",
    ]
    stitched = _stitch_continuation_heads(fragments, limit=5)
    assert stitched == [
        "Intro sentence.",
        "And this fragment trails without punctuation And now the continuation completes the idea.",
    ]


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


def test_sentence_merge_blocks_dense_fragments_when_override_shrinks_budget() -> None:
    """Dense fragments respect override budgets instead of merging endlessly."""

    fragments = ["x" * 250, "y" * 250]
    merged_default = _merge_sentence_fragments(
        fragments,
        chunk_size=400,
        overlap=0,
        min_chunk_size=None,
    )
    merged_override = _merge_sentence_fragments(
        fragments,
        chunk_size=80,
        overlap=0,
        min_chunk_size=None,
    )

    assert len(merged_default) == 1
    assert merged_default[0].startswith("x" * 250)
    assert merged_default[0].endswith("y" * 250)
    assert merged_override == fragments


def test_compute_limit_handles_small_chunk_override() -> None:
    """Fallback limits keep small chunk overrides from inflating merges."""

    assert _compute_limit(chunk_size=5, overlap=0, min_chunk_size=8) == 5


def test_compute_limit_applies_overlap_margin() -> None:
    """Chunk limits deduct overlap before enforcing minimum capacity."""

    assert _compute_limit(chunk_size=123, overlap=23, min_chunk_size=None) == 100


def test_merge_budget_respects_word_total_when_text_has_spacing() -> None:
    """Normal sentences rely on word counts for their merge budget."""

    prev = tuple("a bb ccc".split())
    current = tuple("dd ee".split())
    budget = _derive_merge_budget(
        prev,
        current,
        chunk_size=10,
        overlap=2,
        min_chunk_size=3,
    )
    assert budget.limit == _compute_limit(10, 2, 3)
    assert budget.hard_limit == 10
    assert budget.word_total == 5
    assert budget.effective_total == budget.word_total


def test_merge_budget_accounts_for_dense_fragments() -> None:
    """Whitespace-free runs trigger the character-density fallback."""

    fragment = tuple(("a" * 200,))
    budget = _derive_merge_budget(
        fragment,
        fragment,
        chunk_size=50,
        overlap=0,
        min_chunk_size=None,
    )
    expected_dense = math.ceil((200 + 200) / 5.0)
    assert budget.word_total == 2
    assert budget.dense_total == expected_dense
    assert budget.effective_total == expected_dense
    assert budget.effective_total > 50


def test_merge_budget_handles_missing_chunk_limits() -> None:
    """Absent chunk overrides still compute deterministic fragment load."""

    budget = _derive_merge_budget((), ("dense",), chunk_size=None, overlap=5)
    assert budget.limit is None
    assert budget.hard_limit is None
    assert budget.word_total == 1
    assert budget.dense_total == 1


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

    dense_tail = ["Alphabeta", "gamma delta."]
    merged_dense = _merge_sentence_fragments(
        dense_tail,
        chunk_size=3,
        overlap=0,
        min_chunk_size=None,
    )
    assert merged_dense == ["Alphabeta gamma delta."]


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


def test_inline_metadata_flags_list_blocks() -> None:
    """Inline style metadata marks list items even without leading bullets."""

    block = {
        "text": "continuation without visible bullet",
        "inline_styles": (
            {"start": 0, "end": 3, "attrs": {"list_kind": "bullet"}},
        ),
    }
    assert _starts_list_like(block, block["text"])


def test_stitch_skips_list_context_and_preserves_tail() -> None:
    """List blocks do not borrow sentence context that would drop tails."""

    list_block = {
        "text": "• Lead section for long bullet",
        "type": "list_item",
        "list_kind": "bullet",
        "inline_styles": (
            {"start": 0, "end": 1, "attrs": {"list_kind": "bullet"}},
        ),
    }
    continuation = dict(list_block, text="and the trailing tail")
    stitched = _stitch_block_continuations(
        [
            (1, list_block, list_block["text"]),
            (1, continuation, continuation["text"]),
        ],
        limit=None,
    )
    assert [text for _, _, text in stitched][-1] == "and the trailing tail"


def test_inject_continuation_context_skips_lists() -> None:
    """Continuation injection keeps list chunks intact."""

    items = [
        {
            "text": "• Lead section",
            "meta": {"block_type": "list_item", "list_kind": "bullet"},
        },
        {
            "text": "and remaining details",
            "meta": {"block_type": "list_item", "list_kind": "bullet"},
        },
    ]
    processed = list(_inject_continuation_context(items, limit=None, overlap=0))
    assert processed[1]["text"] == "and remaining details"


def test_stitch_logs_warning_when_limit_prevents_context(caplog) -> None:
    """Continuation stitching emits a warning when the limit blocks context."""

    lead_block = {"text": "Intro fragment missing punctuation", "type": "paragraph"}
    continuation = {"text": "And the trailing completion", "type": "paragraph"}
    with caplog.at_level("WARNING"):
        stitched = _stitch_block_continuations(
            [
                (1, lead_block, lead_block["text"]),
                (1, continuation, continuation["text"]),
            ],
            limit=1,
        )
    assert len(stitched) == 1
    assert any("split_semantic" in record.message for record in caplog.records)
