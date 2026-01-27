from __future__ import annotations

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_modules.segments import (
    _allow_colon_list_overflow,
    _collapse_step,
    _CollapseEmitter,
    _effective_counts,
    _segment_is_colon_list,
)
from pdf_chunker.passes.split_semantic import (
    _SplitSemanticPass,
    SplitOptions,
)
from pdf_chunker.strategies.bullets import default_bullet_strategy


def _doc(text: str) -> dict:
    return {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [{"page": 1, "blocks": [{"text": text}]}],
    }


def _doc_blocks(blocks: list[dict]) -> dict:
    """Create a page_blocks document from a list of block dicts."""
    return {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [{"page": 1, "blocks": blocks}],
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
    assert metrics["soft_limit_hits"] == 1


# ---------------------------------------------------------------------------
# Colon-list cohesion tests
# ---------------------------------------------------------------------------


def test_segment_is_colon_list_detects_colon_bullet_pattern() -> None:
    """Verify _segment_is_colon_list identifies colon-introduced lists."""
    strategy = default_bullet_strategy()

    # Colon ending followed by bullet list
    segment = (
        (1, {"text": "Intro:", "type": "paragraph"}, "Intro:"),
        (1, {"text": "• Item", "type": "paragraph"}, "• Item"),
    )
    assert _segment_is_colon_list(segment, strategy=strategy) is True

    # No colon - should not match
    segment_no_colon = (
        (1, {"text": "Intro", "type": "paragraph"}, "Intro"),
        (1, {"text": "• Item", "type": "paragraph"}, "• Item"),
    )
    assert _segment_is_colon_list(segment_no_colon, strategy=strategy) is False

    # Colon but no list following - should not match
    segment_no_list = (
        (1, {"text": "Question:", "type": "paragraph"}, "Question:"),
        (1, {"text": "Answer text", "type": "paragraph"}, "Answer text"),
    )
    assert _segment_is_colon_list(segment_no_list, strategy=strategy) is False


def test_allow_colon_list_overflow_permits_merge() -> None:
    """Verify _allow_colon_list_overflow allows keeping colon-intro with list."""
    strategy = default_bullet_strategy()

    intro = (1, {"text": "Key points:", "type": "paragraph"}, "Key points:")
    bullets = (1, {"text": "• First\n• Second", "type": "paragraph"}, "• First\n• Second")

    assert _allow_colon_list_overflow((intro,), bullets, strategy=strategy) is True

    # Without colon - should not allow overflow
    intro_no_colon = (1, {"text": "Key points", "type": "paragraph"}, "Key points")
    assert _allow_colon_list_overflow((intro_no_colon,), bullets, strategy=strategy) is False


def test_colon_intro_and_bullet_list_stay_together_in_single_chunk() -> None:
    """Colon-introduced list blocks should merge into single chunk.

    This is the key regression test: when text ends with a colon introducing
    a list, the list items should stay in the same chunk even when the
    combined size would normally trigger a split.
    """
    strategy = default_bullet_strategy()

    intro_text = (
        "This is a comprehensive introduction to the key challenges. "
        "After careful analysis, we identified the following critical issues:"
    )
    bullet_text = (
        "• First challenge: Managing technical debt\n"
        "• Second challenge: Ensuring code quality\n"
        "• Third challenge: Coordinating deployments"
    )

    records = [
        (1, {"text": intro_text, "type": "paragraph"}, intro_text),
        (1, {"text": bullet_text, "type": "paragraph"}, bullet_text),
    ]

    # Use a small limit that would normally cause a split
    small_limit = 50  # word-based limit, both records exceed this together
    emitter = _CollapseEmitter(
        resolved_limit=small_limit,
        hard_limit=small_limit,
        overlap=0,
        strategy=strategy,
    )

    state = emitter
    for idx, record in enumerate(records):
        counts = _effective_counts(record[2])
        state = _collapse_step(state, (idx, record))

    state = state.flush()

    # Both records should be in a single output
    assert len(state.outputs) == 1, (
        f"Expected 1 merged output, got {len(state.outputs)}. "
        "Colon-intro and bullet list should stay together."
    )

    # Verify the merged text contains both the intro and bullets
    merged_text = state.outputs[0][2]
    assert "critical issues:" in merged_text
    assert "• First challenge" in merged_text
    assert "• Third challenge" in merged_text


def test_colon_intro_and_numbered_list_stay_together_when_list_exceeds_limit() -> None:
    """Colon-introduced numbered list should merge even when list alone exceeds limit.

    This regression test covers the case where a numbered list block alone exceeds
    chunk_size, but should still merge with a preceding colon-ending intro because
    the colon introduces the list.
    """
    from pdf_chunker.passes.split_modules.segments import collapse_records

    strategy = default_bullet_strategy()

    # Intro that ends with colon
    intro_text = (
        "This is a comprehensive introduction to the key challenges in software "
        "engineering. After careful analysis, we identified the following critical "
        "issues that need to be addressed immediately:"
    )
    # Numbered list that alone exceeds the chunk_size limit (40 words)
    numbered_text = (
        "1. First issue is the growing technical debt that has been accumulating "
        "over the years.\n"
        "2. Second issue is the lack of proper documentation for critical systems.\n"
        "3. Third issue is the insufficient test coverage in production code."
    )

    records = [
        (1, {"text": intro_text, "type": "paragraph"}, intro_text),
        (1, {"text": numbered_text, "type": "paragraph"}, numbered_text),
    ]

    # Use a limit smaller than the numbered list alone (38 words)
    opts = SplitOptions(chunk_size=40, overlap=10, min_chunk_size=10)
    result = list(collapse_records(records, opts, limit=40, strategy=strategy))

    # Should produce single merged output despite exceeding limit
    assert len(result) == 1, (
        f"Expected 1 merged output, got {len(result)}. "
        "Colon-intro and numbered list should stay together even when list exceeds limit."
    )

    merged_text = result[0][2]
    assert "issues that need to be addressed immediately:" in merged_text
    assert "1. First issue" in merged_text
    assert "3. Third issue" in merged_text
