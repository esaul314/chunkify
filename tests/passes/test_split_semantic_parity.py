"""Regression parity checks for semantic splitting.

These tests exercise both the refactored chunk pipeline utilities and the
existing :class:`_SplitSemanticPass`.  The goal is to demonstrate that the new
functional pipeline preserves canonical chunk text and metadata for fixtures
such as ``platform-eng-excerpt.pdf``.  Maintaining this invariance is a
prerequisite for rolling the modular pipeline out across the CLI.

Invariants covered here:

* Sentence continuations remain merged across page boundaries.
* List metadata (``list_kind``) survives the refactor.
* Split metrics – notably ``soft_limit_hits`` – are identical between the two
  implementations.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from pathlib import Path

import pytest

from pdf_chunker.adapters.io_pdf import read
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.chunk_options import SplitOptions
from pdf_chunker.passes.chunk_pipeline import (
    attach_headings as attach_headings_pipeline,
)
from pdf_chunker.passes.chunk_pipeline import (
    chunk_records as chunk_records_pipeline,
)
from pdf_chunker.passes.chunk_pipeline import (
    iter_blocks as iter_blocks_pipeline,
)
from pdf_chunker.passes.chunk_pipeline import (
    merge_adjacent_blocks as merge_adjacent_blocks_pipeline,
)
from pdf_chunker.passes.split_semantic import (
    DEFAULT_SPLITTER,
    _block_text,
    _collapse_records,
    _get_split_fn,
    _inject_continuation_context,
    _is_heading,
    _merge_blocks,
    _merge_heading_texts,
    _merge_record_block,
    _merge_styled_list_records,
    _restore_overlap_words,
    _split_inline_heading_records,
    _stitch_block_continuations,
    build_chunk,
    build_chunk_with_meta,
)
from pdf_chunker.strategies.bullets import (
    BulletHeuristicStrategy,
    default_bullet_strategy,
)

_BULLET_HEURISTICS: BulletHeuristicStrategy = (
    DEFAULT_SPLITTER.bullet_strategy or default_bullet_strategy()
)


def _manual_pipeline(doc: dict) -> tuple[list[dict], dict[str, int]]:
    """Compose the functional chunk pipeline mirroring ``_SplitSemanticPass``."""

    options = SplitOptions.from_base(
        DEFAULT_SPLITTER.chunk_size,
        DEFAULT_SPLITTER.overlap,
        DEFAULT_SPLITTER.min_chunk_size,
    )
    split_fn, metric_fn = _get_split_fn(options.chunk_size, options.overlap, options.min_chunk_size)
    limit = options.compute_limit()
    records = merge_adjacent_blocks_pipeline(
        iter_blocks_pipeline(doc),
        text_of=_block_text,
        fold=_merge_blocks,
        split_fn=split_fn,
    )
    styled = _split_inline_heading_records(records)
    headed = attach_headings_pipeline(
        styled,
        is_heading=_is_heading,
        merge_block_text=_merge_heading_texts,
    )
    merged_lists = _merge_styled_list_records(headed)
    stitched = _stitch_block_continuations(merged_lists, limit)
    collapsed = _collapse_records(stitched, options, limit)
    build_meta = partial(
        build_chunk_with_meta,
        filename=doc.get("source_path"),
        bullet_strategy=DEFAULT_SPLITTER.bullet_strategy,
    )
    base_chunks = chunk_records_pipeline(
        collapsed,
        generate_metadata=DEFAULT_SPLITTER.generate_metadata,
        build_plain=build_chunk,
        build_with_meta=build_meta,
    )
    overlap = options.overlap if options is not None else DEFAULT_SPLITTER.overlap
    items = list(_inject_continuation_context(base_chunks, limit, overlap))
    return items, {"chunks": len(items), **metric_fn()}


def _legacy_chunks(doc: dict) -> tuple[list[dict], dict[str, int]]:
    """Run the registered split pass and expose its metrics."""

    artifact = DEFAULT_SPLITTER(Artifact(payload=doc))
    items = artifact.payload["items"]
    metrics = (artifact.meta or {}).get("metrics", {}).get("split_semantic", {})
    return items, {k: int(v) for k, v in metrics.items() if isinstance(v, int)}


def _texts(items: Iterable[dict]) -> list[str]:
    return [item.get("text", "") for item in items]


def _metas(items: Iterable[dict]) -> list[dict]:
    return [item.get("meta", {}) for item in items]


def _chunk_list_kinds(items: Iterable[dict]) -> list[str]:
    return [
        meta.get("list_kind")
        for meta in _metas(items)
        if isinstance(meta.get("list_kind"), str) and meta.get("list_kind")
    ]


def _textual_list_kind(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    heuristics = _BULLET_HEURISTICS
    if any(heuristics.starts_with_bullet(line) for line in lines):
        return "bullet"
    if any(heuristics.starts_with_number(line) for line in lines):
        return "numbered"
    return None


def _bullet_fragment_positions(items: Iterable[dict]) -> list[int]:
    texts = _texts(items)
    return [
        index
        for index, (current, following) in enumerate(zip(texts, texts[1:]))
        if _BULLET_HEURISTICS.is_bullet_fragment(current, following)
    ]


def _bullet_continuation_positions(items: Iterable[dict]) -> list[int]:
    texts = _texts(items)
    return [
        index
        for index, (current, following) in enumerate(zip(texts, texts[1:]))
        if _BULLET_HEURISTICS.is_bullet_continuation(current, following)
    ]


def _find_chunk(items: Iterable[dict], *needles: str) -> dict:
    """Return the first chunk whose text contains all ``needles``."""

    match = next(
        (item for item in items if all(needle in item.get("text", "") for needle in needles)),
        None,
    )
    if match is None:
        joined = ", ".join(needles) or "<any>"
        raise AssertionError(f"expected chunk containing: {joined}")
    return match


def _find_chunk_text(items: Iterable[dict], *needles: str) -> str:
    """Return the text of the first chunk containing all ``needles``."""

    return _find_chunk(items, *needles).get("text", "")


def _pdf(path: str) -> dict:
    return read(path)


def test_merge_heading_texts_inserts_blank_line() -> None:
    headings = ("CHAPTER 1", "Why Platform Engineering Is Becoming Essential")
    body = "Platform teams accelerate delivery."
    merged = _merge_heading_texts(headings, body)
    assert (
        merged
        == "CHAPTER 1\nWhy Platform Engineering Is Becoming Essential\nPlatform teams accelerate delivery."
    )


def test_split_inline_heading_records_promotes_styled_list() -> None:
    block = {
        "text": "Focus Body text remains intact for testing scenarios.",
        "type": "paragraph",
        "inline_styles": [{"start": 0, "end": 5, "style": "italic"}],
    }
    records = list(_split_inline_heading_records([(1, block, block["text"])]))

    assert len(records) == 2
    _, heading_block, heading_text = records[0]
    assert heading_block["type"] == "heading"
    assert heading_text == "Focus"

    _, body_block, body_text = records[1]
    assert body_block["type"] == "list_item"
    assert body_block.get("list_kind") == "styled"
    assert body_text.startswith("Body text remains")


@pytest.mark.usefixtures("_nltk_data")
def test_platform_eng_parity() -> None:
    pytest.importorskip("fitz")
    doc = _pdf(str(Path("platform-eng-excerpt.pdf")))

    legacy_items, legacy_metrics = _legacy_chunks(doc)
    refactored_items, refactored_metrics = _manual_pipeline(doc)

    assert _texts(refactored_items) == _texts(legacy_items)
    assert _metas(refactored_items) == _metas(legacy_items)
    assert refactored_metrics == legacy_metrics

    continuation = "ownership of operating the application's infrastructure"
    assert any(continuation in chunk for chunk in _texts(legacy_items))

    # Headings with has_heading_prefix now start new chunks, so "Wrapping Up"
    # appears at the start of its own chunk rather than merged after "Chapter 10."
    wrapping_up_heading = "Wrapping Up\n"
    assert any(chunk.startswith(wrapping_up_heading) for chunk in _texts(legacy_items))
    assert any(chunk.startswith(wrapping_up_heading) for chunk in _texts(refactored_items))

    list_meta = [meta for meta in _metas(legacy_items) if meta.get("list_kind")]
    assert list_meta, "expected list metadata to be present"
    assert list_meta == [meta for meta in _metas(refactored_items) if meta.get("list_kind")]

    leverage_chunk = _find_chunk(refactored_items, "Leverage", "Core to the value")
    leverage_text = leverage_chunk["text"]
    assert leverage_text == _find_chunk_text(legacy_items, "Leverage", "Core to the value")
    assert "Leverage\nCore to the value" in leverage_text
    assert "Leverage\n\nCore to the value" not in leverage_text

    platform_chunk = _find_chunk(
        refactored_items,
        "Platform",
        "We use Evan Bottcher's definition",
    )
    assert platform_chunk["meta"].get("list_kind")
    platform_text = platform_chunk["text"]
    assert "Platform\nWe use" in platform_text
    assert "Platform\n\nWe use" not in platform_text

    list_chunk = next(
        text
        for text in _texts(refactored_items)
        if "Treat building blocks as foundational." in text
    )
    assert "Blocks are composable." in list_chunk
    assert "\n\nBlocks are composable." in list_chunk


def test_sample_book_list_metadata() -> None:
    pytest.importorskip("fitz")
    doc = _pdf(str(Path("sample_book-bullets.pdf")))

    legacy_items, legacy_metrics = _legacy_chunks(doc)
    refactored_items, refactored_metrics = _manual_pipeline(doc)

    assert _texts(refactored_items) == _texts(legacy_items)
    assert _metas(refactored_items) == _metas(legacy_items)
    assert refactored_metrics == legacy_metrics

    legacy_kinds = _chunk_list_kinds(legacy_items)
    refactored_kinds = _chunk_list_kinds(refactored_items)

    assert refactored_kinds, "expected list metadata to survive"
    assert legacy_kinds == refactored_kinds
    assert set(refactored_kinds) == {"bullet"}
    assert all(
        meta.get("block_type") == "list_item"
        for meta in _metas(refactored_items)
        if meta.get("list_kind")
    )
    assert all(
        _textual_list_kind(item.get("text", "")) == meta.get("list_kind")
        for item, meta in zip(refactored_items, _metas(refactored_items))
        if meta.get("list_kind") in {"bullet", "numbered"}
    )


def test_sample_book_bullet_fragments_respect_strategy() -> None:
    """Bullet fragment/continuation pairs remain aligned with default markers."""

    pytest.importorskip("fitz")
    doc = _pdf(str(Path("sample_book-bullets.pdf")))

    legacy_items, _ = _legacy_chunks(doc)
    refactored_items, _ = _manual_pipeline(doc)

    legacy_fragments = _bullet_fragment_positions(legacy_items)
    refactored_fragments = _bullet_fragment_positions(refactored_items)
    assert refactored_fragments == legacy_fragments

    legacy_continuations = _bullet_continuation_positions(legacy_items)
    refactored_continuations = _bullet_continuation_positions(refactored_items)
    assert refactored_continuations == legacy_continuations

    bullet_markers = {
        line.lstrip()[:1]
        for text in _texts(refactored_items)
        for line in text.splitlines()
        if line.strip() and _BULLET_HEURISTICS.starts_with_bullet(line)
    }
    assert bullet_markers
    assert bullet_markers <= set(_BULLET_HEURISTICS.bullet_chars)


@pytest.mark.usefixtures("_nltk_data")
def test_platform_eng_figure_caption_retains_label() -> None:
    pytest.importorskip("fitz")
    doc = _pdf(str(Path("platform-eng-excerpt.pdf")))

    items, _ = _legacy_chunks(doc)
    texts = [item.get("text", "") for item in items]

    caption = "Figure 1-1. The over-general swamp, held together by glue"
    assert any(caption in text for text in texts)

    truncated = "The over-general swamp, held together by glue"
    offenders = [text for text in texts if text.startswith(truncated) and caption not in text]
    assert not offenders, "caption should retain its figure label"
    starters = [text for text in texts if text.lstrip().startswith("Figure 1-1.")]
    assert not starters, "caption should not start a fresh chunk"
    combined = "seen in Figure 1-1.\n\nFigure 1-1. The over-general swamp"
    assert any(combined in text for text in texts), "caption should follow its callout"


def test_restore_overlap_words_prefers_minimal_prefix() -> None:
    chunks = [
        "A car-load of drovers and their wives",  # previous chunk tail
        "their wives kept singing through the town",  # leading words missing "and"
    ]
    restored = _restore_overlap_words(chunks, overlap=3)
    assert restored[1].startswith("and their wives kept"), restored[1]
    assert "their wives their wives" not in restored[1]


def test_restore_overlap_words_drops_duplicate_prefix() -> None:
    chunks = [
        "A car-load of drovers, too, in the midst, on a level with their droves now.",
        (
            "A car-load of drovers, too, in the midst, on a level with their droves now. "
            "But their dogs, where are they"
        ),
    ]
    restored = _restore_overlap_words(chunks, overlap=8)
    assert "But their dogs, where are they" in restored[1]
    assert restored[1].count("A car-load of drovers") <= 1


def test_merge_record_block_preserves_list_kind_in_mixed_merge() -> None:
    list_block = {"type": "list_item", "text": "• First", "list_kind": "bullet"}
    paragraph_block = {"type": "paragraph", "text": "Second"}
    records = [
        (1, list_block, list_block["text"]),
        (1, paragraph_block, paragraph_block["text"]),
    ]
    merged = _merge_record_block(records, "\n\n".join(block for _, _, block in records))
    assert merged.get("type") == "paragraph"
    assert merged.get("list_kind") == "bullet"


def test_heading_stays_with_section_content() -> None:
    """Headings must appear at the start of a chunk WITH their section content.

    When a heading is detected (via has_heading_prefix), it should start a new
    chunk, but the chunk must also contain the section content following the
    heading, not just the heading text alone.
    """
    heading = {"type": "paragraph", "text": "Some Title", "has_heading_prefix": True}
    content = {"type": "paragraph", "text": "This is the section content that follows."}
    records = [
        (1, heading, "Some Title\nThis is the section content that follows."),
        (1, content, "More content in another block."),
    ]

    # The heading-prefixed block should contain both heading and content
    text = records[0][2]
    assert text.startswith("Some Title\n"), "Heading should be at start"
    assert "section content" in text, "Section content must be in same chunk as heading"
    assert len(text) > 50, "Heading chunk should not be tiny"
