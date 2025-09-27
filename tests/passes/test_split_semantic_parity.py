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

from functools import partial
from pathlib import Path
from typing import Iterable

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
    _inject_continuation_context,
    _merge_blocks,
    _merge_record_block,
    _merge_heading_texts,
    _restore_overlap_words,
    _stitch_block_continuations,
    _get_split_fn,
    _is_heading,
    build_chunk,
    build_chunk_with_meta,
)


def _manual_pipeline(doc: dict) -> tuple[list[dict], dict[str, int]]:
    """Compose the functional chunk pipeline mirroring ``_SplitSemanticPass``."""

    options = SplitOptions.from_base(
        DEFAULT_SPLITTER.chunk_size,
        DEFAULT_SPLITTER.overlap,
        DEFAULT_SPLITTER.min_chunk_size,
    )
    split_fn, metric_fn = _get_split_fn(
        options.chunk_size, options.overlap, options.min_chunk_size
    )
    limit = options.compute_limit()
    records = merge_adjacent_blocks_pipeline(
        iter_blocks_pipeline(doc),
        text_of=_block_text,
        fold=_merge_blocks,
        split_fn=split_fn,
    )
    headed = attach_headings_pipeline(
        records,
        is_heading=_is_heading,
        merge_block_text=_merge_heading_texts,
    )
    stitched = _stitch_block_continuations(headed, limit)
    collapsed = _collapse_records(stitched, options, limit)
    build_meta = partial(
        build_chunk_with_meta,
        filename=doc.get("source_path"),
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


def _pdf(path: str) -> dict:
    return read(path)


def test_merge_heading_texts_inserts_blank_line() -> None:
    headings = ("CHAPTER 1", "Why Platform Engineering Is Becoming Essential")
    body = "Platform teams accelerate delivery."
    merged = _merge_heading_texts(headings, body)
    assert (
        merged
        == "CHAPTER 1\nWhy Platform Engineering Is Becoming Essential\n\nPlatform teams accelerate delivery."
    )


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

    list_meta = [meta for meta in _metas(legacy_items) if meta.get("list_kind")]
    assert list_meta, "expected list metadata to be present"
    assert list_meta == [
        meta for meta in _metas(refactored_items) if meta.get("list_kind")
    ]


def test_sample_book_list_metadata() -> None:
    pytest.importorskip("fitz")
    doc = _pdf(str(Path("sample_book-bullets.pdf")))

    legacy_items, legacy_metrics = _legacy_chunks(doc)
    refactored_items, refactored_metrics = _manual_pipeline(doc)

    assert _texts(refactored_items) == _texts(legacy_items)
    assert _metas(refactored_items) == _metas(legacy_items)
    assert refactored_metrics == legacy_metrics

    kinds = {meta.get("list_kind") for meta in _metas(legacy_items) if meta.get("list_kind")}
    assert kinds == {"bullet"}
    assert kinds == {
        meta.get("list_kind")
        for meta in _metas(refactored_items)
        if meta.get("list_kind")
    }


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
