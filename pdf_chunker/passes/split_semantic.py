"""Split ``page_blocks`` into canonical ``chunks``.

This pass wraps the legacy :mod:`pdf_chunker.splitter` semantic chunker
while keeping a pure function boundary. When the splitter cannot be
imported, each block becomes a single chunk. Chunks carry page and source
metadata so downstream passes can enrich and emit JSONL rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import partial
from typing import Any

from pdf_chunker.framework import Artifact, register


def _soft_truncate(text: str, max_size: int = 8_000) -> str:
    """Truncate ``text`` to ``max_size`` characters using simple heuristics."""
    if len(text) <= max_size:
        return text
    cut = text.rfind(" ", 0, max_size - 100)
    return text[: cut if cut > 0 else max_size - 100].strip()


Doc = dict[str, Any]
Block = dict[str, Any]
Chunk = dict[str, Any]
SplitFn = Callable[[str], list[str]]
MetricFn = Callable[[], dict[str, int | bool]]


def _get_split_fn(chunk_size: int, overlap: int, min_chunk_size: int) -> tuple[SplitFn, MetricFn]:
    """Return a semantic splitter enforcing size limits and collecting metrics."""

    soft_hits = 0
    hard_hit = False

    try:
        from pdf_chunker.splitter import semantic_chunker

        semantic = partial(
            semantic_chunker,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

        def split(text: str) -> list[str]:
            nonlocal soft_hits, hard_hit
            hard_hit |= len(text) > 25_000
            raw = semantic(text[:25_000])
            soft_hits += sum(len(c) > 8_000 for c in raw)
            return [_soft_truncate(c) for c in raw if c]

    except Exception:  # pragma: no cover - safety fallback

        def split(text: str) -> list[str]:
            nonlocal soft_hits, hard_hit
            hard_hit |= len(text) > 25_000
            truncated = text[:25_000]
            soft_hits += int(len(truncated) > 8_000)
            return [_soft_truncate(truncated)] if truncated else []

    metrics = lambda: {"soft_limit_hits": soft_hits, "hard_limit_hit": hard_hit}
    return split, metrics


def _iter_blocks(doc: Doc) -> Iterable[tuple[int, Block]]:
    """Yield ``(page_number, block)`` pairs from a document."""

    return (
        (page.get("page", i + 1), block)
        for i, page in enumerate(doc.get("pages", []))
        for block in page.get("blocks", [])
    )


def _block_texts(doc: Doc, split_fn: SplitFn) -> Iterator[tuple[int, Block, str]]:
    """Yield ``(page, block, text)`` triples from a document."""

    return (
        (page, block, text)
        for page, block in _iter_blocks(doc)
        for text in split_fn(block.get("text", ""))
        if text
    )


def _merge_headings(
    seq: Iterator[tuple[int, Block, str]]
) -> Iterator[tuple[int, Block, str]]:
    """Attach standalone heading blocks to the following block.

    When a block is marked as a ``heading`` it should not form its own chunk
    unless it is the last block. This generator buffers a pending heading and
    prepends it to the text of the next block, mirroring the legacy
    :func:`pdf_chunker.splitter._fix_heading_splitting_issues` behaviour.
    """

    pending: tuple[int, Block, str] | None = None

    for page, block, text in seq:
        if pending:
            h_page, h_block, h_text = pending
            merged_text = f"{h_text}\n{text}".strip()
            merged_block = {**block}
            yield h_page, merged_block, merged_text
            pending = None
            continue

        if block.get("type") == "heading":
            pending = (page, block, text)
            continue

        yield page, block, text

    if pending:
        yield pending


def _list_meta(block: Block) -> dict[str, str]:
    return (
        {"list_kind": block["list_kind"]}
        if block.get("type") == "list_item" and block.get("list_kind")
        else {}
    )


def _chunk_meta(page: int, block: Block, source: str | None) -> dict[str, Any]:
    base = {k: v for k, v in {"page": page, "source": source}.items() if v is not None}
    block_meta = block.get("meta") if isinstance(block.get("meta"), dict) else {}
    return {**base, **block_meta, **_list_meta(block)}


def _chunk_items(doc: Doc, split_fn: SplitFn) -> Iterator[Chunk]:
    """Yield chunk records from ``doc`` using ``split_fn``."""

    source = doc.get("source_path")
    merged = _merge_headings(_block_texts(doc, split_fn))
    return (
        {"id": str(i), "text": text, "meta": _chunk_meta(page, block, source)}
        for i, (page, block, text) in enumerate(merged)
    )


def _update_meta(
    meta: dict[str, Any] | None, count: int, extra: dict[str, int | bool]
) -> dict[str, Any]:
    metrics = {**{"chunks": count}, **extra}
    existing = ((meta or {}).get("metrics") or {}).get("split_semantic", {})
    merged_metrics = {**existing, **metrics}
    existing_metrics = (meta or {}).get("metrics") or {}
    return {
        **(meta or {}),
        "metrics": {**existing_metrics, "split_semantic": merged_metrics},
    }


class _SplitSemanticPass:
    name = "split_semantic"
    input_type = dict  # expects {"type": "page_blocks"}
    output_type = dict  # returns {"type": "chunks", "items": [...]}

    def __init__(
        self, chunk_size: int = 400, overlap: int = 50, min_chunk_size: int | None = None
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size or max(8, chunk_size // 10)

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload
        if not isinstance(doc, dict) or doc.get("type") != "page_blocks":
            return a
        split_fn, metric_fn = _get_split_fn(self.chunk_size, self.overlap, self.min_chunk_size)
        items = list(_chunk_items(doc, split_fn))
        meta = _update_meta(a.meta, len(items), metric_fn())
        return Artifact(payload={"type": "chunks", "items": items}, meta=meta)


split_semantic = register(_SplitSemanticPass())
