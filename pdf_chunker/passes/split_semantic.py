"""Split ``page_blocks`` into canonical ``chunks``.

This pass wraps the legacy :mod:`pdf_chunker.splitter` semantic chunker
while keeping a pure function boundary. When the splitter cannot be
imported, each block becomes a single chunk. Chunks carry page and source
metadata so downstream passes can enrich and emit JSONL rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pdf_chunker.framework import Artifact, register

Doc = dict[str, Any]
Block = dict[str, Any]
Chunk = dict[str, Any]
SplitFn = Callable[[str], list[str]]


def _get_split_fn() -> SplitFn:
    """Return the semantic splitter or a block-level fallback."""

    try:
        from pdf_chunker.splitter import semantic_chunker

        return lambda text: semantic_chunker(text) or ([text] if text else [])
    except Exception:  # pragma: no cover - safety fallback
        return lambda text: [text] if text else []


def _iter_blocks(doc: Doc) -> Iterable[tuple[int, Block]]:
    """Yield ``(page_number, block)`` pairs from a document."""

    return (
        (page.get("page", i + 1), block)
        for i, page in enumerate(doc.get("pages", []))
        for block in page.get("blocks", [])
    )


def _chunk_meta(page: int, block: Block, source: str | None) -> dict[str, Any]:
    base = {"page": page}
    if source is not None:
        base["source"] = source
    if (block_meta := block.get("meta")) and isinstance(block_meta, dict):
        base.update(block_meta)
    return base


def _chunk_items(doc: Doc, split_fn: SplitFn) -> list[Chunk]:
    source = doc.get("source_path")
    seq = (
        (page, block, text)
        for page, block in _iter_blocks(doc)
        for text in split_fn(block.get("text", ""))
    )
    return [
        {"id": str(i), "text": text, "meta": _chunk_meta(page, block, source)}
        for i, (page, block, text) in enumerate(seq)
        if text
    ]


def _update_meta(meta: dict[str, Any] | None, count: int) -> dict[str, Any]:
    base = dict(meta or {})
    metrics = base.setdefault("metrics", {}).setdefault("split_semantic", {})
    metrics["chunks"] = count
    return base


class _SplitSemanticPass:
    name = "split_semantic"
    input_type = dict  # expects {"type": "page_blocks"}
    output_type = dict  # returns {"type": "chunks", "items": [...]}

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload
        if not isinstance(doc, dict) or doc.get("type") != "page_blocks":
            return a
        split_fn = _get_split_fn()
        items = _chunk_items(doc, split_fn)
        meta = _update_meta(a.meta, len(items))
        return Artifact(payload={"type": "chunks", "items": items}, meta=meta)


split_semantic = register(_SplitSemanticPass())

