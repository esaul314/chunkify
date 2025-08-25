"""Split ``page_blocks`` into canonical ``chunks``.

This pass wraps the legacy :mod:`pdf_chunker.splitter` semantic chunker
while keeping a pure function boundary. When the splitter cannot be
imported, each block becomes a single chunk. Chunks carry page and source
metadata so downstream passes can enrich and emit JSONL rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from functools import partial
from itertools import chain
from typing import Any, ClassVar

from pdf_chunker.framework import Artifact, register
from pdf_chunker.utils import _build_metadata


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


def _is_heading(block: Block) -> bool:
    """Return ``True`` when ``block`` represents a heading."""

    return block.get("type") == "heading"


def _merge_headings(seq: Iterator[tuple[int, Block, str]]) -> Iterator[tuple[int, Block, str]]:
    """Attach consecutive headings to the following block and drop trailing ones."""

    it = iter(seq)
    for page, block, text in it:
        if not _is_heading(block):
            yield page, block, text
            continue

        pages = [page]
        texts = [text]
        for page, block, text in it:
            if _is_heading(block):
                pages.append(page)
                texts.append(text)
                continue
            merged_text = "\n".join(chain(texts, [text])).strip()
            yield pages[0], {**block}, merged_text
            break
        else:
            return


def _with_source(block: Block, page: int, filename: str | None) -> Block:
    """Attach ``filename`` and ``page`` as a ``source`` entry when absent."""

    existing = block.get("source") or {}
    source = {**{"filename": filename, "page": page}, **existing}
    return {**block, "source": {k: v for k, v in source.items() if v is not None}}


def build_chunk(text: str) -> Chunk:
    """Return chunk payload containing only ``text``."""

    return {"text": text}


def build_chunk_with_meta(
    text: str, block: Block, page: int, filename: str | None, index: int
) -> Chunk:
    """Return chunk payload enriched with metadata."""
    return {
        "text": text,
        "meta": _build_metadata(text, _with_source(block, page, filename), index, {}),
    }


def _chunk_items(doc: Doc, split_fn: SplitFn, generate_metadata: bool = True) -> Iterator[Chunk]:
    """Yield chunk records from ``doc`` using ``split_fn``."""

    filename = doc.get("source_path")
    merged = _merge_headings(_block_texts(doc, split_fn))
    return (
        {
            "id": str(i),
            **(
                build_chunk_with_meta(text, block, page, filename, i)
                if generate_metadata
                else build_chunk(text)
            ),
        }
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


def _resolve_opts(
    meta: dict[str, Any] | None, defaults: _SplitSemanticPass
) -> tuple[int, int, int]:
    """Return ``chunk_size``, ``overlap``, ``min_chunk_size`` from meta overrides."""
    opts = ((meta or {}).get("options") or {}).get("split_semantic", {})
    chunk = int(opts.get("chunk_size", defaults.chunk_size))
    overlap = int(opts.get("overlap", defaults.overlap))
    calc = max(8, chunk // 10)
    explicit = (
        defaults.min_chunk_size is not None
        and defaults.min_chunk_size <= max(8, defaults.chunk_size // 10)
        and "chunk_size" not in opts
    )
    min_size = defaults.min_chunk_size if explicit else calc
    return chunk, overlap, min_size


@dataclass(frozen=True)
class _SplitSemanticPass:
    name: ClassVar[str] = "split_semantic"
    input_type: ClassVar[type] = dict  # expects {"type": "page_blocks"}
    output_type: ClassVar[type] = dict  # returns {"type": "chunks", "items": [...]}
    chunk_size: int = 400
    overlap: int = 50
    min_chunk_size: int | None = None
    generate_metadata: bool = True

    def __post_init__(self) -> None:
        if self.min_chunk_size is None:
            object.__setattr__(self, "min_chunk_size", max(8, self.chunk_size // 10))

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload
        if not isinstance(doc, dict) or doc.get("type") != "page_blocks":
            return a
        chunk_size, overlap, min_chunk_size = _resolve_opts(a.meta, self)
        split_fn, metric_fn = _get_split_fn(chunk_size, overlap, min_chunk_size)
        items = list(_chunk_items(doc, split_fn, self.generate_metadata))
        meta = _update_meta(a.meta, len(items), metric_fn())
        return Artifact(payload={"type": "chunks", "items": items}, meta=meta)


DEFAULT_SPLITTER = _SplitSemanticPass()


def make_splitter(**opts: Any) -> _SplitSemanticPass:
    """Return a configured ``split_semantic`` pass from ``opts``."""
    opts_map = {
        "chunk_size": int(opts.get("chunk_size", DEFAULT_SPLITTER.chunk_size)),
        "overlap": int(opts.get("overlap", DEFAULT_SPLITTER.overlap)),
        "generate_metadata": bool(
            opts.get("generate_metadata", DEFAULT_SPLITTER.generate_metadata)
        ),
    }
    return replace(DEFAULT_SPLITTER, **opts_map)


split_semantic = register(make_splitter())
