"""Split ``page_blocks`` into canonical ``chunks``.

This pass wraps the legacy :mod:`pdf_chunker.splitter` semantic chunker
while keeping a pure function boundary. When the splitter cannot be
imported, each block becomes a single chunk. Chunks carry page and source
metadata so downstream passes can enrich and emit JSONL rows.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import partial, reduce
from itertools import chain
from typing import Any, TypedDict, cast

from pdf_chunker.framework import Artifact, Pass, register
from pdf_chunker.list_detection import starts_with_bullet, starts_with_number
from pdf_chunker.utils import _build_metadata

SOFT_LIMIT = 8_000


def _soft_segments(text: str, max_size: int = SOFT_LIMIT) -> list[str]:
    """Split ``text`` into segments of at most ``max_size`` characters."""

    def _split(chunk: str) -> Iterator[str]:
        if len(chunk) <= max_size:
            yield chunk.strip()
            return
        cut = chunk.rfind(" ", 0, max_size)
        head = chunk[: cut if cut != -1 else max_size].strip()
        tail = chunk[len(head) :].lstrip()
        yield head
        yield from _split(tail)

    return list(_split(text))


_ENDS_SENTENCE = re.compile(r"[.?!][\"')\]]*\s*$")


def _merge_sentence_fragments(
    chunks: Iterable[str],
    *,
    max_words: int = 80,
    chunk_size: int | None = None,
    overlap: int = 0,
) -> list[str]:
    """Merge trailing fragments until a sentence boundary or limits reached."""

    allowed_overlap = max(overlap, 0)
    limit = max(chunk_size - allowed_overlap, 0) if chunk_size and chunk_size > 0 else None

    def _should_merge(previous: str, current: str, prev_words: list[str]) -> bool:
        """Return ``True`` when ``previous`` and ``current`` should coalesce."""

        if not previous:
            return False
        if _ENDS_SENTENCE.search(previous.rstrip()):
            return False
        lead = current.lstrip()
        if not lead:
            return False
        current_words = current.split()
        first_word = current_words[0] if current_words else ""
        if prev_words and prev_words[-1] == first_word:
            return False
        head = lead[0]
        continuation_chars = ",.;:)]\"'"
        if not (head.islower() or head in continuation_chars):
            return False
        combined = len(previous) + 1 + len(current)
        if len(previous) >= SOFT_LIMIT or len(current) >= SOFT_LIMIT:
            return False
        return combined <= SOFT_LIMIT

      
    def _actual_overlap(
        prev_words: tuple[str, ...],
        current_words: tuple[str, ...],
    ) -> int:
        if not allowed_overlap or not prev_words or not current_words:
            return 0
        window = min(allowed_overlap, len(prev_words), len(current_words))
        return window if window and prev_words[-window:] == current_words[:window] else 0

    def _dedupe_overlap(
        prev_words: tuple[str, ...],
        words: tuple[str, ...],
    ) -> tuple[str, tuple[str, ...]]:
        if not words:
            return "", words
        overlap_words = _actual_overlap(prev_words, words)
        trimmed_words = words[overlap_words:] if overlap_words else words
        return " ".join(trimmed_words), trimmed_words

    def _append(
        acc: list[tuple[str, tuple[str, ...]]],
        text: str,
        words: tuple[str, ...],
    ) -> list[tuple[str, tuple[str, ...]]]:
        return [*acc, (text, words)]

    def _merge(
        acc: list[tuple[str, tuple[str, ...]]],
        chunk: str,
    ) -> list[tuple[str, tuple[str, ...]]]:
        words = tuple(chunk.split())
        if not words:
            return acc

        if not acc:
            return _append(acc, chunk, words)

        prev_text, prev_words = acc[-1]
        trimmed_text, trimmed_words = _dedupe_overlap(prev_words, words)
        if not trimmed_words:
            return _append(acc, chunk, words)

        if not _should_merge(prev_text, trimmed_text, list(prev_words)):
            return _append(acc, chunk, words)

        projected = len(prev_words) + len(trimmed_words)
        if limit is not None and projected > limit:
            return _append(acc, chunk, words)

        merged_text = f"{prev_text} {trimmed_text}".strip()
        merged_words = (*prev_words, *trimmed_words)
        return [*acc[:-1], (merged_text, merged_words)]

    merged: list[tuple[str, tuple[str, ...]]] = reduce(
        _merge,
        chunks,
        [],
    )

    return [text for text, _ in merged]


Doc = dict[str, Any]
Block = dict[str, Any]
Chunk = dict[str, Any]
SplitFn = Callable[[str], list[str]]
MetricFn = Callable[[], dict[str, int | bool]]


class _SplitOpts(TypedDict, total=False):
    chunk_size: int
    overlap: int
    generate_metadata: bool
    min_chunk_size: int


def _derive_min_chunk_size(chunk_size: int, min_size: int | None) -> int:
    """Return ``min_size`` or derive it as a fraction of ``chunk_size``."""
    return min_size if min_size is not None else max(8, chunk_size // 10)


def _get_split_fn(
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> tuple[SplitFn, MetricFn]:
    """Return a semantic splitter enforcing size limits and collecting metrics."""

    soft_hits = 0

    try:
        from pdf_chunker.splitter import (
            iter_word_chunks,
            semantic_chunker,
        )

        semantic = partial(
            semantic_chunker,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

        def split(text: str) -> list[str]:
            """Split ``text`` while guarding against truncation."""

            nonlocal soft_hits
            pieces = semantic(text)
            merged = pieces if sum(len(p.split()) for p in pieces) >= len(text.split()) else [text]

            def _soften(segment: str) -> list[str]:
                nonlocal soft_hits
                splits = _soft_segments(segment)
                if len(splits) > 1:
                    soft_hits += 1
                return splits

            raw = [
                seg
                for c in merged
                for sub in iter_word_chunks(c, chunk_size, overlap)
                for seg in _soften(sub)
            ]
            final = _merge_sentence_fragments(raw, chunk_size=chunk_size, overlap=overlap)
            soft_hits += sum(len(c) > SOFT_LIMIT for c in final)
            return final

    except Exception:  # pragma: no cover - safety fallback

        def split(text: str) -> list[str]:
            nonlocal soft_hits
            raw = _soft_segments(text)
            final = _merge_sentence_fragments(raw, chunk_size=chunk_size, overlap=overlap)
            soft_hits += sum(len(seg) > SOFT_LIMIT for seg in final)
            return final

    def metrics() -> dict[str, int]:
        return {"soft_limit_hits": soft_hits}

    return split, metrics


def _iter_blocks(doc: Doc) -> Iterable[tuple[int, Block]]:
    """Yield ``(page_number, block)`` pairs from a document."""

    return (
        (page.get("page", i + 1), block)
        for i, page in enumerate(doc.get("pages", []))
        for block in page.get("blocks", [])
    )


def _block_texts(doc: Doc, split_fn: SplitFn) -> Iterator[tuple[int, Block, str]]:
    """Yield ``(page, block, text)`` triples after merging sentence fragments."""

    def _merge(
        acc: list[tuple[int, Block, str]],
        cur: tuple[int, Block, str],
    ) -> list[tuple[int, Block, str]]:
        page, block, text = cur
        if not acc:
            return acc + [cur]
        prev_page, prev_block, prev_text = acc[-1]
        if prev_page != page:
            return acc + [cur]
        if _is_heading(prev_block) or _is_heading(block):
            return acc + [cur]
        if not _ENDS_SENTENCE.search(prev_text.rstrip()) or text[:1].islower():
            acc[-1] = (
                prev_page,
                prev_block,
                f"{prev_text} {text}".strip(),
            )
            return acc
        return acc + [cur]

    merged: list[tuple[int, Block, str]] = reduce(
        _merge,
        ((p, b, b.get("text", "")) for p, b in _iter_blocks(doc)),
        cast(list[tuple[int, Block, str]], []),
    )

    return ((page, block, text) for page, block, raw in merged for text in split_fn(raw) if text)


def _is_heading(block: Block) -> bool:
    """Return ``True`` when ``block`` represents a heading."""

    return block.get("type") == "heading"


def _infer_list_kind(text: str) -> str | None:
    if starts_with_bullet(text):
        return "bullet"
    if starts_with_number(text):
        return "numbered"
    return None


def _tag_list(block: Block) -> Block:
    if block.get("type") == "list_item" and block.get("list_kind"):
        return block
    kind = _infer_list_kind(block.get("text", ""))
    return {**block, "type": "list_item", "list_kind": kind} if kind else block


def _merge_headings(
    seq: Iterator[tuple[int, Block, str]],
) -> Iterator[tuple[int, Block, str]]:
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
    annotated = _tag_list(block)
    return {
        "text": text,
        "meta": _build_metadata(
            text,
            _with_source(annotated, page, filename),
            index,
            {},
        ),
    }


def _chunk_items(
    doc: Doc,
    split_fn: SplitFn,
    generate_metadata: bool = True,
) -> Iterator[Chunk]:
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
    meta: Mapping[str, Any] | None, base: _SplitSemanticPass
) -> tuple[int, int, int]:  # noqa: E501
    """Return ``chunk_size``, ``overlap``, and ``min_chunk_size`` from ``meta``."""

    opts = ((meta or {}).get("options") or {}).get("split_semantic", {})
    chunk = int(opts.get("chunk_size", base.chunk_size))
    overlap = int(opts.get("overlap", base.overlap))
    min_size = (
        int(opts["min_chunk_size"])
        if "min_chunk_size" in opts
        else (
            base.min_chunk_size
            if "chunk_size" not in opts
            else _derive_min_chunk_size(chunk, None)  # noqa: E501
        )
    )
    return chunk, overlap, cast(int, min_size)


@dataclass
class _SplitSemanticPass:
    name: str = field(default="split_semantic", init=False)
    input_type: type = field(
        default=dict, init=False
    )  # expects {"type": "page_blocks"}  # noqa: E501
    output_type: type = field(
        default=dict, init=False
    )  # returns {"type": "chunks", "items": [...]}
    chunk_size: int = 400
    overlap: int = 50
    min_chunk_size: int | None = None
    generate_metadata: bool = True

    def __post_init__(self) -> None:
        self.min_chunk_size = _derive_min_chunk_size(
            self.chunk_size, self.min_chunk_size
        )  # noqa: E501

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
    opts_map: _SplitOpts = {
        "chunk_size": int(opts.get("chunk_size", DEFAULT_SPLITTER.chunk_size)),
        "overlap": int(opts.get("overlap", DEFAULT_SPLITTER.overlap)),
        "generate_metadata": bool(
            opts.get("generate_metadata", DEFAULT_SPLITTER.generate_metadata)
        ),
    }
    base = replace(DEFAULT_SPLITTER, **opts_map)
    if "chunk_size" in opts and "min_chunk_size" not in opts:
        base = replace(base, min_chunk_size=None)
    base.__post_init__()
    return base


split_semantic: Pass = register(make_splitter())
