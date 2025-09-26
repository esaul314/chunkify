"""Split ``page_blocks`` into canonical ``chunks``.

This pass wraps the legacy :mod:`pdf_chunker.splitter` semantic chunker
while keeping a pure function boundary. When the splitter cannot be
imported, each block becomes a single chunk. Chunks carry page and source
metadata so downstream passes can enrich and emit JSONL rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import partial, reduce
from itertools import chain
from math import ceil
from typing import Any, TypedDict

from pdf_chunker.framework import Artifact, Pass, register
from pdf_chunker.list_detection import starts_with_bullet, starts_with_number
from pdf_chunker.passes.chunk_options import (
    SplitMetrics,
    SplitOptions,
    derive_min_chunk_size,
)
from pdf_chunker.passes.chunk_pipeline import (
    attach_headings as pipeline_attach_headings,
)
from pdf_chunker.passes.chunk_pipeline import (
    chunk_records as pipeline_chunk_records,
)
from pdf_chunker.passes.chunk_pipeline import (
    iter_blocks as pipeline_iter_blocks,
)
from pdf_chunker.passes.chunk_pipeline import (
    merge_adjacent_blocks as pipeline_merge_adjacent_blocks,
)
from pdf_chunker.passes.sentence_fusion import (
    _AVERAGE_CHARS_PER_TOKEN,
    _ENDS_SENTENCE,
    SOFT_LIMIT,
    _is_continuation_lead,
    _last_sentence,
    _merge_sentence_fragments,
)
from pdf_chunker.text_cleaning import STOPWORDS
from pdf_chunker.utils import _build_metadata

_STOPWORD_TITLES = frozenset(word.title() for word in STOPWORDS)
_FOOTNOTE_TAILS = {"", ".", ",", ";", ":"}


def _collect_superscripts(
    block: Block, text: str
) -> tuple[list[dict[str, str]], tuple[tuple[int, int], ...]]:
    if not text:
        return [], ()
    limit = len(text)

    def _normalize(span: Any) -> tuple[dict[str, str], tuple[int, int]] | None:
        if getattr(span, "style", None) != "superscript":
            return None
        start = max(0, min(limit, getattr(span, "start", 0)))
        end = max(start, min(limit, getattr(span, "end", start)))
        raw = text[start:end]
        snippet = raw.strip()
        if not snippet or text[end:].strip() not in _FOOTNOTE_TAILS:
            return None
        attrs = getattr(span, "attrs", None)
        note_id = attrs.get("note_id") if isinstance(attrs, Mapping) else None
        focus = raw.find(snippet)
        span_start = start + (focus if focus >= 0 else 0)
        span_end = span_start + len(snippet)
        public = {"text": snippet, **({"note_id": note_id} if note_id else {})}
        return public, (span_start, span_end)

    entries = tuple(
        entry for entry in (_normalize(span) for span in block.get("inline_styles") or ()) if entry
    )
    anchors = [public for public, _ in entries]
    spans = tuple(span for _, span in entries if span[0] < span[1])
    return anchors, spans


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


def _restore_overlap_words(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return chunks
    restored: list[str] = []
    previous: tuple[str, ...] = ()

    for chunk in chunks:
        words = tuple(chunk.split())
        if previous:
            window = min(overlap, len(previous))
            prefix = words[:window]
            overlap_words = previous[-window:]
            if overlap_words and tuple(prefix) != overlap_words:
                words = (*overlap_words, *words)
                chunk = " ".join(words)
        restored.append(chunk)
        previous = words

    return restored


def _stitch_block_continuations(
    seq: Iterable[tuple[int, Block, str]], limit: int | None
) -> list[tuple[int, Block, str]]:
    def _consume(
        acc: list[tuple[int, Block, str]],
        cur: tuple[int, Block, str],
    ) -> list[tuple[int, Block, str]]:
        page, block, text = cur
        if not acc:
            return [*acc, cur]
        lead = text.lstrip()
        if not lead or not _is_continuation_lead(lead):
            return [*acc, cur]
        context = _last_sentence(acc[-1][2])
        if not context or text.lstrip().startswith(context):
            return [*acc, cur]
        context_words = tuple(context.split())
        text_words = tuple(text.split())
        if limit is not None and len(text_words) + len(context_words) > limit:
            return [*acc, cur]
        enriched = f"{context} {text}".strip()
        return [*acc, (page, block, enriched)]

    return reduce(_consume, seq, [])


def _merge_record_block(records: list[tuple[int, Block, str]], text: str) -> Block:
    block = records[0][1]
    base = {k: v for k, v in block.items() if k not in {"text", "list_kind"}}
    block_type = block.get("type")
    normalized_type = block_type if block_type and block_type != "heading" else "paragraph"
    return {**base, "type": normalized_type, "text": text}


def _with_chunk_index(block: Block, index: int) -> Block:
    return {**block, "_chunk_start_index": index}


def _collapse_records(
    records: Iterable[tuple[int, Block, str]],
    options: SplitOptions,
    limit: int | None,
) -> Iterator[tuple[int, Block, str]]:
    seq = list(records)
    if limit is None or limit <= 0:
        for idx, (page, block, text) in enumerate(seq):
            yield page, _with_chunk_index(block, idx), text
        return

    hard_limit = options.chunk_size if options.chunk_size > 0 else None
    buffer: list[tuple[int, Block, str]] = []
    running_words = 0
    running_dense = 0
    start_index: int | None = None

    def _effective_counts(text: str) -> tuple[int, int, int]:
        words = tuple(text.split())
        word_count = len(words)
        char_total = sum(len(token) for token in words)
        dense_total = int(ceil(char_total / _AVERAGE_CHARS_PER_TOKEN)) if char_total else 0
        if word_count <= 1 and text:
            dense_total = max(dense_total, len(text))
        effective_total = max(word_count, dense_total)
        return word_count, dense_total, effective_total

    def emit() -> Iterator[tuple[int, Block, str]]:
        nonlocal buffer, running_words, running_dense, start_index
        if not buffer:
            return
        first_index = start_index if start_index is not None else 0
        if len(buffer) == 1:
            page, block, text = buffer[0]
            yield page, _with_chunk_index(block, first_index), text
        else:
            effective_total = max(running_words, running_dense)
            exceeds = False
            if limit is not None and effective_total > limit:
                exceeds = True
            if not exceeds and hard_limit is not None and effective_total > hard_limit:
                exceeds = True
            if exceeds:
                for offset, (page, block, text) in enumerate(buffer):
                    yield page, _with_chunk_index(block, first_index + offset), text
            else:
                joined = "\n\n".join(part.strip() for _, _, part in buffer if part.strip()).strip()
                if not joined or len(joined) > SOFT_LIMIT:
                    for offset, (page, block, text) in enumerate(buffer):
                        yield page, _with_chunk_index(block, first_index + offset), text
                else:
                    merged = _merge_record_block(buffer, joined)
                    yield buffer[0][0], _with_chunk_index(merged, first_index), joined
        buffer, running_words, running_dense, start_index = [], 0, 0, None

    for idx, record in enumerate(seq):
        page, block, text = record
        if buffer and page != buffer[-1][0]:
            yield from emit()
        word_count, dense_count, effective_count = _effective_counts(text)
        if (limit is not None and effective_count > limit) or (
            hard_limit is not None and effective_count > hard_limit
        ):
            yield from emit()
            yield page, _with_chunk_index(block, idx), text
            continue
        next_is_list = idx + 1 < len(seq) and _starts_list_like(seq[idx + 1][1], seq[idx + 1][2])
        if buffer and _starts_list_like(block, text):
            if not buffer[-1][2].rstrip().endswith(":"):
                yield from emit()
        elif buffer and text.rstrip().endswith(":") and next_is_list:
            yield from emit()
        if buffer:
            projected_words = running_words + word_count
            projected_dense = running_dense + dense_count
            projected_effective = max(projected_words, projected_dense)
            if (limit is not None and projected_effective > limit) or (
                hard_limit is not None and projected_effective > hard_limit
            ):
                yield from emit()
        if not buffer:
            start_index = idx
            running_words, running_dense = word_count, dense_count
        else:
            running_words += word_count
            running_dense += dense_count
        buffer.append((page, block, text))

    yield from emit()


Doc = dict[str, Any]
Block = dict[str, Any]
Chunk = dict[str, Any]
SplitFn = Callable[[str], list[str]]
MetricFn = Callable[[], dict[str, int | bool]]


class _OverrideOpts(TypedDict, total=False):
    chunk_size: int
    overlap: int
    generate_metadata: bool


def _get_split_fn(
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> tuple[SplitFn, MetricFn]:
    """Return a semantic splitter enforcing size limits and collecting metrics."""

    soft_hits = 0

    try:
        from pdf_chunker.splitter import semantic_chunker

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

            raw = [softened for chunk in merged for softened in _soften(chunk)]
            final = _merge_sentence_fragments(
                raw,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min_chunk_size,
            )
            soft_hits += sum(len(c) > SOFT_LIMIT for c in final)
            return _restore_overlap_words(final, overlap)

    except Exception:  # pragma: no cover - safety fallback

        def split(text: str) -> list[str]:
            nonlocal soft_hits
            raw = _soft_segments(text)
            final = _merge_sentence_fragments(
                raw,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min_chunk_size,
            )
            soft_hits += sum(len(seg) > SOFT_LIMIT for seg in final)
            return _restore_overlap_words(final, overlap)

    def metrics() -> dict[str, int]:
        return {"soft_limit_hits": soft_hits}

    return split, metrics


def _block_text(block: Block) -> str:
    return block.get("text", "")


def _starts_list_like(block: Block, text: str) -> bool:
    if block.get("type") == "list_item" and block.get("list_kind"):
        return True
    stripped = text.lstrip()
    return bool(stripped) and (starts_with_bullet(stripped) or starts_with_number(stripped))


def _should_break_after_colon(prev_text: str, block: Block, text: str) -> bool:
    if not prev_text.rstrip().endswith(":"):
        return False
    lead = text.lstrip()
    if not lead:
        return False
    head = lead[0]
    return head.isupper() or head.isdigit() or _starts_list_like(block, text)


def _merge_blocks(
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
    if _starts_list_like(block, text):
        return acc + [cur]
    if _should_break_after_colon(prev_text, block, text):
        return acc + [cur]
    lead = text.lstrip()
    continuation_chars = ",.;:)]\"'"
    prev_ends_sentence = _ENDS_SENTENCE.search(prev_text.rstrip())
    if lead and (
        _is_continuation_lead(lead)
        or (not prev_ends_sentence and (lead[0].islower() or lead[0] in continuation_chars))
    ):
        acc[-1] = (prev_page, prev_block, f"{prev_text} {text}".strip())
        return acc
    return acc + [cur]


def _block_texts(doc: Doc, split_fn: SplitFn) -> Iterator[tuple[int, Block, str]]:
    """Yield ``(page, block, text)`` triples after merging sentence fragments."""

    return pipeline_merge_adjacent_blocks(
        pipeline_iter_blocks(doc),
        text_of=_block_text,
        fold=_merge_blocks,
        split_fn=split_fn,
    )


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


def _normalize_bullet_tail(tail: str) -> str:
    if not tail:
        return ""
    head, *rest = tail.split(" ", 1)
    normalized = head.lower() if head in _STOPWORD_TITLES else head
    return f"{normalized} {rest[0]}".strip() if rest and rest[0] else normalized


def _merge_heading_texts(headings: Iterable[str], body: str) -> str:
    if any(starts_with_bullet(h.lstrip()) for h in headings):
        lead = " ".join(h.rstrip() for h in headings).rstrip()
        tail = _normalize_bullet_tail(body.lstrip()) if body else ""
        return f"{lead} {tail}".strip()
    return "\n".join(chain(headings, [body])).strip()


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
    start_index = annotated.pop("_chunk_start_index", None)
    chunk_index = start_index if isinstance(start_index, int) else index
    metadata = _build_metadata(
        text,
        _with_source(annotated, page, filename),
        chunk_index,
        {},
    )
    anchors, spans = _collect_superscripts(annotated, text)
    if anchors:
        metadata["footnote_anchors"] = anchors
    chunk = {"text": text, "meta": metadata}
    return {**chunk, "_footnote_spans": spans} if spans else chunk


def _chunk_items(
    doc: Doc,
    split_fn: SplitFn,
    generate_metadata: bool = True,
    *,
    options: SplitOptions,
) -> Iterator[Chunk]:
    """Yield chunk records from ``doc`` using ``split_fn``."""

    filename = doc.get("source_path")
    limit = options.compute_limit()
    merged = _stitch_block_continuations(
        pipeline_attach_headings(
            _block_texts(doc, split_fn),
            is_heading=_is_heading,
            merge_block_text=_merge_heading_texts,
        ),
        limit,
    )
    collapsed = _collapse_records(merged, options, limit)
    builder = partial(build_chunk_with_meta, filename=filename)
    return pipeline_chunk_records(
        collapsed,
        generate_metadata=generate_metadata,
        build_plain=build_chunk,
        build_with_meta=builder,
    )


def _inject_continuation_context(items: Iterable[Chunk], limit: int | None) -> Iterator[Chunk]:
    prev_text: str | None = None
    for item in items:
        text = item.get("text", "")
        lead = text.lstrip()
        if prev_text is None or not lead or not _is_continuation_lead(lead):
            prev_text = text
            yield item
            continue
        context = _last_sentence(prev_text)
        if not context or lead.startswith(context):
            prev_text = text
            yield item
            continue
        combined = f"{context} {text}".strip()
        prev_text = combined
        yield {**item, "text": combined}


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
        self.min_chunk_size = derive_min_chunk_size(
            self.chunk_size, self.min_chunk_size
        )  # noqa: E501

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload
        if not isinstance(doc, dict) or doc.get("type") != "page_blocks":
            return a
        options = SplitOptions.from_base(
            self.chunk_size, self.overlap, self.min_chunk_size
        ).with_meta(a.meta)
        split_fn, metric_fn = _get_split_fn(
            options.chunk_size, options.overlap, options.min_chunk_size
        )
        limit = options.compute_limit()
        chunk_records = _chunk_items(
            doc,
            split_fn,
            self.generate_metadata,
            options=options,
        )
        items = list(_inject_continuation_context(chunk_records, limit))
        meta = SplitMetrics(len(items), metric_fn()).apply(a.meta)
        return Artifact(payload={"type": "chunks", "items": items}, meta=meta)


DEFAULT_SPLITTER = _SplitSemanticPass()


def make_splitter(**opts: Any) -> _SplitSemanticPass:
    """Return a configured ``split_semantic`` pass from ``opts``."""
    opts_map: _OverrideOpts = {
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
