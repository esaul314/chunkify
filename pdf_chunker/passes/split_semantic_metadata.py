"""Chunk metadata assembly helpers for :mod:`pdf_chunker.passes.split_semantic`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pdf_chunker.passes.split_semantic_inline import (
    _span_attrs,
    _span_bounds,
    _span_style,
)
from pdf_chunker.passes.split_semantic_lists import _tag_list
from pdf_chunker.text_cleaning import STOPWORDS
from pdf_chunker.utils import _build_metadata
from pdf_chunker.strategies.bullets import (
    BulletHeuristicStrategy,
    default_bullet_strategy,
)

Block = dict[str, Any]
Chunk = dict[str, Any]

_STOPWORD_TITLES = frozenset(word.title() for word in STOPWORDS)
_FOOTNOTE_TAILS = {"", ".", ",", ";", ":"}


def _chunk_meta(chunk: Chunk) -> Mapping[str, Any]:
    meta = chunk.get("meta")
    if isinstance(meta, Mapping):
        return meta
    legacy = chunk.get("metadata")
    return legacy if isinstance(legacy, Mapping) else {}


def _meta_list_kind(meta: Mapping[str, Any] | None) -> str | None:
    if not isinstance(meta, Mapping):
        return None
    kind = meta.get("list_kind")
    return kind if isinstance(kind, str) and kind else None


def _meta_is_list(meta: Mapping[str, Any] | None) -> bool:
    if not isinstance(meta, Mapping):
        return False
    block_type = meta.get("block_type")
    if block_type == "list_item":
        return True
    return block_type in {None, ""} and _meta_list_kind(meta) is not None


def _collect_superscripts(
    block: Block, text: str
) -> tuple[list[dict[str, str]], tuple[tuple[int, int], ...]]:
    if not text:
        return [], ()
    limit = len(text)

    def _normalize(span: Any) -> tuple[dict[str, str], tuple[int, int]] | None:
        if _span_style(span) != "superscript":
            return None
        bounds = _span_bounds(span, limit)
        if bounds is None:
            return None
        start, end = bounds
        raw = text[start:end]
        snippet = raw.strip()
        if not snippet or text[end:].strip() not in _FOOTNOTE_TAILS:
            return None
        attrs = _span_attrs(span)
        note_id = attrs.get("note_id") if attrs else None
        focus = raw.find(snippet)
        span_start = start + (focus if focus >= 0 else 0)
        span_end = span_start + len(snippet)
        public = {"text": snippet, **({"note_id": note_id} if note_id else {})}
        return public, (span_start, span_end)

    entries = tuple(
        entry
        for entry in (_normalize(span) for span in tuple(block.get("inline_styles") or ()))
        if entry
    )
    anchors = [public for public, _ in entries]
    spans = tuple(span for _, span in entries if span[0] < span[1])
    return anchors, spans


def _normalize_bullet_tail(tail: str) -> str:
    if not tail:
        return ""
    head, *rest = tail.split(" ", 1)
    normalized = head.lower() if head in _STOPWORD_TITLES else head
    return f"{normalized} {rest[0]}".strip() if rest and rest[0] else normalized


def _normalized_heading_lines(headings: Iterable[str]) -> tuple[str, ...]:
    return tuple(stripped for heading in headings if heading and (stripped := heading.strip()))


_HEADING_BODY_SEPARATOR = "\n"


def _resolve_bullet_strategy(
    strategy: BulletHeuristicStrategy | None,
) -> BulletHeuristicStrategy:
    return strategy or default_bullet_strategy()


def _merge_heading_texts(
    headings: Iterable[str],
    body: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> str:
    normalized_headings = _normalized_heading_lines(headings)
    heuristics = _resolve_bullet_strategy(strategy)
    if any(heuristics.starts_with_bullet(h.lstrip()) for h in normalized_headings):
        lead = " ".join(h.rstrip() for h in normalized_headings).rstrip()
        tail = _normalize_bullet_tail(body.lstrip()) if body else ""
        return f"{lead} {tail}".strip()

    heading_block = "\n".join(normalized_headings)
    body_text = body.strip() if body else ""

    if not heading_block:
        return body_text
    if not body_text:
        return heading_block

    return f"{heading_block}{_HEADING_BODY_SEPARATOR}{body_text}"


def _existing_source(block: Block) -> Mapping[str, Any]:
    current = block.get("source")
    return current if isinstance(current, Mapping) else {}


def _fallback_source(page: int, filename: str | None) -> dict[str, Any]:
    return {k: v for k, v in (("filename", filename), ("page", page)) if v is not None}


def _merge_source(existing: Mapping[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    has_page = existing.get("page") is not None
    filtered = {
        key: value
        for key, value in fallback.items()
        if value is not None and (key != "page" or not has_page)
    }
    return {**filtered, **existing}


def _strip_nulls(source: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if value is not None}


def _with_source(block: Block, page: int, filename: str | None) -> Block:
    existing = _existing_source(block)
    fallback = _fallback_source(page, filename)
    merged = _merge_source(existing, fallback)
    return {**block, "source": _strip_nulls(merged)}


def build_chunk(text: str) -> Chunk:
    return {"text": text}


def build_chunk_with_meta(
    text: str,
    block: Block,
    page: int,
    filename: str | None,
    index: int,
    *,
    bullet_strategy: BulletHeuristicStrategy | None = None,
) -> Chunk:
    heuristics = _resolve_bullet_strategy(bullet_strategy)
    annotated = _tag_list(block, strategy=heuristics)
    start_index = annotated.pop("_chunk_start_index", None)
    chunk_index = start_index if isinstance(start_index, int) else index
    metadata = _build_metadata(
        text,
        _with_source(annotated, page, filename),
        chunk_index,
        {},
        strategy=heuristics,
    )
    anchors, spans = _collect_superscripts(annotated, text)
    if anchors:
        metadata["footnote_anchors"] = anchors
    chunk = {"text": text, "meta": metadata}
    return {**chunk, "_footnote_spans": spans} if spans else chunk


__all__ = ["Block", "Chunk", "build_chunk", "build_chunk_with_meta", "_chunk_meta", "_meta_is_list", "_merge_heading_texts"]
