"""List and caption heuristics for the semantic split pass."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any

from pdf_chunker.strategies.bullets import (
    BulletHeuristicStrategy,
    default_bullet_strategy,
)
from pdf_chunker.passes.split_semantic_inline import _span_attrs

Block = dict[str, Any]

_STYLED_LIST_KIND = "styled"
_CAPTION_PREFIXES = (
    "figure",
    "fig.",
    "table",
    "tbl.",
    "image",
    "img.",
    "diagram",
)
_CAPTION_LABEL_RE = re.compile(
    r"(?:\d+(?:[-–—.]\d+)*[a-z]?|[ivxlcdm]+(?:[-–—.][ivxlcdm]+)*[a-z]?)",
    re.IGNORECASE,
)
_CAPTION_FLAG = "_caption_attached"


def _resolve_strategy(
    strategy: BulletHeuristicStrategy | None,
) -> BulletHeuristicStrategy:
    return strategy or default_bullet_strategy()


def _inline_list_kinds(block: Mapping[str, Any]) -> tuple[str, ...]:
    styles = tuple(block.get("inline_styles") or ())
    return tuple(
        attrs["list_kind"]
        for attrs in (_span_attrs(span) for span in styles)
        if isinstance(attrs, Mapping)
        and isinstance(attrs.get("list_kind"), str)
        and attrs["list_kind"]
    )


def _block_list_kind(block: Mapping[str, Any]) -> str | None:
    if not isinstance(block, Mapping):
        return None
    declared = block.get("list_kind")
    if isinstance(declared, str) and declared:
        return declared
    inline = _inline_list_kinds(block)
    return next(iter(inline), None)


def _leading_list_kind(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> str | None:
    heuristics = _resolve_strategy(strategy)
    lines = (line.lstrip() for line in text.splitlines())
    first = next((line for line in lines if line), "")
    if heuristics.starts_with_bullet(first):
        return "bullet"
    if heuristics.starts_with_number(first):
        return "numbered"
    return None


def _infer_list_kind(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> str | None:
    heuristics = _resolve_strategy(strategy)
    if heuristics.starts_with_bullet(text):
        return "bullet"
    if heuristics.starts_with_number(text):
        return "numbered"
    lines = tuple(line.lstrip() for line in text.splitlines())
    if any(heuristics.starts_with_bullet(line) for line in lines):
        return "bullet"
    if any(heuristics.starts_with_number(line) for line in lines):
        return "numbered"
    return None


def _list_line_ratio(
    text: str,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> tuple[int, int]:
    heuristics = _resolve_strategy(strategy)
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not lines:
        return 0, 0
    list_lines = sum(
        1
        for line in lines
        if heuristics.starts_with_bullet(line) or heuristics.starts_with_number(line)
    )
    return list_lines, len(lines)


def _tag_list(
    block: Block,
    *,
    strategy: BulletHeuristicStrategy | None = None,
) -> Block:
    heuristics = _resolve_strategy(strategy)
    text = block.get("text", "")
    block_type = block.get("type")
    existing_kind = block.get("list_kind")

    if block_type == "list_item":
        if existing_kind:
            return block
        inferred = _infer_list_kind(text, strategy=heuristics)
        return {**block, "list_kind": inferred} if inferred else block

    leading_kind = _leading_list_kind(text, strategy=heuristics)
    if not leading_kind:
        inferred = _infer_list_kind(text, strategy=heuristics)
        if not inferred:
            return block
        list_lines, total_lines = _list_line_ratio(text, strategy=heuristics)
        if total_lines and (list_lines * 2) >= total_lines:
            return {**block, "type": "list_item", "list_kind": inferred}
        return block

    return {**block, "type": "list_item", "list_kind": leading_kind}


def _merge_styled_list_text(first: str, second: str) -> str:
    lead = first.rstrip()
    tail = second.lstrip()
    if not lead:
        return tail
    if not tail:
        return lead
    return f"{lead}\n\n{tail}"


def _chain_sequences(*values: Any) -> tuple[Any, ...]:
    def _normalize(value: Any) -> tuple[Any, ...]:
        if not value:
            return ()
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value,)

    return tuple(chain.from_iterable(_normalize(value) for value in values))


def _without_keys(mapping: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    drop = frozenset(keys)
    return {k: v for k, v in mapping.items() if k not in drop}


def _with_optional_tuple(
    mapping: Mapping[str, Any], key: str, values: tuple[Any, ...]
) -> dict[str, Any]:
    if values:
        return {**mapping, key: values}
    return {k: v for k, v in mapping.items() if k != key}


@dataclass(frozen=True)
class _BlockEnvelope:
    block_type: str
    list_kind: str | None = None


def _coalesce_list_kind(blocks: Iterable[Block]) -> str | None:
    kinds = {block.get("list_kind") for block in blocks if block.get("list_kind")}
    return next(iter(kinds)) if len(kinds) == 1 else None


def _coalesce_block_type(blocks: Iterable[Block]) -> str:
    types = tuple(
        block.get("type") for block in blocks if isinstance(block, Mapping) and block.get("type")
    )
    candidates = tuple(t for t in types if t != "heading")
    if not candidates:
        return "paragraph"
    if all(t == "list_item" for t in candidates):
        return "list_item"
    unique = frozenset(candidates)
    if len(unique) == 1:
        return candidates[0]
    if "list_item" in unique:
        return "paragraph"
    return candidates[0]


def _resolve_envelope(
    blocks: Iterable[Block], *, default_list_kind: str | None = None
) -> _BlockEnvelope:
    sequence = tuple(blocks)
    block_type = _coalesce_block_type(sequence)
    kind = _coalesce_list_kind(sequence) or default_list_kind
    return _BlockEnvelope(block_type, kind)


def _apply_envelope(base: Mapping[str, Any], text: str, envelope: _BlockEnvelope) -> Block:
    payload = {
        **_without_keys(base, {"text", "list_kind"}),
        "text": text,
        "type": envelope.block_type,
    }
    return {**payload, "list_kind": envelope.list_kind} if envelope.list_kind else payload


def _merge_styled_list_block(primary: Block, secondary: Block) -> Block:
    merged_text = _merge_styled_list_text(
        str(primary.get("text", "")), str(secondary.get("text", ""))
    )
    envelope = _resolve_envelope((primary, secondary), default_list_kind=_STYLED_LIST_KIND)
    merged = _apply_envelope(primary, merged_text, envelope)
    inline_styles = _chain_sequences(primary.get("inline_styles"), secondary.get("inline_styles"))
    source_blocks = _chain_sequences(primary.get("source_blocks"), secondary.get("source_blocks"))
    without_bbox = _without_keys(merged, {"bbox"})
    with_styles = _with_optional_tuple(without_bbox, "inline_styles", inline_styles)
    return _with_optional_tuple(with_styles, "source_blocks", source_blocks)


def _merge_styled_list_records(
    records: Iterable[tuple[int, Block, str]],
) -> Iterator[tuple[int, Block, str]]:
    pending: tuple[int, Block, str] | None = None
    for page, block, text in records:
        if block.get("list_kind") == _STYLED_LIST_KIND:
            block_copy = dict(block)
            if pending is None:
                pending = (page, block_copy, text)
                continue
            pending_page, pending_block, pending_text = pending
            merged_block = _merge_styled_list_block(pending_block, block_copy)
            merged_text = _merge_styled_list_text(pending_text, text)
            pending = (min(pending_page, page), merged_block, merged_text)
            continue
        if pending is not None:
            yield pending
            pending = None
        yield page, block, text
    if pending is not None:
        yield pending


def _looks_like_caption(text: str) -> bool:
    stripped = text.lstrip()
    lower = stripped.lower()
    prefix = next(
        (candidate for candidate in _CAPTION_PREFIXES if lower.startswith(candidate)),
        None,
    )
    if not prefix:
        return False
    remainder = stripped[len(prefix) :].lstrip()
    if not remainder:
        return False
    label_match = _CAPTION_LABEL_RE.match(remainder)
    if not label_match:
        return False
    tail = remainder[label_match.end() :].lstrip()
    if not tail:
        return False
    head = tail[0]
    if head in '.:()"“”':
        return True
    if head in "–—":
        return True
    if head == "-":
        if len(tail) == 1:
            return True
        return not tail[1].isalnum()
    return False


def _contains_caption_line(text: str) -> bool:
    return any(_looks_like_caption(line) for line in text.splitlines())


def _has_caption(block: Block) -> bool:
    return isinstance(block, Mapping) and bool(dict(block).get(_CAPTION_FLAG))


def _mark_caption(block: Block) -> Block:
    if not isinstance(block, Mapping):
        return block
    data = dict(block)
    data[_CAPTION_FLAG] = True
    return data


def _append_caption(prev_text: str, caption: str) -> str:
    head = prev_text.rstrip()
    tail = caption.strip()
    if not head:
        return tail
    return "\n\n".join(filter(None, (head, tail)))


__all__ = [
    "_STYLED_LIST_KIND",
    "_BlockEnvelope",
    "_block_list_kind",
    "_resolve_envelope",
    "_apply_envelope",
    "_merge_styled_list_records",
    "_looks_like_caption",
    "_contains_caption_line",
    "_has_caption",
    "_mark_caption",
    "_append_caption",
]
