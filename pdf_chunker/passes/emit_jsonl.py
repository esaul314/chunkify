from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from functools import reduce
from itertools import dropwhile
from typing import Any, cast

from pdf_chunker.framework import Artifact, register
from pdf_chunker.list_detection import starts_with_bullet, starts_with_number
from pdf_chunker.utils import _truncate_chunk

Row = dict[str, Any]
Doc = dict[str, Any]


def _min_words() -> int:
    return int(os.getenv("PDF_CHUNKER_JSONL_MIN_WORDS", "50"))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _metadata_key() -> str:
    return os.getenv("PDF_CHUNKER_JSONL_META_KEY", "metadata")


def _compat_chunk_id(chunk_id: str) -> str:
    match = re.search(r"_p(\d+)_c", chunk_id)
    if not match:
        return chunk_id
    page = max(int(match.group(1)) - 1, 0)
    return f"{chunk_id[: match.start(1)]}{page}{chunk_id[match.end(1):]}"


def _max_chars() -> int:
    return int(os.getenv("PDF_CHUNKER_JSONL_MAX_CHARS", "8000"))


def _is_list_line(line: str) -> bool:
    stripped = line.lstrip()
    return starts_with_bullet(stripped) or starts_with_number(stripped)


def _first_non_empty_line(text: str) -> str:
    return next((ln for ln in text.splitlines() if ln.strip()), "")


def _trim_trailing_empty(lines: list[str]) -> list[str]:
    return list(reversed(list(dropwhile(lambda ln: not ln.strip(), reversed(lines)))))

def _split(text: str, limit: int) -> list[str]:
    """Yield ``text`` slices no longer than ``limit`` using soft boundaries."""

    pieces: list[str] = []
    t = text
    while t:
        raw = _truncate_chunk(t, limit)
        rest = t[len(raw) :]
        if rest:
            head = _first_non_empty_line(rest)
            if head and _is_list_line(head):
                lines = _trim_trailing_empty(raw.splitlines())
                if not all(_is_list_line(ln) for ln in lines):
                    while lines and _is_list_line(lines[-1]):
                        rest = f"{lines.pop()}\n{rest.lstrip()}"
                    raw = "\n".join(lines)
                else:
                    raw, rest = t, ""
        trimmed = _trim_overlap(pieces[-1], raw) if pieces else raw
        if trimmed:
            pieces.append(trimmed)
        t = rest.lstrip()
    return pieces


def _coherent(text: str, min_chars: int = 40) -> bool:
    stripped = text.strip()
    return (
        len(stripped) >= min_chars
        and re.match(r"^[\"'(]*[A-Z0-9]", stripped) is not None
        and re.search(r"[.!?][\"')\]]*$", stripped) is not None
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle and needle in haystack)


def _overlap_len(prev_lower: str, curr_lower: str) -> int:
    length = min(len(prev_lower), len(curr_lower))
    return next(
        (i for i in range(length, 0, -1) if prev_lower.endswith(curr_lower[:i])),
        0,
    )


def _prefix_contained_len(haystack: str, needle: str) -> int:
    return next(
        (
            i
            for i in range(len(needle), 0, -1)
            if needle[:i] in haystack and (i == len(needle) or needle[i].isspace())
        ),
        0,
    )


def _trim_overlap(prev: str, curr: str) -> str:
    """Remove duplicated prefix from ``curr`` that already exists in ``prev``."""

    prev_lower, curr_lower = prev.lower(), curr.lower()
    if _contains(prev_lower, curr_lower):
        return ""
    overlap = _overlap_len(prev_lower, curr_lower)
    if overlap and overlap < len(curr) * 0.9:
        return curr[overlap:].lstrip()
    prefix = curr_lower.split("\n\n", 1)[0]
    return curr[len(prefix) :].lstrip() if _contains(prev_lower, prefix) else curr


def _starts_mid_sentence(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and re.match(r"^[\"'(]*[A-Z0-9]", stripped) is None


def _merge_text(prev: str, curr: str) -> str:
    return f"{prev}\n\n{curr}".strip()


def _merge_sentence_pieces(
    pieces: Iterable[str],
    limit: int | None = None,
) -> list[str]:
    def step(acc: list[str], piece: str) -> list[str]:
        if (
            acc
            and _starts_mid_sentence(piece)
            and (limit is None or len(acc[-1]) + 1 + len(piece) <= limit)
        ):
            merged = f"{acc[-1].rstrip()} {piece}".strip()
            return [*acc[:-1], merged]
        return [*acc, piece]

    return reduce(step, pieces, [])


def _merge_if_fragment(
    acc: list[dict[str, Any]],
    acc_text: str,
    acc_norm: str,
    item: dict[str, Any],
    text: str,
    text_norm: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """Merge ``text`` into ``acc`` if it begins mid-sentence."""

    if _starts_mid_sentence(text) and acc:
        prev = acc[-1]
        merged_text = f"{prev['text'].rstrip()} {text}".strip()
        merged_item = {**prev, "text": merged_text}
        merged_acc = f"{acc_text.rstrip()} {text}".strip()
        return (
            [*acc[:-1], merged_item],
            merged_acc,
            acc_norm + text_norm,
        )
    new_text = _merge_text(acc_text, text) if acc_text else text
    return (
        [*acc, {**item, "text": text}],
        new_text,
        acc_norm + text_norm,
    )


def _merge_items(
    acc: list[dict[str, Any]],
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    text = item["text"]
    if acc:
        prev = acc[-1]
        text = _trim_overlap(prev["text"], text)
        prev_words, curr_words = _word_count(prev["text"]), _word_count(text)
        should_merge = (
            prev_words < _min_words()
            or curr_words < _min_words()
            or not _coherent(prev["text"])
            or _starts_mid_sentence(text)
        )
        if should_merge:
            merged = _merge_text(prev["text"], text)
            return [*acc[:-1], {**prev, "text": merged}]
    return [*acc, {**item, "text": text}]


def _coalesce(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize item boundaries, trimming overlap and merging fragments."""

    cleaned = [{**i, "text": (i.get("text") or "").strip()} for i in items]
    cleaned = [i for i in cleaned if i["text"]]
    merged = reduce(_merge_items, cleaned, cast(list[dict[str, Any]], []))
    return merged


def _dedupe(
    items: Iterable[dict[str, Any]], *, log: list[str] | None = None
) -> list[dict[str, Any]]:
    """Remove items whose text already appears in prior items.

    When ``log`` is provided, dropped duplicate snippets are appended to it for
    debug inspection. The function itself remains pure; callers decide whether
    to record diagnostics.
    """

    def step(
        state: tuple[list[dict[str, Any]], str, str], item: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, str]:
        acc, acc_text, acc_norm = state
        text = item["text"]
        text_norm = _normalize(text)
        if _contains(acc_norm, text_norm):
            if log is not None:
                log.append(text)
            return state
        overlap = _overlap_len(acc_norm, text_norm) or _prefix_contained_len(
            acc_norm,
            text_norm,
        )
        if overlap:
            if log is not None:
                log.append(text[:overlap])
            text = text[overlap:].lstrip()
            if not text:
                return state
            text_norm = _normalize(text)
        return _merge_if_fragment(acc, acc_text, acc_norm, item, text, text_norm)

    initial: tuple[list[dict[str, Any]], str, str] = ([], "", "")
    return reduce(step, items, initial)[0]


def _flag_potential_duplicates(
    items: Iterable[dict[str, Any]], *, min_words: int = 10
) -> list[str]:
    """Return sentences appearing more than once after dedupe."""

    seen: set[str] = set()
    flagged: list[str] = []
    for sent in (s for item in items for s in re.split(r"(?<=[.!?])\s+", item["text"])):
        words = re.findall(r"\w+", sent.lower())
        if len(words) < min_words:
            continue
        key = " ".join(words)
        if key in seen:
            flagged.append(sent.strip())
        else:
            seen.add(key)
    return flagged


def _rows_from_item(item: dict[str, Any]) -> list[Row]:
    meta_key = _metadata_key()
    max_chars = _max_chars()
    meta: dict[str, Any] = item.get("meta") or {}
    chunk_id = meta.get("chunk_id")
    if chunk_id:
        meta = {**meta, "chunk_id": _compat_chunk_id(chunk_id)}
    base_meta = {meta_key: meta} if meta else {}
    overhead = len(json.dumps({"text": "", **base_meta}, ensure_ascii=False)) - 2
    avail = max(max_chars - overhead, 0)
    if avail <= 0:
        return []

    pieces = _merge_sentence_pieces(_split(item.get("text", ""), avail), avail)

    def build(idx_piece: tuple[int, str]) -> Row:
        idx, piece = idx_piece
        piece = piece.lstrip()
        if meta and len(pieces) > 1:
            meta_part = {meta_key: {**meta, "chunk_part": idx}}
        else:
            meta_part = base_meta
        row = {"text": piece, **meta_part}
        while len(json.dumps(row, ensure_ascii=False)) > max_chars:
            allowed = avail - (len(json.dumps(row, ensure_ascii=False)) - max_chars)
            allowed = min(allowed, len(piece) - 1)
            if allowed <= 0:
                return {"text": "", **meta_part}
            piece = _truncate_chunk(piece[:allowed], allowed)
            row = {"text": piece, **meta_part}
        return row

    return [build(x) for x in enumerate(pieces)]


def _rows(doc: Doc) -> list[Row]:
    debug_log: list[str] | None = [] if os.getenv("PDF_CHUNKER_DEDUP_DEBUG") else None
    items = _dedupe(_coalesce(doc.get("items", [])), log=debug_log)
    if debug_log is not None:
        logger = logging.getLogger(__name__)
        logger.warning("dedupe dropped %d duplicates", len(debug_log))
        for dup in debug_log:
            logger.warning("dedupe dropped: %s", dup[:80])
        for dup in _flag_potential_duplicates(items):
            logger.warning("possible duplicate retained: %s", dup[:80])
    return [r for i in items for r in _rows_from_item(i)]


def _update_meta(meta: dict[str, Any] | None, count: int) -> dict[str, Any]:
    base = dict(meta or {})
    base.setdefault("metrics", {}).setdefault("emit_jsonl", {})["rows"] = count
    return base


class _EmitJsonlPass:
    name = "emit_jsonl"
    input_type = dict
    output_type = list

    def __call__(self, a: Artifact) -> Artifact:
        doc = a.payload if isinstance(a.payload, dict) else {}
        rows = _rows(doc) if doc.get("type") == "chunks" else []
        meta = _update_meta(a.meta, len(rows))
        return Artifact(payload=rows, meta=meta)


emit_jsonl = register(_EmitJsonlPass())
