from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from functools import reduce
from itertools import accumulate, chain, dropwhile, repeat, takewhile
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


def _last_non_empty_line(text: str) -> str:
    return next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")


def _trim_trailing_empty(lines: list[str]) -> list[str]:
    return list(reversed(list(dropwhile(lambda ln: not ln.strip(), reversed(lines)))))


def _partition_preamble(lines: list[str]) -> tuple[list[str], list[str]]:
    if not lines:
        return [], []

    idx = len(lines)
    while idx > 0 and lines[idx - 1].strip():
        idx -= 1
    while idx > 0 and not lines[idx - 1].strip():
        idx -= 1

    if idx == 0:
        return lines, []
    return lines[:idx], lines[idx:]


_LIST_GAP_RE = re.compile(r"\n{2,}(?=\s*(?:[-\*\u2022]|\d+\.))")


def _collapse_list_gaps(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        prior = text[: match.start()]
        prev_line = prior.splitlines()[-1] if "\n" in prior else prior
        return "\n" if not _is_list_line(prev_line) else match.group(0)

    return _LIST_GAP_RE.sub(repl, text)


def _split_inline_list_start(line: str) -> tuple[str, str] | None:
    for idx, char in enumerate(line):
        if char in "-\u2022*" and (idx == 0 or line[idx - 1].isspace()):
            tail = line[idx:].lstrip()
            if _is_list_line(tail):
                return line[:idx].rstrip(), tail
        if char.isdigit() and (idx == 0 or line[idx - 1].isspace()):
            end = idx
            while end < len(line) and line[end].isdigit():
                end += 1
            if (
                end < len(line)
                and line[end] == "."
                and end + 1 < len(line)
                and line[end + 1].isspace()
            ):
                tail = line[idx:].lstrip()
                if _is_list_line(tail):
                    return line[:idx].rstrip(), tail
    return None


def _reserve_for_list(text: str, limit: int) -> tuple[str, str, str | None]:
    collapsed = _collapse_list_gaps(text)
    lines = collapsed.splitlines()

    inline = next(
        (
            (idx, result)
            for idx, line in enumerate(lines)
            if (result := _split_inline_list_start(line))
        ),
        None,
    )
    list_idx = next((i for i, ln in enumerate(lines) if _is_list_line(ln)), len(lines))

    if inline and inline[0] <= list_idx:
        idx, (head, tail) = inline
        pre_lines = [*lines[:idx], head] if head else lines[:idx]
        tail_lines = [tail, *lines[idx + 1 :]]
    elif list_idx < len(lines):
        pre_lines = lines[:list_idx]
        tail_lines = lines[list_idx:]
    else:
        return collapsed, "", None

    if not pre_lines:
        return collapsed, "", None

    block_lines = list(takewhile(lambda ln: not ln.strip() or _is_list_line(ln), tail_lines))
    if not block_lines:
        return collapsed, "", None

    rest_lines = tail_lines[len(block_lines) :]
    trimmed_pre = _trim_trailing_empty(pre_lines)
    trailing_gaps = pre_lines[len(trimmed_pre) :]

    if not trimmed_pre:
        return collapsed, "", None

    pre_text = "\n".join(trimmed_pre)
    block_text = "\n".join(block_lines)
    combined_len = len(pre_text) + (1 if pre_text and block_text else 0) + len(block_text)
    if combined_len <= limit:
        return collapsed, "", None

    keep_lines, intro_lines = _partition_preamble(trimmed_pre)
    if not intro_lines and len(keep_lines) > 1 and any(_is_list_line(ln) for ln in block_lines):
        candidate_intro = keep_lines[-1]
        if candidate_intro.strip() and not _is_list_line(candidate_intro):
            keep_lines = keep_lines[:-1]
            intro_lines = [candidate_intro, *intro_lines]
    if not keep_lines:
        return collapsed, "", None

    chunk_text = "\n".join(keep_lines)
    remainder_parts = [
        *intro_lines,
        *trailing_gaps,
        *block_lines,
        *rest_lines,
    ]
    remainder = "\n".join(remainder_parts).lstrip("\n")
    intro_line = _first_non_empty_line("\n".join(intro_lines)) if intro_lines else ""
    intro_hint = intro_line if intro_line.strip() else None
    return chunk_text, remainder, intro_hint


def _list_intro_start(text: str) -> int:
    """Return the index where a trailing list introduction begins."""

    return max(
        (
            pos + span
            for token, span in (("\n\n", 2), (". ", 2), ("! ", 2), ("? ", 2))
            if (pos := text.rfind(token)) != -1
        ),
        default=-1,
    )


def _peel_list_intro(text: str) -> tuple[str, str]:
    """Split ``text`` into non-intro content and the trailing list preamble."""

    stripped = text.rstrip()
    colon_idx = max(stripped.rfind(":"), stripped.rfind("："))
    if colon_idx == -1:
        return text, ""
    prefix = stripped[: colon_idx + 1]
    start = _list_intro_start(prefix)
    if start <= 0:
        return text, ""
    return prefix[:start].rstrip(), prefix[start:].lstrip()


def _compose_intro_with_chunk(intro: str, chunk: str, separators: int) -> str:
    """Compose ``intro`` and ``chunk`` with controlled blank-line separators."""

    intro_lines = intro.splitlines()
    chunk_body = chunk.strip("\n")
    chunk_lines = chunk_body.splitlines() if chunk_body else []

    if not intro_lines:
        return chunk_body
    if not chunk_lines:
        return "\n".join(intro_lines)

    desired_gaps = max(separators, 1)
    spacer = [""] * max(desired_gaps - 1, 0)
    return "\n".join(chain(intro_lines, spacer, chunk_lines))


def _prepend_intro(intro: str, rest: str) -> str:
    """Attach ``intro`` ahead of ``rest`` while normalizing spacing."""

    intro_core = intro.strip("\n")
    if not rest:
        return intro_core

    leading_newlines = len(rest) - len(rest.lstrip("\n"))
    tail = rest[leading_newlines:]
    if not intro_core:
        return tail.strip("\n")

    trailing_intro_newlines = len(intro) - len(intro.rstrip("\n"))
    separators = trailing_intro_newlines + (leading_newlines or 1)
    return _compose_intro_with_chunk(intro_core, tail, separators)


def _rebalance_lists(raw: str, rest: str) -> tuple[str, str]:
    """Shift trailing context or list block into ``rest`` when it starts with a list."""

    if not rest or not _is_list_line(_first_non_empty_line(rest)):
        return raw, rest

    lines = _trim_trailing_empty(raw.splitlines())
    trimmed = "\n".join(lines)
    has_list = any(_is_list_line(ln) for ln in lines)

    if not has_list:
        kept, intro = _peel_list_intro(trimmed)
        if intro:
            return kept, _prepend_intro(intro, rest)

    # Determine split point: last non-list line if ``raw`` already contains list items,
    # otherwise the preceding blank line so that list introductions move with the list.
    # fmt: off
    idx = next(
        (
            i
            for i, ln in enumerate(reversed(lines))
            if (
                (ln.strip() and not _is_list_line(ln))
                if has_list
                else not ln.strip()
            )
        ),
        len(lines),
    )
    # fmt: on
    start = len(lines) - idx
    if not has_list and start == 0:
        return trimmed, rest
    block = lines[start:]
    if not block:
        return trimmed, rest

    moved = "\n".join(block).strip()
    kept = "\n".join(lines[:start]).rstrip()
    return kept, _prepend_intro(moved, rest)


def _truncate_with_remainder(text: str, limit: int) -> tuple[str, str]:
    if len(text) <= limit or limit <= 0:
        return text, ""

    if limit <= 100:
        prefix = text[:limit]
        chunk = prefix.rstrip() or prefix
        return chunk, text[len(chunk) :]

    truncate_point = limit - 100
    sentence_endings = (". ", ".\n", "! ", "!\n", "? ", "?\n")
    best_sentence = max(
        (
            pos
            for ending in sentence_endings
            if (pos := text.rfind(ending, 0, truncate_point)) > truncate_point * 0.7
        ),
        default=-1,
    )
    sentence_idx: int | None = best_sentence + 1 if best_sentence > 0 else None

    paragraph_idx_raw = text.rfind("\n\n", 0, truncate_point)
    paragraph_idx: int | None = (
        paragraph_idx_raw if paragraph_idx_raw > truncate_point * 0.7 else None
    )

    word_idx_raw = text.rfind(" ", 0, truncate_point)
    word_idx: int | None = word_idx_raw if word_idx_raw > truncate_point * 0.8 else None

    for idx in (sentence_idx, paragraph_idx, word_idx):
        if idx:
            chunk = text[:idx].rstrip()
            if chunk:
                return chunk, text[idx:]

    fallback = text[:truncate_point]
    chunk = fallback.rstrip() or fallback
    return chunk, text[len(chunk) :]


def _split(text: str, limit: int) -> list[str]:
    """Yield ``text`` slices no longer than ``limit`` using soft boundaries."""

    def step(
        state: tuple[list[str], str, str | None], _: object
    ) -> tuple[list[str], str, str | None]:
        pieces, remaining, intro_hint = state
        if not remaining:
            return state

        candidate, rem, next_intro = _reserve_for_list(remaining, limit)
        source = candidate or remaining
        first = _first_non_empty_line(source)
        second = source.splitlines()[1] if "\n" in source else ""
        is_list = _is_list_line(first) or (_is_list_line(second) and len(first) < limit)

        if is_list and len(source) > limit:
            suffix = f"\n{rem}" if rem else ""
            raw, rest = f"{source}{suffix}", ""
        else:
            raw, leftover = _truncate_with_remainder(source, limit)
            suffix = f"\n{rem}" if rem else ""
            rest = f"{leftover}{suffix}"
        raw, rest = _collapse_list_gaps(raw), _collapse_list_gaps(rest)
        raw, rest = _rebalance_lists(raw, rest)
        raw_intro_line = _first_non_empty_line(raw)
        rest_first_line = _first_non_empty_line(rest)
        if (
            intro_hint
            and intro_hint.strip()
            and raw_intro_line.strip() == intro_hint.strip()
            and rest_first_line
            and _is_list_line(rest_first_line)
        ):
            rest_lines = rest.splitlines()
            block_lines = list(
                takewhile(lambda ln: not ln.strip() or _is_list_line(ln), rest_lines)
            )
            block_text = "\n".join(block_lines).lstrip("\n")
            if block_text:
                raw = f"{raw.rstrip()}\n{block_text}".strip("\n")
                rest = "\n".join(rest_lines[len(block_lines) :])
        raw_first_line = _first_non_empty_line(raw)
        skip_trim = bool(pieces) and (
            _is_list_line(raw_first_line)
            or (intro_hint and intro_hint.strip() and raw_first_line.strip() == intro_hint.strip())
        )
        trimmed = raw if not pieces or skip_trim else _trim_overlap(pieces[-1], raw)
        if trimmed and trimmed.strip():
            if pieces and _is_list_line(_first_non_empty_line(trimmed)):
                merged = f"{pieces[-1].rstrip()}\n{trimmed.lstrip()}"
                if len(merged) <= limit:  # noqa: SIM108
                    pieces = [*pieces[:-1], merged]
                else:
                    pieces = [*pieces, trimmed]
            else:
                pieces = [*pieces, trimmed]
        return pieces, rest.lstrip(), next_intro

    states: Iterable[tuple[list[str], str, str | None]] = accumulate(
        repeat(None),
        step,
        initial=([], text, None),
    )
    return next(p for p, r, _ in states if not r)


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
        prefix = curr[:overlap]
        prev_index = len(prev) - overlap
        prev_char = prev[prev_index - 1] if prev_index > 0 else ""
        next_non_space = next((ch for ch in curr[overlap:] if not ch.isspace()), "")
        stripped_prefix = prefix.strip()
        words = re.findall(r"\b\w+\b", stripped_prefix)
        single_title = len(words) == 1 and words[0][0].isupper() and words[0][1:].islower()
        if prev_char.isalnum():
            return curr
        if single_title and (next_non_space.islower() or next_non_space.isdigit()):
            return curr
        return curr[overlap:].lstrip()
    prefix = curr_lower.split("\n\n", 1)[0]
    return curr[len(prefix) :].lstrip() if _contains(prev_lower, prefix) else curr


def _starts_mid_sentence(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and re.match(r"^[\"'(]*[A-Z0-9]", stripped) is None


_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*")


def _steal_sentence_prefix(prev: str, fragment: str, limit: int | None) -> tuple[str, str] | None:
    """Move the leading sentence from ``fragment`` onto ``prev`` when possible."""

    stripped = fragment.lstrip()
    if not stripped:
        return None

    offset = len(fragment) - len(stripped)
    for match in _SENTENCE_END_RE.finditer(stripped):
        end = match.end()
        if end < len(stripped) and not stripped[end].isspace():
            continue
        prefix = fragment[: offset + end]
        remainder = fragment[offset + end :]
        candidate = f"{prev.rstrip()} {prefix.strip()}".strip()
        if limit is not None and len(candidate) > limit:
            return None
        return candidate, remainder.lstrip()

    candidate = f"{prev.rstrip()} {stripped}".strip()
    if limit is not None and len(candidate) > limit:
        return None
    return candidate, ""


def _merge_text(prev: str, curr: str) -> str:
    last = _last_non_empty_line(prev)
    first = _first_non_empty_line(curr)
    cond = _is_list_line(last) and _is_list_line(first)
    sep = "\n" if cond else "\n\n"
    return f"{prev.rstrip()}{sep}{curr}".strip()


def _merge_sentence_pieces(
    pieces: Iterable[str],
    limit: int | None = None,
) -> list[str]:
    def step(acc: list[str], piece: str) -> list[str]:
        if acc and _starts_mid_sentence(piece):
            merged_prev = acc[-1]
            remainder = piece
            while remainder:
                result = _steal_sentence_prefix(merged_prev, remainder, limit)
                if result is None:
                    break
                merged_prev, remainder = result
                if remainder:
                    remainder = remainder.lstrip()
                else:
                    return [*acc[:-1], merged_prev]
                if not _starts_mid_sentence(remainder):
                    break
            if merged_prev is not acc[-1]:
                acc = [*acc[:-1], merged_prev]
                piece = remainder
            elif limit is None or len(acc[-1]) + 1 + len(piece) <= limit:
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


def _should_merge(prev_text: str, curr_text: str, min_words: int) -> bool:
    prev_words = _word_count(prev_text)
    curr_words = _word_count(curr_text)
    prev_coherent = _coherent(prev_text)
    curr_coherent = _coherent(curr_text)
    return any(
        (
            prev_words < min_words,
            (curr_words < min_words and not curr_coherent),
            not prev_coherent,
            _starts_mid_sentence(curr_text),
        )
    )


def _merge_items(
    acc: list[dict[str, Any]],
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    text = item["text"]
    if acc:
        prev = acc[-1]
        text = _trim_overlap(prev["text"], text)
        if _should_merge(prev["text"], text, _min_words()):
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

    pieces = [
        piece
        for piece in _merge_sentence_pieces(_split(item.get("text", ""), avail), avail)
        if piece.strip()
    ]

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

    return [row for row in (build(x) for x in enumerate(pieces)) if row["text"].strip()]


def _preserve_chunks(meta: dict[str, Any] | None) -> bool:
    opts = ((meta or {}).get("options") or {}).get("split_semantic", {})
    chunk_size = opts.get("chunk_size")
    overlap = opts.get("overlap")
    return chunk_size is not None or (isinstance(overlap, int | float) and overlap > 0)


def _rows(doc: Doc, *, preserve: bool = False) -> list[Row]:
    debug_log: list[str] | None = (
        [] if (not preserve and os.getenv("PDF_CHUNKER_DEDUP_DEBUG")) else None
    )
    items = doc.get("items", [])
    processed = items if preserve else _dedupe(_coalesce(items), log=debug_log)
    if debug_log is not None:
        logger = logging.getLogger(__name__)
        logger.warning("dedupe dropped %d duplicates", len(debug_log))
        for dup in debug_log:
            logger.warning("dedupe dropped: %s", dup[:80])
        for dup in _flag_potential_duplicates(processed):
            logger.warning("possible duplicate retained: %s", dup[:80])
    return [r for i in processed for r in _rows_from_item(i)]


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
        preserve = _preserve_chunks(a.meta)
        rows = _rows(doc, preserve=preserve) if doc.get("type") == "chunks" else []
        meta = _update_meta(a.meta, len(rows))
        return Artifact(payload=rows, meta=meta)


emit_jsonl = register(_EmitJsonlPass())
