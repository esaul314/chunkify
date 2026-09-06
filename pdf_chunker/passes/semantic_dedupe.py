"""Pure deduplication for canonical semantic chunks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pdf_chunker.passes.emit_jsonl_text import (
    contains,
    has_sentence_ending,
    normalize,
)
from pdf_chunker.passes.transform_log import TransformationLog, maybe_record

Chunk = dict[str, Any]


def deduplicate_chunks(
    items: Iterable[Mapping[str, Any]],
    *,
    transform_log: TransformationLog | None = None,
) -> list[Chunk]:
    """Remove repeated sentence-bearing semantic chunk text.

    Repetitive fragments without sentence boundaries are retained because they
    may be legitimate overlap or extraction output.  This is deliberately a
    semantic-stage operation: JSONL emission must not make another content
    decision after this function establishes the canonical chunk boundaries.
    """
    result: list[Chunk] = []
    accumulated = ""

    for source in items:
        text = source.get("text", "")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        normalized = normalize(text)

        if contains(accumulated, normalized) and has_sentence_ending(text):
            maybe_record(
                transform_log,
                "deduplicated",
                "split_semantic.dedupe",
                "duplicate_chunk_removed",
                text,
                "",
            )
            continue

        result.append({**dict(source), "text": text})
        accumulated += normalized

    return result


__all__ = ["deduplicate_chunks"]
