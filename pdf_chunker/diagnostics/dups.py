from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence
import re

__all__ = ["find_dups_pageblocks", "find_dups_chunks", "overlap_dups"]


def _fingerprint(text: str) -> str:
    table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    return re.sub(r"\s+", " ", text.strip().translate(table)).lower()


def _group(items: Sequence[Mapping[str, Any]], pos_fn):
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for idx, item in enumerate(items):
        fp = _fingerprint(str(item.get("text", "")))
        if fp:
            groups[fp].append(pos_fn(item, idx))
    return groups


def _ngram_fps(text: str, size: int = 5) -> set[str]:
    tokens = _fingerprint(text).split()
    return {
        " ".join(tokens[i : i + size])
        for i in range(len(tokens) - size + 1)
    }


def overlap_dups(
    items: Sequence[Mapping[str, Any]], pos_fn, size: int = 5
):
    fps = [_ngram_fps(str(it.get("text", "")), size) for it in items]
    return [
        {
            "fp": next(iter(overlap)),
            "text": items[min(i, j, key=lambda k: len(str(items[k].get("text", ""))))].get(
                "text", ""
            )[:80],
            "count": 2,
            "first": pos_fn(items[i], i),
            "second": pos_fn(items[j], j),
        }
        for i in range(len(items))
        for j in range(i + 1, len(items))
        if (overlap := fps[i] & fps[j])
    ]


def _format(items: Sequence[Mapping[str, Any]], groups: Mapping[str, list[Mapping[str, Any]]]):
    return [
        {
            "fp": fp,
            "text": items[pos[0]["index"]].get("text", "")[:80],
            "count": len(pos),
            "first": pos[0],
            "second": pos[1],
        }
        for fp, pos in groups.items()
        if len(pos) > 1
    ]


def find_dups_pageblocks(blocks: Sequence[Mapping[str, Any]]):
    """Return duplicate page blocks based on normalized text."""
    pos = lambda b, i: {
        "index": i,
        **{k: b.get(k) for k in ("page", "bbox") if b.get(k) is not None},
    }
    groups = _group(blocks, pos)
    return _format(blocks, groups) + overlap_dups(blocks, pos)


def find_dups_chunks(chunks: Sequence[Mapping[str, Any]]):
    """Return duplicate chunks based on normalized text."""
    pos = lambda c, i: {
        "index": i,
        **(
            {"chunk_id": c.get("metadata", {}).get("chunk_id")}
            if c.get("metadata", {}).get("chunk_id")
            else {}
        )
    }
    groups = _group(chunks, pos)
    return _format(chunks, groups) + overlap_dups(chunks, pos)
