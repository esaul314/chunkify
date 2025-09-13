from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence
import re

__all__ = ["find_dups_pageblocks", "find_dups_chunks"]


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
    groups = _group(
        blocks,
        lambda b, i: {
            "index": i,
            **{k: b.get(k) for k in ("page", "bbox") if b.get(k) is not None},
        },
    )
    return _format(blocks, groups)


def find_dups_chunks(chunks: Sequence[Mapping[str, Any]]):
    """Return duplicate chunks based on normalized text."""
    groups = _group(
        chunks,
        lambda c, i: {
            "index": i,
            **(
                {"chunk_id": c.get("metadata", {}).get("chunk_id")}
                if c.get("metadata", {}).get("chunk_id")
                else {}
            ),
        },
    )
    return _format(chunks, groups)
