from __future__ import annotations

from collections import defaultdict
from itertools import combinations
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


def _subset_record(i: int, j: int, items, fps, pos_fn):
    a, b = (i, j) if len(fps[i]) <= len(fps[j]) else (j, i)
    short_item = items[a]
    return {
        "fp": fps[a],
        "text": short_item.get("text", "")[:80],
        "count": 2,
        "first": pos_fn(short_item, a),
        "second": pos_fn(items[b], b),
    }


def _subset_dups(items: Sequence[Mapping[str, Any]], pos_fn):
    fps = [_fingerprint(str(it.get("text", ""))) for it in items]
    token_map: dict[str, set[int]] = defaultdict(set)
    for idx, fp in enumerate(fps):
        for t in set(fp.split()):
            token_map[t].add(idx)

    pairs = {
        (i, j)
        for idxs in token_map.values()
        for i, j in combinations(sorted(idxs), 2)
        if fps[i] and fps[j]
    }

    return [
        _subset_record(i, j, items, fps, pos_fn)
        for i, j in pairs
        if fps[i] != fps[j] and (fps[i] in fps[j] or fps[j] in fps[i])
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
    return _format(blocks, groups) + _subset_dups(blocks, pos)


def find_dups_chunks(chunks: Sequence[Mapping[str, Any]]):
    """Return duplicate chunks based on normalized text."""
    pos = lambda c, i: {
        "index": i,
        **(
            {"chunk_id": c.get("metadata", {}).get("chunk_id")}
            if c.get("metadata", {}).get("chunk_id")
            else {}
        ),
    }
    groups = _group(chunks, pos)
    return _format(chunks, groups) + _subset_dups(chunks, pos)
