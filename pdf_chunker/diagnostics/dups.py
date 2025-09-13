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


def _subset_dups(items: Sequence[Mapping[str, Any]], pos_fn):
    fps: list[str] = []
    token_map: dict[str, set[int]] = defaultdict(set)
    seen: set[tuple[int, int]] = set()
    records: list[Mapping[str, Any]] = []
    for idx, item in enumerate(items):
        fp = _fingerprint(str(item.get("text", "")))
        tokens = fp.split()
        candidates = {i for t in tokens for i in token_map.get(t, set())}
        for j in candidates:
            other = fps[j]
            pair = (j, idx) if j < idx else (idx, j)
            if fp and other and fp != other and (fp in other or other in fp) and pair not in seen:
                short_fp, short_item = (fp, item) if len(fp) <= len(other) else (other, items[j])
                records.append(
                    {
                        "fp": short_fp,
                        "text": short_item.get("text", "")[:80],
                        "count": 2,
                        "first": pos_fn(items[j], j) if len(fp) > len(other) else pos_fn(item, idx),
                        "second": pos_fn(item, idx)
                        if len(fp) > len(other)
                        else pos_fn(items[j], j),
                    }
                )
                seen.add(pair)
        fps.append(fp)
        for t in set(tokens):
            token_map[t].add(idx)
    return records


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
