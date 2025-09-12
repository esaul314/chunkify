from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

_RUN_ID = uuid4().hex
_CALLS: list[str] = []


def _path(step: str) -> Path:
    base = Path("artifacts") / "trace" / _RUN_ID
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{step}.json"


def write_snapshot(step: str, data: Any) -> None:
    """Persist ``data`` for ``step`` under a unique run directory."""
    _path(step).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _normalize(text: str) -> str:
    table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    return " ".join(text.strip().translate(table).split())


def _items(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if "pages" in payload:
            return [
                {**b, "page": p.get("page_number")}
                for p in payload.get("pages", [])
                for b in p.get("blocks", [])
            ]
        if "items" in payload:
            return list(payload.get("items", []))
    return list(payload) if isinstance(payload, Sequence) else []


def _pos(item: Mapping[str, Any], idx: int) -> Mapping[str, Any]:
    return {
        "index": idx,
        **{k: item.get(k) for k in ("page", "bbox") if item.get(k) is not None},
    }


def summarize_duplicates(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for i, it in enumerate(items):
        text = _normalize(str(it.get("text", "")))
        if text:
            groups[text].append(_pos(it, i))
    dups = [
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
    return {"total": len(items), "dups": dups}


def write_dups(step: str, payload: Any) -> None:
    data = summarize_duplicates(_items(payload))
    _path(f"{step}_dups").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def record_call(step: str) -> None:
    _CALLS.append(step)
    data = {"calls": list(_CALLS), "counts": {s: _CALLS.count(s) for s in set(_CALLS)}}
    _path("calls").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
