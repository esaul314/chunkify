from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from pdf_chunker.passes.transform_log import TransformationLog

_RUN_ID = uuid4().hex
_CALLS: list[str] = []
_TRANSFORM_LOGS: dict[str, list[TransformationLog]] = {}


def _path(step: str) -> Path:
    base = Path("artifacts") / "trace" / _RUN_ID
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{step}.json"


def write_snapshot(step: str, data: Any) -> None:
    """Persist ``data`` for ``step`` under a unique run directory."""
    _path(step).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_inline_styles(step, data)


def _normalize(text: str) -> str:
    table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    return " ".join(text.strip().translate(table).split())


def _items(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, MappingABC):
        if "pages" in payload:
            return [
                {**b, "page": p.get("page_number")}
                for p in payload.get("pages", [])
                for b in p.get("blocks", [])
            ]
        if "items" in payload:
            return list(payload.get("items", []))
    return list(payload) if isinstance(payload, SequenceABC) else []


def _pos(item: Mapping[str, Any], idx: int) -> Mapping[str, Any]:
    return {
        "index": idx,
        **{k: item.get(k) for k in ("page", "bbox") if item.get(k) is not None},
    }


_MISSING = object()


def _span_field(span: Any, name: str) -> Any:
    """Return ``name`` from ``span`` supporting dataclass and mapping inputs."""

    value = getattr(span, name, _MISSING)
    if value is not _MISSING:
        return value
    if isinstance(span, MappingABC):
        return span.get(name)
    return None


def _span_summary(span: Any, text: str) -> Mapping[str, Any] | None:
    """Build a JSON-serializable summary for ``span`` against ``text``."""

    start = _span_field(span, "start")
    end = _span_field(span, "end")
    style = _span_field(span, "style")
    if style is None or start is None or end is None:
        return None

    snippet = text[start:end]
    summary: dict[str, Any] = {
        "style": style,
        "range": [start, end],
        "text": snippet,
    }

    confidence = _span_field(span, "confidence")
    if confidence is not None:
        summary["confidence"] = confidence

    attrs = _span_field(span, "attrs")
    if attrs:
        summary["attrs"] = dict(attrs) if isinstance(attrs, MappingABC) else attrs

    return summary


def _block_inline_styles(block: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return inline-style summaries for ``block`` if present."""

    spans = block.get("inline_styles") or []
    text = block.get("text", "")
    return [
        summary for summary in (_span_summary(span, text) for span in spans) if summary is not None
    ]


def _inline_style_payload(payload: Any) -> dict[str, Any]:
    """Collect inline style spans from ``payload`` for trace emission."""

    blocks = [
        {
            "index": idx,
            **{k: block.get(k) for k in ("page", "bbox") if block.get(k) is not None},
            "text": block.get("text", ""),
            "spans": spans,
        }
        for idx, block in enumerate(_items(payload))
        if (spans := _block_inline_styles(block))
    ]
    return {"blocks": blocks} if blocks else {}


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


def _write_inline_styles(step: str, payload: Any) -> None:
    """Emit inline-style summaries alongside the default trace snapshot."""

    summary = _inline_style_payload(payload)
    if summary:
        _path(f"{step}_inline_styles").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def record_call(step: str) -> None:
    _CALLS.append(step)
    data = {"calls": list(_CALLS), "counts": {s: _CALLS.count(s) for s in set(_CALLS)}}
    _path("calls").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Transformation log collection (Phase 0 of refactoring plan)
# ---------------------------------------------------------------------------


def record_transform_log(step: str, log: TransformationLog) -> None:
    """Record a transformation log for a step.

    Call this from passes that support transformation tracking to build
    an audit trail of all text transformations.
    """
    if step not in _TRANSFORM_LOGS:
        _TRANSFORM_LOGS[step] = []
    _TRANSFORM_LOGS[step].append(log)


def write_transform_logs(step: str) -> None:
    """Write all collected transformation logs for a step."""
    logs = _TRANSFORM_LOGS.get(step, [])
    if not logs:
        return

    data = {
        "step": step,
        "fragment_count": len(logs),
        "fragments": [log.to_dict() for log in logs],
    }
    _path(f"{step}_transforms").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_transform_logs(step: str | None = None) -> dict[str, list[TransformationLog]]:
    """Return collected transformation logs.

    If step is provided, return only logs for that step.
    Otherwise return all logs.
    """
    if step is not None:
        return {step: _TRANSFORM_LOGS.get(step, [])}
    return dict(_TRANSFORM_LOGS)


def clear_transform_logs() -> None:
    """Clear all collected transformation logs (for testing)."""
    _TRANSFORM_LOGS.clear()
