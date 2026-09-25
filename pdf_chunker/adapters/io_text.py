"""Plain-text IO adapter returning canonical PageBlocks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_WEBVTT_HEADER = "WEBVTT"
_CUE_INDEX_RE = re.compile(r"^\d+$")
_TIMESTAMP_PART = r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
_TIMESTAMP_RE = re.compile(
    rf"^(?P<start>{_TIMESTAMP_PART})" rf"(?:\s*-->\s*(?P<end>{_TIMESTAMP_PART})(?:\s+[^\s].*)?)?$"
)
_SPEAKER_RE = re.compile(r"^(?:\[[^\]\r\n]{1,80}\]|[A-Za-z][A-Za-z0-9 .'\-]{0,79}):$")


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _timestamp_source(line: str) -> dict[str, str] | None:
    match = _TIMESTAMP_RE.match(line)
    if not match:
        return None
    start = match.group("start")
    end = match.group("end") or start
    return {"timestamp_start": start, "timestamp_end": end}


def _is_speaker(line: str) -> bool:
    return bool(_SPEAKER_RE.match(line))


def _join_block(lines: list[str]) -> str:
    return " ".join(line.strip() for line in lines if line.strip()).strip()


def _iter_block_records(text: str) -> list[tuple[str, dict[str, str]]]:
    lines = _normalize_text(text).split("\n")
    blocks: list[tuple[str, dict[str, str]]] = []
    current: list[str] = []
    pending_source: dict[str, str] | None = None

    def flush() -> None:
        nonlocal pending_source
        joined = _join_block(current)
        if joined:
            blocks.append((joined, dict(pending_source or {})))
        current.clear()
        pending_source = None

    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        idx += 1

        if not stripped:
            flush()
            continue
        if stripped == _WEBVTT_HEADER or stripped.startswith(f"{_WEBVTT_HEADER} "):
            continue
        next_timestamp = _timestamp_source(lines[idx].strip()) if idx < len(lines) else None
        if _CUE_INDEX_RE.match(stripped) and next_timestamp is not None:
            continue
        timestamp_source = _timestamp_source(stripped)
        if timestamp_source is not None:
            flush()
            pending_source = timestamp_source
            continue
        if current and _is_speaker(stripped):
            flush()
        current.append(stripped)

    flush()
    return blocks


def _iter_blocks(text: str) -> list[str]:
    """Return normalized text blocks without their timing provenance."""
    return [block for block, _ in _iter_block_records(text)]


def read(path: str, exclude_pages: str | None = None, **_: Any) -> dict[str, Any]:
    """Load a plain-text transcript and emit grouped PageBlocks."""
    del exclude_pages

    abs_path = str(Path(path).resolve())
    source_name = Path(abs_path).name
    raw_text = Path(abs_path).read_text(encoding="utf-8")
    blocks = [
        {
            "text": text,
            "source": {
                "filename": source_name,
                "page": 1,
                "location": f"paragraph:{index}",
                **timing,
            },
        }
        for index, (text, timing) in enumerate(_iter_block_records(raw_text), start=1)
    ]
    return {
        "type": "page_blocks",
        "source_path": abs_path,
        "pages": [{"page": 1, "blocks": blocks}],
    }
