"""Plain-text IO adapter returning canonical PageBlocks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_WEBVTT_HEADER = "WEBVTT"
_CUE_INDEX_RE = re.compile(r"^\d+$")
_TIMESTAMP_RE = re.compile(
    r"^(?:(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?)"
    r"(?:\s*-->\s*(?:(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?))?$"
)
_SPEAKER_RE = re.compile(r"^[A-Z][A-Za-z0-9 .'\-]{0,79}:$")


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _is_timestamp(line: str) -> bool:
    return bool(_TIMESTAMP_RE.match(line))


def _is_speaker(line: str) -> bool:
    return bool(_SPEAKER_RE.match(line))


def _join_block(lines: list[str]) -> str:
    return " ".join(line.strip() for line in lines if line.strip()).strip()


def _iter_blocks(text: str) -> list[str]:
    lines = _normalize_text(text).split("\n")
    blocks: list[str] = []
    current: list[str] = []
    pending_prefix: list[str] = []

    def flush() -> None:
        joined = _join_block([*pending_prefix, *current])
        if joined and not _is_timestamp(joined):
            blocks.append(joined)
        current.clear()
        pending_prefix.clear()

    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        idx += 1

        if not stripped:
            flush()
            continue
        if stripped == _WEBVTT_HEADER:
            continue
        if _CUE_INDEX_RE.match(stripped) and idx < len(lines) and _is_timestamp(lines[idx].strip()):
            continue
        if _is_timestamp(stripped):
            flush()
            pending_prefix = [stripped]
            continue
        if current and _is_speaker(stripped):
            flush()
        current.append(stripped)

    flush()
    return blocks


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
            },
        }
        for index, text in enumerate(_iter_blocks(raw_text), start=1)
    ]
    return {
        "type": "page_blocks",
        "source_path": abs_path,
        "pages": [{"page": 1, "blocks": blocks}],
    }
