from __future__ import annotations

from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterable, List

# heavy epub parsing imports deferred to runtime to avoid expensive deps


def _spine_map(path: str) -> Dict[str, int]:
    """Map spine locations to their 1-based indices."""
    from pdf_chunker.epub_parsing import list_epub_spines

    return {item["filename"]: item["index"] for item in list_epub_spines(path)}


def _spine_key(block: Dict[str, Any], mapping: Dict[str, int]) -> int:
    """Spine index for grouping; defaults to 0 when missing."""
    location = block.get("source", {}).get("location", "")
    return mapping.get(location, 0)


def _group_blocks(
    blocks: Iterable[Dict[str, Any]], mapping: Dict[str, int]
) -> List[Dict[str, Any]]:
    """Group blocks by spine index in ascending order."""
    key = partial(_spine_key, mapping=mapping)
    sorted_blocks = sorted(blocks, key=key)
    return [
        {"page": page, "blocks": list(group)}
        for page, group in groupby(sorted_blocks, key)
    ]


def _extract_blocks(path: str, exclude_spines: str | None) -> List[Dict[str, Any]]:
    from pdf_chunker.epub_parsing import extract_text_blocks_from_epub

    return extract_text_blocks_from_epub(path, exclude_spines=exclude_spines)


def read(path: str, exclude_pages: str | None = None) -> Dict[str, Any]:
    """Return a page_blocks document for the given EPUB."""
    abs_path = str(Path(path))
    blocks = _extract_blocks(abs_path, exclude_pages)
    mapping = _spine_map(abs_path)
    return {
        "type": "page_blocks",
        "source_path": abs_path,
        "pages": _group_blocks(blocks, mapping),
    }


def describe_epub(path: str) -> Dict[str, str]:
    """Return a lightweight descriptor for an EPUB file."""
    return {
        "type": "epub_document",
        "source_path": str(Path(path).resolve()),
    }


def read_epub(path: str, spine: str | None = None) -> Dict[str, Any]:
    """Compatibility wrapper around :func:`read`."""
    return read(path, exclude_pages=spine)

