from __future__ import annotations

from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


def _page_key(block: Dict[str, Any]) -> int:
    """Page number for grouping; defaults to 0 when missing."""
    return block.get("source", {}).get("page", 0)


def _group_blocks(blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group blocks by page in ascending order."""
    key = _page_key
    sorted_blocks = sorted(blocks, key=key)
    return [{"page": page, "blocks": list(group)} for page, group in groupby(sorted_blocks, key)]


def _extract_blocks(path: str, exclude_pages: str | None) -> List[Dict[str, Any]]:
    return extract_text_blocks_from_pdf(path, exclude_pages)


def read(path: str, exclude_pages: str | None = None) -> Dict[str, Any]:
    """Return a page_blocks document for the given PDF."""
    abs_path = str(Path(path))
    blocks = _extract_blocks(abs_path, exclude_pages)
    return {
        "type": "page_blocks",
        "source_path": abs_path,
        "pages": _group_blocks(blocks),
    }
