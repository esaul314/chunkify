"""EPUB IO adapter returning canonical PageBlocks."""

from __future__ import annotations

from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _page_key(mapping: Dict[str, int], block: Dict[str, Any]) -> int:
    """Resolve spine index for grouping.

    Unknown locations are appended after known spine items instead of
    collapsing into page ``0``.
    """
    location = block.get("source", {}).get("location")
    return mapping.get(location, len(mapping) + 1)


def _group_blocks(
    blocks: Iterable[Dict[str, Any]], mapping: Dict[str, int]
) -> List[Dict[str, Any]]:
    """Group blocks by computed spine index."""

    key = lambda blk: _page_key(mapping, blk)
    sorted_blocks = sorted(blocks, key=key)
    return [
        {"page": page, "blocks": list(group)}
        for page, group in groupby(sorted_blocks, key)
    ]


def _excluded_spines(spec: str | None, total: int, filename: str) -> set[int]:
    """Parse and validate spine exclusion specification."""

    if not spec:
        return set()
    try:
        from pdf_chunker.page_utils import parse_page_ranges, validate_page_exclusions

        return validate_page_exclusions(parse_page_ranges(spec), total, filename)
    except ValueError:
        return set()


def read_epub(path: str, spine: str | None = None) -> Dict[str, Any]:
    """Load an EPUB file and emit grouped PageBlocks."""

    import ebooklib
    from ebooklib import epub
    from pdf_chunker.epub_parsing import process_epub_item

    abs_path = str(Path(path).resolve())
    book = epub.read_epub(abs_path)
    filename = Path(abs_path).name
    spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    mapping = {item.get_name(): idx for idx, item in enumerate(spine_items, 1)}
    excluded = _excluded_spines(spine, len(spine_items), filename)

    blocks = [
        block
        for idx, item in enumerate(spine_items, 1)
        if idx not in excluded
        for block in process_epub_item(item, filename)
    ]

    return {
        "type": "page_blocks",
        "source_path": abs_path,
        "pages": _group_blocks(blocks, mapping),
    }


def describe_epub(path: str) -> Dict[str, str]:
    """Return a lightweight descriptor for an EPUB file."""

    return {"type": "epub_document", "source_path": str(Path(path).resolve())}
