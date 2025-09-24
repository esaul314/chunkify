"""High-level PDF block extraction orchestrator with streaming support."""

from __future__ import annotations

import os
import logging
from dataclasses import asdict, replace
from typing import Iterable, Iterator, Tuple

import fitz

from .page_utils import validate_page_exclusions, parse_page_ranges
from .page_artifacts import strip_artifacts
from .pdf_blocks import Block, read_pages, merge_continuation_blocks
from .fallbacks import apply_fallbacks
from .text_cleaning import restore_leading_capitalization

logger = logging.getLogger(__name__)

def _excluded_pages(filepath: str, exclude: str | None) -> set[int]:
    if not exclude:
        return set()
    with fitz.open(filepath) as doc:
        total = len(doc)
    return validate_page_exclusions(
        parse_page_ranges(exclude), total, os.path.basename(filepath)
    )


def _block_pages(block: Block) -> Tuple[int, ...]:
    source = block.source if isinstance(block.source, dict) else {}
    page_range = source.get("page_range")
    if (
        isinstance(page_range, tuple)
        and len(page_range) == 2
        and all(isinstance(num, int) for num in page_range)
    ):
        start, end = page_range
        return tuple(range(start, end + 1))

    page = source.get("page")
    return (page,) if isinstance(page, int) else ()


def _restore_page_leading_case(blocks: Iterable[Block]) -> Iterator[Block]:
    seen: set[int] = set()

    for block in blocks:
        pages = _block_pages(block)
        if pages and any(page not in seen for page in pages):
            restored = restore_leading_capitalization(block.text)
            if restored != block.text:
                block = replace(block, text=restored)
        seen.update(pages)
        yield block


def _block_pipeline(filepath: str, excluded: set[int]) -> list[Block]:
    """Return merged blocks for ``filepath`` respecting ``excluded`` pages."""

    merged = merge_continuation_blocks(
        apply_fallbacks(
            strip_artifacts(
                (
                    blk
                    for page in read_pages(filepath, excluded)
                    for blk in page.blocks
                ),
                None,
            ),
            filepath,
            excluded,
        )
    )

    return list(_restore_page_leading_case(merged))


def extract_text_blocks_from_pdf(
    filepath: str, exclude_pages: str | None = None
) -> Iterable[Block]:
    """Yield ``Block`` objects from ``filepath`` respecting ``exclude_pages``.

    A list-based wrapper is available via :func:`extract_text_blocks_from_pdf_list`
    for consumers that require materialised dictionaries. The iterator form
    enables streaming consumption and reduces peak memory usage.
    """

    excluded = _excluded_pages(filepath, exclude_pages)
    return iter(_block_pipeline(filepath, excluded))


def extract_text_blocks_from_pdf_list(
    filepath: str, exclude_pages: str | None = None
) -> list[dict]:
    """Deprecated shim returning a materialised list of block dictionaries."""

    return [asdict(b) for b in _block_pipeline(filepath, _excluded_pages(filepath, exclude_pages))]


def _legacy_extract_text_blocks_from_pdf(
    filepath: str, exclude_pages: str | None = None
) -> list[dict]:
    """Backward-compatible shim for older imports."""

    return extract_text_blocks_from_pdf_list(filepath, exclude_pages)
