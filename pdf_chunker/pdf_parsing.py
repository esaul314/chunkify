"""High-level PDF block extraction orchestrator with streaming support."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, replace

import fitz

from .fallbacks import apply_fallbacks
from .page_artifacts import strip_artifacts
from .page_utils import parse_page_ranges, validate_page_exclusions
from .pdf_blocks import Block, merge_continuation_blocks, read_pages
from .text_cleaning import restore_leading_capitalization

logger = logging.getLogger(__name__)


def _excluded_pages(filepath: str, exclude: str | None) -> set[int]:
    if not exclude:
        return set()
    with fitz.open(filepath) as doc:
        total = len(doc)
    return validate_page_exclusions(parse_page_ranges(exclude), total, os.path.basename(filepath))


def _block_pages(block: Block) -> tuple[int, ...]:
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


def _get_page_heights(filepath: str) -> dict[int, float]:
    """Return a dict mapping page numbers (1-indexed) to heights in points."""
    try:
        import fitz

        page_heights = {}
        with fitz.open(filepath) as doc:
            for i, page in enumerate(doc, start=1):
                page_heights[i] = page.rect.height
        return page_heights
    except Exception:
        return {}


def _block_pipeline(
    filepath: str,
    excluded: set[int],
    *,
    interactive: bool = False,
    footer_margin: float | None = None,
    header_margin: float | None = None,
) -> list[Block]:
    """Return merged blocks for ``filepath`` respecting ``excluded`` pages.

    Args:
        filepath: Path to PDF file
        excluded: Set of page numbers to exclude
        interactive: If True, skip aggressive footer detection to allow
            downstream interactive confirmation
        footer_margin: Points from bottom to exclude as footer zone
        header_margin: Points from top to exclude as header zone
    """
    # Collect page heights for footer zone detection during merge
    page_heights = _get_page_heights(filepath) if footer_margin else None

    merged = merge_continuation_blocks(
        apply_fallbacks(
            strip_artifacts(
                (
                    blk
                    for page in read_pages(
                        filepath,
                        excluded,
                        footer_margin=footer_margin,
                        header_margin=header_margin,
                    )
                    for blk in page.blocks
                ),
                None,
                skip_footer_detection=interactive,
            ),
            filepath,
            excluded,
        ),
        page_heights=page_heights,
        footer_margin=footer_margin,
    )

    return list(_restore_page_leading_case(merged))


def extract_text_blocks_from_pdf(
    filepath: str,
    exclude_pages: str | None = None,
    *,
    interactive: bool = False,
    footer_margin: float | None = None,
    header_margin: float | None = None,
) -> Iterable[Block]:
    """Yield ``Block`` objects from ``filepath`` respecting ``exclude_pages``.

    A list-based wrapper is available via :func:`extract_text_blocks_from_pdf_list`
    for consumers that require materialised dictionaries. The iterator form
    enables streaming consumption and reduces peak memory usage.

    Args:
        filepath: Path to PDF file
        exclude_pages: Page ranges to exclude (e.g., "1-3,5")
        interactive: If True, skip aggressive footer detection to allow
            downstream interactive confirmation
        footer_margin: Points from bottom to exclude as footer zone
        header_margin: Points from top to exclude as header zone
    """

    excluded = _excluded_pages(filepath, exclude_pages)
    return iter(
        _block_pipeline(
            filepath,
            excluded,
            interactive=interactive,
            footer_margin=footer_margin,
            header_margin=header_margin,
        )
    )


def extract_text_blocks_from_pdf_list(
    filepath: str,
    exclude_pages: str | None = None,
    *,
    interactive: bool = False,
    footer_margin: float | None = None,
    header_margin: float | None = None,
) -> list[dict]:
    """Deprecated shim returning a materialised list of block dictionaries."""

    return [
        asdict(b)
        for b in _block_pipeline(
            filepath,
            _excluded_pages(filepath, exclude_pages),
            interactive=interactive,
            footer_margin=footer_margin,
            header_margin=header_margin,
        )
    ]


def _legacy_extract_text_blocks_from_pdf(
    filepath: str, exclude_pages: str | None = None
) -> list[dict]:
    """Backward-compatible shim for older imports."""

    return extract_text_blocks_from_pdf_list(filepath, exclude_pages)
