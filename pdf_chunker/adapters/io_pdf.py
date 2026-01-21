"""PDF IO adapter providing extraction with optional fallbacks."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any

import fitz  # PyMuPDF

from pdf_chunker.page_utils import parse_page_ranges


_PDFTOTEXT_TIMEOUT = 60


def _page_key(block: dict[str, Any]) -> int:
    """Page number for grouping; defaults to 0 when missing."""

    return block.get("source", {}).get("page", 0)


def _sorted_blocks(blocks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return blocks sorted by page and original order."""

    return [
        block
        for _, block in sorted(
            enumerate(blocks),
            key=lambda t: (
                _page_key(t[1]),
                t[1].get("source", {}).get("index", t[0]),
            ),
        )
    ]


def _group_blocks(blocks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group blocks by page in ascending order."""

    key = _page_key
    sorted_blocks = _sorted_blocks(blocks)
    grouped = groupby(sorted_blocks, key)
    return [{"page": p, "blocks": list(g)} for p, g in grouped]


def _excluded(pages: Sequence[int] | str | None) -> set[int]:
    """Parse ``pages`` spec into a set of page numbers."""

    if pages is None or pages == "":
        return set()
    if isinstance(pages, str):
        return parse_page_ranges(pages)
    return {int(p) for p in pages}


def _format_exclusions(pages: Sequence[int] | str | None) -> str | None:
    """Return comma-separated pages for legacy extractor compatibility."""

    if pages is None:
        return None
    if isinstance(pages, str):
        return pages
    return ",".join(str(p) for p in pages)


@contextmanager
def _env(name: str, value: str):
    """Temporarily set an environment variable."""

    original = os.getenv(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def _primary_blocks(
    path: str,
    exclude_pages: Sequence[int] | str | None,
    use_pymupdf4llm: bool,
    *,
    interactive: bool = False,
    footer_margin: float | None = None,
    header_margin: float | None = None,
) -> list[dict[str, Any]]:
    """Extract blocks using the legacy parser with optional enhancement."""

    exclude = _format_exclusions(exclude_pages)
    with _env("PDF_CHUNKER_USE_PYMUPDF4LLM", "1" if use_pymupdf4llm else "0"):
        from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf

        return [
            asdict(b)
            for b in extract_text_blocks_from_pdf(
                path, exclude,
                interactive=interactive,
                footer_margin=footer_margin,
                header_margin=header_margin,
            )
        ]


def _fallback_blocks(
    path: str,
    exclude_pages: Sequence[int] | str | None,
) -> list[dict[str, Any]]:
    """Invoke subprocess-based fallback extraction strategies."""

    from pdf_chunker.fallbacks import execute_fallback_extraction

    return execute_fallback_extraction(
        path,
        exclude_pages=_format_exclusions(exclude_pages),
    )


def _page_numbers(path: str) -> range:
    """Enumerate all PDF page numbers using PyMuPDF."""

    with fitz.open(path) as doc:
        return range(1, doc.page_count + 1)


def _all_blocks(
    path: str,
    excluded: set[int],
    use_pymupdf4llm: bool,
    *,
    interactive: bool = False,
    footer_margin: float | None = None,
    header_margin: float | None = None,
) -> list[dict[str, Any]]:
    """Return primary blocks, falling back to second extractor for gaps."""

    primary = _primary_blocks(
        path, sorted(excluded), use_pymupdf4llm,
        interactive=interactive,
        footer_margin=footer_margin,
        header_margin=header_margin,
    )
    existing = {b.get("source", {}).get("page") for b in primary}
    missing = [p for p in _page_numbers(path) if p not in excluded and p not in existing]
    if not missing:
        return primary
    fallback = _fallback_blocks(path, sorted(excluded))
    return primary + [b for b in fallback if b.get("source", {}).get("page") in missing]


def _ensure_all_pages(
    path: str, pages: list[dict[str, Any]], excluded: set[int]
) -> list[dict[str, Any]]:
    """Append empty entries for missing pages."""

    existing = {p["page"]: p["blocks"] for p in pages}
    return [
        {
            "page": p,
            "blocks": existing.get(p, []),
        }
        for p in _page_numbers(path)
        if p not in excluded
    ]


def _filter_by_zones(
    path: str,
    blocks: list[dict[str, Any]],
    footer_margin: float | None,
    header_margin: float | None,
) -> list[dict[str, Any]]:
    """Filter out blocks in header/footer zones using positional data.
    
    This uses the bbox coordinates stored in block['bbox'] (top-level) to
    determine if a block falls within the header or footer zones.
    """
    if not footer_margin and not header_margin:
        return blocks
    
    # Get page heights from the PDF
    page_heights: dict[int, float] = {}
    try:
        with fitz.open(path) as doc:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_heights[page_num + 1] = page.rect.height
    except Exception:
        return blocks  # Can't filter without page dimensions
    
    filtered = []
    for block in blocks:
        source = block.get("source", {})
        page = source.get("page")
        # bbox is at top level of block dict, not in source
        bbox = block.get("bbox")
        
        # If no positioning info, keep the block
        if not bbox or page not in page_heights:
            filtered.append(block)
            continue
        
        page_height = page_heights[page]
        y0, y1 = bbox[1], bbox[3]  # bbox is (x0, y0, x1, y1)
        
        # Check if block is in header zone (top of page)
        if header_margin and y0 < header_margin:
            continue  # Skip header block
        
        # Check if block is in footer zone (bottom of page)
        if footer_margin and y1 > (page_height - footer_margin):
            continue  # Skip footer block
        
        filtered.append(block)
    
    return filtered


def read(
    path: str,
    exclude_pages: Sequence[int] | str | None = None,
    use_pymupdf4llm: bool = False,
    timeout: int = 60,
    *,
    interactive: bool = False,
    footer_margin: float | None = None,
    header_margin: float | None = None,
) -> dict[str, Any]:
    """Return a ``page_blocks`` document for the given PDF.

    Args:
        path: Path to PDF file
        exclude_pages: Page numbers/ranges to exclude
        use_pymupdf4llm: Whether to use PyMuPDF4LLM enhancement
        timeout: Timeout for pdftotext subprocess
        interactive: If True, skip aggressive footer detection to allow
            downstream interactive confirmation
        footer_margin: Points from bottom to exclude as footer zone
        header_margin: Points from top to exclude as header zone
    """

    global _PDFTOTEXT_TIMEOUT
    _PDFTOTEXT_TIMEOUT = timeout
    abs_path = str(Path(path))
    excluded = _excluded(exclude_pages)
    # Zone margins are now applied at extraction time (before block merging)
    blocks = _all_blocks(
        abs_path, excluded, use_pymupdf4llm,
        interactive=interactive,
        footer_margin=footer_margin,
        header_margin=header_margin,
    )
    
    filtered = [b for b in blocks if b.get("source", {}).get("page") not in excluded]
    grouped = _group_blocks(filtered)
    pages = _ensure_all_pages(abs_path, grouped, excluded)
    return {"type": "page_blocks", "source_path": abs_path, "pages": pages}


def run_pdftotext(
    cmd: Sequence[str],
    timeout: int | None = None,
) -> CompletedProcess[str]:
    """Execute ``pdftotext`` and capture its output."""

    return run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout or _PDFTOTEXT_TIMEOUT,
    )
