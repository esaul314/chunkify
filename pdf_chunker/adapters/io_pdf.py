"""PDF IO adapter providing extraction with optional fallbacks."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any

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
) -> list[dict[str, Any]]:
    """Extract blocks using the legacy parser with optional enhancement."""

    exclude = _format_exclusions(exclude_pages)
    with _env("PDF_CHUNKER_USE_PYMUPDF4LLM", "1" if use_pymupdf4llm else "0"):
        from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf

        return extract_text_blocks_from_pdf(path, exclude)


def _fallback_blocks(
    path: str,
    exclude_pages: Sequence[int] | str | None,
) -> list[dict[str, Any]]:
    """Invoke subprocess-based fallback extraction strategies."""

    from pdf_chunker.extraction_fallbacks import execute_fallback_extraction

    return execute_fallback_extraction(
        path,
        exclude_pages=_format_exclusions(exclude_pages),
    )


def read(
    path: str,
    exclude_pages: Sequence[int] | str | None = None,
    use_pymupdf4llm: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    """Return a ``page_blocks`` document for the given PDF."""

    global _PDFTOTEXT_TIMEOUT
    _PDFTOTEXT_TIMEOUT = timeout
    abs_path = str(Path(path))
    excluded = _excluded(exclude_pages)
    blocks = _primary_blocks(abs_path, sorted(excluded), use_pymupdf4llm)
    filtered = [b for b in blocks if b.get("source", {}).get("page") not in excluded]
    pages = [p for p in _group_blocks(filtered) if p["page"] not in excluded]
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
