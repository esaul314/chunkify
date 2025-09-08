"""High-level PDF block extraction orchestrator."""

from __future__ import annotations

import os
import logging
from dataclasses import asdict
from typing import Iterable, Optional

import fitz

from .page_utils import validate_page_exclusions, parse_page_ranges
from .page_artifacts import strip_artifacts
from .pdf_blocks import Block, read_pages, merge_continuation_blocks
from .fallbacks import apply_fallbacks

logger = logging.getLogger(__name__)

def _excluded_pages(filepath: str, exclude: Optional[str]) -> set[int]:
    if not exclude:
        return set()
    with fitz.open(filepath) as doc:
        total = len(doc)
    return validate_page_exclusions(
        parse_page_ranges(exclude), total, os.path.basename(filepath)
    )


def extract_text_blocks_from_pdf(
    filepath: str, exclude_pages: Optional[str] = None
) -> list[dict]:
    """Extract text blocks from ``filepath`` respecting ``exclude_pages``."""

    excluded = _excluded_pages(filepath, exclude_pages)
    pipeline = merge_continuation_blocks(
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
    return [asdict(b) for b in pipeline]


def _legacy_extract_text_blocks_from_pdf(
    filepath: str, exclude_pages: Optional[str] = None
) -> list[dict]:
    """Backward-compatible shim for older imports."""

    return extract_text_blocks_from_pdf(filepath, exclude_pages)
