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

def extract_text_blocks_from_pdf(
    filepath: str, exclude_pages: Optional[str] = None
) -> list[dict]:
    """Extract text blocks from ``filepath`` respecting ``exclude_pages``."""

    excluded: set[int] = set()
    if exclude_pages:
        with fitz.open(filepath) as doc:
            total_pages = len(doc)
        excluded = validate_page_exclusions(
            parse_page_ranges(exclude_pages), total_pages, os.path.basename(filepath)
        )

    blocks: Iterable[Block] = (
        blk for page in read_pages(filepath, excluded) for blk in page.blocks
    )
    blocks = strip_artifacts(blocks, None)
    blocks = apply_fallbacks(blocks, filepath, excluded)
    blocks = merge_continuation_blocks(blocks)
    return [asdict(b) for b in blocks]


def _legacy_extract_text_blocks_from_pdf(
    filepath: str, exclude_pages: Optional[str] = None
) -> list[dict]:
    """Backward-compatible shim for older imports."""

    return extract_text_blocks_from_pdf(filepath, exclude_pages)
