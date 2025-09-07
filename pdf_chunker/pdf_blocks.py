from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable, List, Optional
import os

import fitz  # PyMuPDF

from .text_cleaning import clean_text
from .page_artifacts import (
    is_page_artifact_text,
    remove_page_artifact_lines,
)
from .heading_detection import _detect_heading_fallback, TRAILING_PUNCTUATION
from .extraction_fallbacks import default_language

# -- Data models -------------------------------------------------------------------------

@dataclass
class Block:
    text: str
    source: dict
    type: str = "paragraph"
    language: Optional[str] = None
    bbox: Optional[tuple] = None


@dataclass
class PagePayload:
    number: int
    blocks: List[Block]


# -- Page extraction --------------------------------------------------------------------

def _filter_margin_artifacts(blocks, page_height: float) -> list:
    numeric_pattern = r"^[0-9ivxlcdm]+$"

    def is_numeric_fragment(text: str) -> bool:
        import re

        words = text.split()
        return 0 < len(words) <= 6 and all(re.fullmatch(numeric_pattern, w) for w in words)

    filtered: list = []
    for block in blocks:
        x0, y0, x1, y1, raw_text = block[:5]
        if y0 < page_height * 0.15 or y0 > page_height * 0.85:
            cleaned = clean_text(raw_text).strip()
            if is_numeric_fragment(cleaned):
                if filtered and filtered[-1][1] > page_height * 0.85:
                    filtered.pop()
                continue
        filtered.append(block)
    return filtered


def _spans_indicate_heading(spans: list[dict], text: str) -> bool:
    return any(span.get("flags", 0) & 2 for span in spans) and not text.rstrip().endswith(
        TRAILING_PUNCTUATION
    )


def _structured_block(page, block_tuple, page_num, filename) -> Block | None:
    raw_text = block_tuple[4]
    cleaned = clean_text(remove_page_artifact_lines(raw_text, page_num))
    if not cleaned or is_page_artifact_text(cleaned, page_num):
        return None

    is_heading = False
    if len(cleaned.split()) < 15:
        try:
            block_dict = page.get_text("dict", clip=block_tuple[:4])["blocks"][0]
            spans = block_dict["lines"][0]["spans"]
            is_heading = _spans_indicate_heading(spans, cleaned)
        except (KeyError, IndexError, TypeError):
            is_heading = _detect_heading_fallback(cleaned)

    return Block(
        type="heading" if is_heading else "paragraph",
        text=cleaned,
        language=default_language(),
        source={"filename": filename, "page": page_num, "location": None},
        bbox=block_tuple[:4],
    )


def _extract_page_blocks(page, page_num: int, filename: str) -> list[Block]:
    page_height = page.rect.height
    raw_blocks = page.get_text("blocks")
    filtered = _filter_margin_artifacts(raw_blocks, page_height)
    return [
        b
        for block in filtered
        if (b := _structured_block(page, block, page_num, filename)) is not None
    ]


def read_pages(
    filepath: str,
    excluded: set[int],
    extractor: Callable[[fitz.Page, int, str], list[Block]] = _extract_page_blocks,
) -> Iterable[PagePayload]:
    """Yield ``PagePayload`` objects for each non-excluded page."""

    doc = fitz.open(filepath)
    try:
        for page_num, page in enumerate(doc, start=1):
            if page_num in excluded:
                continue
            blocks = extractor(page, page_num, os.path.basename(filepath))
            yield PagePayload(number=page_num, blocks=blocks)
    finally:
        doc.close()


