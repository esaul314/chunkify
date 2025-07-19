# pdf_parsing.py

import os
import sys
import re
import fitz  # PyMuPDF
from .text_cleaning import clean_text
from .heading_detection import _detect_heading_fallback
from .page_utils import parse_page_ranges, validate_page_exclusions
from .extraction_fallbacks import (
    _detect_language,
    _assess_text_quality,
    _extract_with_pdftotext,
    _extract_with_pdfminer,
    PDFMINER_AVAILABLE
)


def is_artifact_block(block, page_height, frac=0.15, max_words=6):
    """
    Detect small numeric artifact blocks near page margins:
    - Block positioned within top or bottom 'frac' of page height,
    - Contains a digit and at most 'max_words' words.
    """
    # Unpack first five elements: x0, y0, x1, y1, raw_text
    x0, y0, x1, y1, raw_text = block[:5]
    # Check if block sits in the margin zones
    if y0 < page_height * frac or y0 > page_height * (1 - frac):
        cleaned = clean_text(raw_text)
        words = cleaned.split()
        if words and len(words) <= max_words and any(any(c.isdigit() for c in w) for w in words):
            return True
    return False


def extract_blocks_from_page(page, page_num, filename) -> list[dict]:
    """
    Extract and classify text blocks from a PDF page,
    filtering out margin artifacts.
    """
    page_height = page.rect.height
    raw_blocks = page.get_text("blocks")
    filtered = [b for b in raw_blocks if not is_artifact_block(b, page_height)]

    structured = []
    for b in filtered:
        raw_text = b[4]
        block_text = clean_text(raw_text)
        if not block_text:
            continue

        # Determine heading via font flags or fallback
        is_heading = False
        if len(block_text.split()) < 15:
            try:
                block_dict = page.get_text("dict", clip=b[:4])["blocks"][0]
                spans = block_dict["lines"][0]["spans"]
                is_heading = any(span.get("flags", 0) & 2 for span in spans)
            except (KeyError, IndexError, TypeError):
                is_heading = _detect_heading_fallback(block_text)

        block_type = "heading" if is_heading else "paragraph"
        structured.append({
            "type": block_type,
            "text": block_text,
            "language": _detect_language(block_text),
            "source": {"filename": filename, "page": page_num}
        })

    return structured


def merge_continuation_blocks(blocks: list[dict]) -> list[dict]:
    """Merge hyphenated or split words across consecutive blocks."""
    merged = []
    skip_next = False

    for i in range(len(blocks) - 1):
        if skip_next:
            skip_next = False
            continue

        curr_text = blocks[i]["text"].strip()
        next_text = blocks[i + 1]["text"].strip()

        if curr_text.endswith("-") and next_text and next_text[0].islower():
            merged_block = blocks[i].copy()
            merged_block["text"] = curr_text[:-1] + next_text
            merged.append(merged_block)
            skip_next = True
        else:
            merged.append(blocks[i])

    if not skip_next and blocks:
        merged.append(blocks[-1])
    return merged


def extract_text_blocks_from_pdf(filepath: str, exclude_pages: str = None) -> list[dict]:
    """
    Extract structured text from a PDF, skipping small numeric header/footer artifacts
    and applying fallback extraction on low-quality pages.
    """
    doc = fitz.open(filepath)
    excluded = set()

    if exclude_pages:
        try:
            excluded = validate_page_exclusions(
                parse_page_ranges(exclude_pages), len(doc), os.path.basename(filepath)
            )
        except ValueError as e:
            print(f"Error parsing page exclusions: {e}", file=sys.stderr)

    all_blocks = []
    for page_num, page in enumerate(doc, start=1):
        if page_num in excluded:
            continue
        all_blocks.extend(
            extract_blocks_from_page(page, page_num, os.path.basename(filepath))
        )
    doc.close()

    merged_blocks = merge_continuation_blocks(all_blocks)
    text_blob = "\n".join(block["text"] for block in merged_blocks)
    quality = _assess_text_quality(text_blob)

    if quality["quality_score"] < 0.7:
        fallback = _extract_with_pdftotext(filepath, exclude_pages)
        if fallback:
            return merge_continuation_blocks(fallback)
        if PDFMINER_AVAILABLE:
            fallback = _extract_with_pdfminer(filepath, exclude_pages)
            if fallback:
                return merge_continuation_blocks(fallback)

    return merged_blocks
