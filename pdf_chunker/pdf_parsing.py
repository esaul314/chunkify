# pdf_parsing.py

import os
import sys
import fitz  # PyMuPDF
from functools import partial
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

def extract_blocks_from_page(page, page_num, filename):
    blocks = []
    page_blocks = page.get_text("blocks")

    for b in page_blocks:
        block_text = clean_text(b[4])
        if not block_text:
            continue

        is_heading = False
        if len(block_text.split()) < 15:
            try:
                block_dict = page.get_text("dict", clip=b[:4])["blocks"][0]
                spans = block_dict['lines'][0]['spans']
                is_heading = any(span.get('flags', 0) & 2 for span in spans)
            except (KeyError, IndexError, TypeError):
                is_heading = _detect_heading_fallback(block_text)

        block_type = "heading" if is_heading else "paragraph"
        blocks.append({
            "type": block_type,
            "text": block_text,
            "language": _detect_language(block_text),
            "source": {"filename": filename, "page": page_num}
        })
    return blocks


def merge_continuation_blocks(blocks):
    merged = []
    skip_next = False

    for i, block in enumerate(blocks[:-1]):
        if skip_next:
            skip_next = False
            continue

        current_text = block["text"].strip()
        next_text = blocks[i + 1]["text"].strip()

        if current_text.endswith('-') and next_text[0].islower():
            merged_text = current_text[:-1] + next_text
            merged_block = block.copy()
            merged_block["text"] = merged_text
            merged.append(merged_block)
            skip_next = True
        else:
            merged.append(block)

    if not skip_next:
        merged.append(blocks[-1])

    return merged


def extract_text_blocks_from_pdf(filepath: str, exclude_pages: str = None) -> list[dict]:
    doc = fitz.open(filepath)
    excluded_pages = set()

    if exclude_pages:
        try:
            excluded_pages = validate_page_exclusions(
                parse_page_ranges(exclude_pages), len(doc), os.path.basename(filepath)
            )
        except ValueError as e:
            print(f"Error parsing page exclusions: {e}", file=sys.stderr)

    all_blocks = []

    for page_num, page in enumerate(doc, start=1):
        if page_num in excluded_pages:
            continue

        all_blocks.extend(extract_blocks_from_page(page, page_num, os.path.basename(filepath)))

    doc.close()

    all_blocks = merge_continuation_blocks(all_blocks)

    quality = _assess_text_quality('\n'.join(block["text"] for block in all_blocks))

    if quality["quality_score"] < 0.7:
        fallback_blocks = _extract_with_pdftotext(filepath, exclude_pages)
        if fallback_blocks:
            return merge_continuation_blocks(fallback_blocks)

        if PDFMINER_AVAILABLE:
            fallback_blocks = _extract_with_pdfminer(filepath, exclude_pages)
            if fallback_blocks:
                return merge_continuation_blocks(fallback_blocks)

    return all_blocks

