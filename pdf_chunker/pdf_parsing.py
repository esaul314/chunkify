# pdf_parsing.py

import os
import sys
import re
import fitz  # PyMuPDF
from .text_cleaning import clean_text, HYPHEN_CHARS_ESC
from .heading_detection import _detect_heading_fallback
from .page_utils import parse_page_ranges, validate_page_exclusions
from .page_artifacts import (
    is_page_artifact_text,
    remove_page_artifact_lines,
    strip_page_artifact_suffix,
)
from .extraction_fallbacks import (
    _detect_language,
    _assess_text_quality,
    _extract_with_pdftotext,
    _extract_with_pdfminer,
    PDFMINER_AVAILABLE,
)
from .pymupdf4llm_integration import (
    extract_with_pymupdf4llm,
    is_pymupdf4llm_available,
    PyMuPDF4LLMExtractionError,
    should_apply_pymupdf4llm_cleaning,
    PyMuPDF4LLMExtractionError,
)

from typing import List, Dict, Any, Tuple


BULLET_CHARS = "*•◦▪‣·●◉○‧"
BULLET_CHARS_ESC = re.escape(BULLET_CHARS)


def _is_bullet_continuation(curr: str, nxt: str) -> bool:
    return curr.rstrip().endswith(tuple(BULLET_CHARS)) and nxt[:1].islower()

  
def _starts_with_bullet(text: str) -> bool:
    return text.lstrip().startswith(tuple(BULLET_CHARS))


def _is_bullet_list_pair(curr: str, nxt: str) -> bool:
    colon_bullet = re.search(rf":\s*[{BULLET_CHARS_ESC}]", curr)
    return _starts_with_bullet(nxt) and (
        _starts_with_bullet(curr)
        or any(_starts_with_bullet(line) for line in curr.splitlines())
        or colon_bullet is not None
    )


def _is_indented_continuation(curr: dict, nxt: dict) -> bool:
    curr_bbox = curr.get("bbox")
    next_bbox = nxt.get("bbox")
    if not curr_bbox or not next_bbox:
        return False
    curr_x0, _, _, curr_y1 = curr_bbox
    next_x0, next_y0, _, _ = next_bbox
    vertical_gap = next_y0 - curr_y1
    indent_diff = next_x0 - curr_x0
    return indent_diff > 10 and vertical_gap < 8

  
def _should_merge_blocks(
    curr_block: Dict[str, Any], next_block: Dict[str, Any]
) -> Tuple[bool, str]:
    """Determine if two blocks should be merged and return the reason."""
    curr_text = curr_block.get("text", "").strip()
    next_text = next_block.get("text", "").strip()

    if not curr_text or not next_text:
        logger.debug(
            f"Merge check: Empty text - curr: {bool(curr_text)}, next: {bool(next_text)}"
        )
        return False, "empty_text"

    curr_page = curr_block.get("source", {}).get("page")
    next_page = next_block.get("source", {}).get("page")

    logger.debug(f"Merge check: Pages curr={curr_page}, next={next_page}")
    logger.debug(
        f"Merge check: Text endings - curr: '{curr_text[-10:]}', next: '{next_text[:10]}'"
    )

    # Check for quote-related splitting issues
    curr_has_quote = '"' in curr_text or "'" in curr_text
    next_has_quote = '"' in next_text or "'" in next_text

    if curr_has_quote or next_has_quote:
        logger.debug(
            f"Merge check: Quote detection - curr: {curr_has_quote}, next: {next_has_quote}"
        )

        # Special handling for quoted text that may have been incorrectly split
        if _is_quote_continuation(curr_text, next_text):
            logger.debug("Merge decision: QUOTE_CONTINUATION")
            return True, "quote_continuation"

    if _is_bullet_continuation(curr_text, next_text):
        logger.debug("Merge decision: BULLET_CONTINUATION")
        return True, "bullet_continuation"

    if _is_bullet_list_pair(curr_text, next_text):
        logger.debug("Merge decision: BULLET_LIST")
        return True, "bullet_list"

    if _is_indented_continuation(
        curr_block, next_block
    ) and not _detect_heading_fallback(next_text):
        logger.debug("Merge decision: INDENTED_CONTINUATION")
        return True, "indented_continuation"

    hyphen_pattern = rf"[{HYPHEN_CHARS_ESC}]$"
    double_hyphen_pattern = rf"[{HYPHEN_CHARS_ESC}]{{2,}}$"
    if (
        re.search(hyphen_pattern, curr_text)
        and not re.search(double_hyphen_pattern, curr_text)
        and next_text
        and next_text[0].islower()
    ):
        logger.debug("Merge decision: HYPHENATED_CONTINUATION")
        return True, "hyphenated_continuation"

    elif (
        curr_page == next_page
        and not curr_text.endswith((".", "!", "?", ":", ";"))
        and next_text
        and next_text[0].islower()
    ):
        logger.debug("Merge decision: SAME_PAGE_CONTINUATION")
        return True, "sentence_continuation"

    # Case 3: Cross-page sentence continuation (no punctuation at end)
    # Enhanced to be more careful with quoted text
    elif (
        curr_text
        and next_text
        and not curr_text.endswith((".", "!", "?"))
        and curr_page != next_page
        and not _looks_like_quote_boundary(curr_text, next_text)
        and not _detect_heading_fallback(next_text)
    ):
        logger.debug("Merge decision: CROSS_PAGE_CONTINUATION")
        return True, "sentence_continuation"

    logger.debug("Merge decision: NO_MERGE")
    return False, "no_merge"


def _is_quote_continuation(curr_text: str, next_text: str) -> bool:
    """Check if the next block is a continuation of quoted text from current block."""
    import re

    # Look for incomplete quotes in current text
    curr_open_quotes = curr_text.count('"') - curr_text.count('\\"')
    curr_open_single = curr_text.count("'") - curr_text.count("\\'")

    # If current text has unmatched opening quotes, next might be continuation
    if curr_open_quotes % 2 == 1 or curr_open_single % 2 == 1:
        # Check if next text starts in a way that suggests quote continuation
        if not next_text[0].isupper() or next_text.startswith(
            ("and", "but", "or", "so", "yet", "for")
        ):
            return True

    # Look for patterns where quotes were split incorrectly
    # Pattern: current ends with quote, next starts with comma/period + space + text
    if curr_text.endswith('"') and re.match(r"^[,.;:]\s+[a-z]", next_text):
        return True

    # Pattern: current ends mid-sentence, next starts with quote
    if (
        not curr_text.endswith((".", "!", "?"))
        and next_text.startswith('"')
        and len(next_text) > 1
        and next_text[1].islower()
    ):
        return True

    return False


def _looks_like_quote_boundary(curr_text: str, next_text: str) -> bool:
    """Check if the boundary between texts looks like a legitimate quote boundary."""
    # If current ends with closing quote and punctuation, and next starts with capital
    if (
        curr_text.endswith(('."', ".'", '!"', "!'", '?"', "?'"))
        and next_text
        and next_text[0].isupper()
    ):
        return True

    # If current ends with quote and next starts with attribution
    attribution_starters = ["said", "asked", "replied", "continued", "added", "noted"]
    if curr_text.endswith(('"', "'")) and any(
        next_text.lower().startswith(starter) for starter in attribution_starters
    ):
        return True

    return False


import logging

logger = logging.getLogger(__name__)


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
        if (
            words
            and len(words) <= max_words
            and any(any(c.isdigit() for c in w) for w in words)
        ):
            return True
    return False


def extract_blocks_from_page(page, page_num, filename) -> list[dict]:
    """
    Extract and classify text blocks from a PDF page,
    filtering out margin artifacts and ensuring all blocks have a proper source dictionary.
    """
    page_height = page.rect.height
    raw_blocks = page.get_text("blocks")
    filtered = [b for b in raw_blocks if not is_artifact_block(b, page_height)]

    structured = []
    for b in filtered:
        raw_text = b[4]
        logger.debug(f"Raw block text before cleaning: {repr(raw_text[:50])}")

        # Remove obvious header/footer lines before full cleaning
        raw_text = remove_page_artifact_lines(raw_text, page_num)

        block_text = clean_text(raw_text)
        logger.debug(f"Block text after cleaning: {repr(block_text[:50])}")

        if not block_text:
            continue

        # Filter out headers, footers, and similar page artifacts
        if is_page_artifact({"text": block_text}, page_num):
            logger.debug(
                f"Skipping page artifact on page {page_num}: {repr(block_text)}"
            )
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
        structured.append(
            {
                "type": block_type,
                "text": block_text,
                "language": _detect_language(block_text),
                "source": {"filename": filename, "page": page_num, "location": None},
                "bbox": b[:4],
            }
        )

    return structured


def is_page_artifact(block: dict, page_num: int) -> bool:
    """
    Detect headers, footers, and page numbers that should be filtered out.

    Args:
        block: Text block with metadata
        page_num: Current page number

    Returns:
        True if block appears to be a page artifact
    """
    text = block.get("text", "").strip()
    return is_page_artifact_text(text, page_num)


def merge_continuation_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge blocks that are continuations of each other."""
    if not blocks:
        return blocks

    logger.info(f"Starting block merging: {len(blocks)} input blocks")

    merged_blocks = []
    i = 0
    merge_count = 0

    while i < len(blocks):
        current_block = blocks[i]
        current_text = current_block.get("text", "").strip()

        preview = current_text[:50].replace(chr(10), "\n")
        logger.debug(
            f"Processing block {i}: {len(current_text)} chars, preview: {preview}"
        )

        # Look ahead for potential merges
        j = i + 1
        merged_any = False

        while j < len(blocks):
            next_block = blocks[j]
            next_text = next_block.get("text", "").strip()

            should_merge, merge_reason = _should_merge_blocks(current_block, next_block)

            if should_merge:
                logger.debug(f"MERGE: Block {i} + Block {j} (reason: {merge_reason})")
                before_i = current_text[:30].replace(chr(10), "\n")
                logger.debug("  Before merge - Block %s: %s", i, before_i)
                before_j = next_text[:30].replace(chr(10), "\n")
                logger.debug("  Before merge - Block %s: %s", j, before_j)

                # Perform the merge
                if merge_reason == "hyphenated_continuation":
                    merged_text = (
                        re.sub(rf"[{HYPHEN_CHARS_ESC}]$", "", current_text) + next_text
                    )
                elif merge_reason == "sentence_continuation":
                    merged_text = current_text + " " + next_text
                elif merge_reason == "bullet_continuation":
                    merged_text = current_text.rstrip(" *•") + " " + next_text
                elif merge_reason == "bullet_list":
                    current_text = re.sub(
                        rf":\s*(?=[{BULLET_CHARS_ESC}])", ":\n", current_text
                    )
                    merged_text = current_text + "\n" + next_text
                elif merge_reason == "indented_continuation":
                    merged_text = current_text + "\n" + next_text
                else:
                    merged_text = current_text + " " + next_text

                after_merge = merged_text[:50].replace(chr(10), "\n")
                logger.debug("  After merge: %s", after_merge)
                current_block["text"] = merged_text
                current_text = merged_text
                merge_count += 1
                merged_any = True
                j += 1
            else:
                logger.debug(
                    f"NO MERGE: Block {i} + Block {j} (different pages/contexts)"
                )
                break

        merged_blocks.append(current_block)
        i = j if merged_any else i + 1

    logger.info(
        f"Block merging complete: {len(blocks)} → {len(merged_blocks)} blocks ({merge_count} merges)"
    )

    # Log final block statistics
    for idx, block in enumerate(merged_blocks):
        text = block.get("text", "")
        text_preview = text[:100].replace("\n", "\\n")
        logger.debug(f"Final block {idx}: {len(text)} chars - {text_preview}...")

    return merged_blocks


def extract_text_blocks_from_pdf(
    filepath: str, exclude_pages: str = None
) -> list[dict]:
    """
    Extract structured text from a PDF using traditional extraction with optional PyMuPDF4LLM text cleaning.

    This simplified approach uses traditional font-based extraction for all structural analysis
    (headings, block boundaries, page metadata) and optionally applies PyMuPDF4LLM's superior
    text cleaning to improve text quality without affecting document structure.

    Enhanced with improved page boundary handling:
    - Better detection and filtering of headers, footers, and page artifacts
    - Improved text flow reconstruction across page boundaries
    - Enhanced sentence continuation handling
    - Debugging output for text flow analysis

    Preserves all existing functionality including page exclusion, heading detection,
    and error handling while optionally enhancing text quality with PyMuPDF4LLM.
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info(f"Starting PDF text extraction from: {filepath}")
    logger.info(
        f"PyMuPDF4LLM enhancement: {'enabled' if os.getenv('PDF_CHUNKER_USE_PYMUPDF4LLM','').lower() not in ('false','0','no','off') else 'disabled'}"
    )

    # Parse excluded pages first
    excluded_pages_set = set()
    if exclude_pages:
        try:
            from .page_utils import parse_page_ranges

            excluded_pages_set = parse_page_ranges(exclude_pages)
            logger.info(f"Excluding pages: {sorted(excluded_pages_set)}")
        except Exception as e:
            logger.error(f"Error parsing page exclusions: {e}")
            excluded_pages_set = set()

    # Always use traditional extraction for structural analysis
    doc = fitz.open(filepath)
    total_pages = len(doc)
    excluded = set()

    if exclude_pages:
        try:
            excluded = validate_page_exclusions(
                excluded_pages_set, total_pages, os.path.basename(filepath)
            )
            logger.debug(f"Validated excluded pages: {sorted(excluded)}")
        except ValueError as e:
            logger.error(f"Error validating page exclusions: {e}")
            excluded = excluded_pages_set

    all_blocks = []
    page_block_counts = {}

    for page_num, page in enumerate(doc, start=1):
        if page_num in excluded:
            logger.debug(f"Skipping excluded page {page_num}")
            continue

        logger.debug(f"Processing page {page_num}/{len(doc)}")
        page_blocks = extract_blocks_from_page(
            page, page_num, os.path.basename(filepath)
        )
        page_block_counts[page_num] = len(page_blocks)
        logger.debug(f"Page {page_num}: extracted {len(page_blocks)} blocks")

        # Log block details for debugging
        for i, block in enumerate(page_blocks):
            text_preview = block.get("text", "")[:100].replace("\n", "\\n")
            logger.debug(
                f"Page {page_num}, Block {i}: {len(block.get('text', ''))} chars - {text_preview}..."
            )
        all_blocks.extend(page_blocks)

    doc.close()
    logger.info(
        f"Raw extraction complete: {len(all_blocks)} total blocks from pages: {sorted(page_block_counts.keys())}"
    )
    logger.debug(f"Page block distribution: {page_block_counts}")

    # Debug: Log sample text from first few blocks to trace cleaning
    for i, block in enumerate(all_blocks[:3]):
        text_preview = block.get("text", "")[:100].replace("\n", "\\n")
        logger.debug(f"Raw block {i} text preview: {repr(text_preview)}")

    # Defensive filter: ensure no excluded pages made it through
    filtered_blocks = []
    for block in all_blocks:
        page = block.get("source", {}).get("page")
        if page is not None and page in excluded:
            logger.warning(f"Filtering out block from excluded page {page}")
            continue
        filtered_blocks.append(block)

    if len(filtered_blocks) != len(all_blocks):
        logger.info(
            f"Filtered out {len(all_blocks) - len(filtered_blocks)} blocks from excluded pages"
        )
    all_blocks = filtered_blocks

    pre_merge_blocks = all_blocks
    logger.debug("Starting block merging process")
    merged_blocks = merge_continuation_blocks(all_blocks)

    logger.debug(f"Total blocks after merging: {len(merged_blocks)}")
    # Log text flow analysis for debugging
    for idx, block in enumerate(merged_blocks[:3]):  # Log first 3 blocks for debugging
        text_preview = block.get("text", "")[:100].replace("\n", "\\n")
        page_info = block.get("source", {})
        logger.debug(
            f"Merged block {idx}: page {page_info.get('page', 'unknown')}, "
            f"type {block.get('type', 'unknown')}, "
            f"text: {repr(text_preview)}"
        )

    # Apply PyMuPDF4LLM enhancement if requested and beneficial
    from .env_utils import use_pymupdf4llm as _use_pymupdf4llm

    use_pymupdf4llm = _use_pymupdf4llm()
    logger.debug(
        f"PDF_CHUNKER_USE_PYMUPDF4LLM environment check: {os.getenv('PDF_CHUNKER_USE_PYMUPDF4LLM', 'not set')}"
    )
    logger.debug(f"use_pymupdf4llm evaluated to: {use_pymupdf4llm}")

    enhancement_stats = {
        "enhanced": 0,
        "failed": 0,
        "skipped": len(merged_blocks),
        "degraded": 0,
        "artifacts_filtered": 0,
    }
    enhanced_blocks = None

    if use_pymupdf4llm and should_apply_pymupdf4llm_cleaning(merged_blocks):
        logger.info("Applying PyMuPDF4LLM enhancement")
        try:
            # Pass excluded pages as a set to the enhancement function
            enhanced_blocks, enhancement_stats = extract_with_pymupdf4llm(
                filepath, exclude_pages=excluded
            )

            # Defensive: filter out any blocks from excluded pages if present
            if enhanced_blocks:
                pre_filter_count = len(enhanced_blocks)
                filtered_enhanced_blocks = []
                for block in enhanced_blocks:
                    page = block.get("source", {}).get("page")
                    if page is not None and page in excluded:
                        logger.warning(
                            f"Filtering out enhanced block from excluded page {page}"
                        )
                        continue
                    filtered_enhanced_blocks.append(block)
                enhanced_blocks = filtered_enhanced_blocks

                if len(enhanced_blocks) != pre_filter_count:
                    logger.info(
                        f"Filtered out {pre_filter_count - len(enhanced_blocks)} enhanced blocks from excluded pages"
                    )

            # If enhancement was successful and high quality, use enhanced blocks
            if enhanced_blocks and enhancement_stats.get("enhanced", 0) > 0:
                if pre_merge_blocks and len(pre_merge_blocks) == len(enhanced_blocks):
                    for eb, ob in zip(enhanced_blocks, pre_merge_blocks):
                        if "bbox" not in eb and ob.get("bbox"):
                            eb["bbox"] = ob["bbox"]
                if enhancement_stats.get("degraded", 0) == 0:
                    logger.info(
                        f"PyMuPDF4LLM enhancement successful: {enhancement_stats}"
                    )
                    merged_blocks = merge_continuation_blocks(enhanced_blocks)
                else:
                    logger.warning(
                        "PyMuPDF4LLM enhancement quality degraded, falling back to traditional text cleaning"
                    )
                    logger.debug(
                        "Ensuring traditional text cleaning pipeline is used for all blocks"
                    )
                    # Re-apply traditional text cleaning to all blocks
                    for block in merged_blocks:
                        if "text" in block:
                            from .text_cleaning import clean_text

                            logger.debug(
                                f"Re-cleaning block text with traditional pipeline: {repr(block['text'][:50])}"
                            )
                            block["text"] = clean_text(block["text"])
                            logger.debug(
                                f"After traditional cleaning: {repr(block['text'][:50])}"
                            )
            else:
                logger.warning(
                    "PyMuPDF4LLM enhancement failed, falling back to traditional text cleaning"
                )
                logger.debug(
                    "Ensuring traditional text cleaning pipeline is used for all blocks"
                )
                # Re-apply traditional text cleaning to all blocks
                for block in merged_blocks:
                    if "text" in block:
                        from .text_cleaning import clean_text

                        logger.debug(
                            f"Re-cleaning block text with traditional pipeline: {repr(block['text'][:50])}"
                        )
                        block["text"] = clean_text(block["text"])
                        logger.debug(
                            f"After traditional cleaning: {repr(block['text'][:50])}"
                        )
        except Exception as e:
            logger.error(f"PyMuPDF4LLM enhancement failed with error: {e}")
            logger.info("Falling back to traditional text cleaning pipeline")
            logger.debug(
                "Ensuring traditional text cleaning pipeline is used for all blocks"
            )
            # Re-apply traditional text cleaning to all blocks
            for block in merged_blocks:
                if "text" in block:
                    from .text_cleaning import clean_text

                    logger.debug(
                        f"Re-cleaning block text with traditional pipeline: {repr(block['text'][:50])}"
                    )
                    block["text"] = clean_text(block["text"])
                    logger.debug(
                        f"After traditional cleaning: {repr(block['text'][:50])}"
                    )
    elif use_pymupdf4llm:
        logger.info("PyMuPDF4LLM enhancement skipped - not beneficial for this content")
        logger.debug("Using traditional text cleaning pipeline")
        enhancement_stats = {
            "enhanced": 0,
            "failed": 0,
            "skipped": len(merged_blocks),
            "degraded": 0,
            "artifacts_filtered": 0,
        }
    else:
        logger.debug("PyMuPDF4LLM disabled by environment variable")
        enhancement_stats = {
            "enhanced": 0,
            "failed": 0,
            "skipped": len(merged_blocks),
            "degraded": 0,
            "artifacts_filtered": 0,
        }

    # Log final enhancement statistics
    logger.info(f"PyMuPDF4LLM enhancement completed: {enhancement_stats}")

    # Assess text quality and apply fallbacks if needed
    text_blob = "\n".join(block["text"] for block in merged_blocks)
    quality = _assess_text_quality(text_blob)

    logger.debug(
        f"Text quality assessment: score={quality.get('quality_score', 0):.2f}"
    )

    if quality["quality_score"] < 0.7:
        logger.warning(
            f"Low quality score ({quality['quality_score']:.2f}), attempting fallback extraction"
        )
        fallback = _extract_with_pdftotext(filepath, exclude_pages)
        if fallback:
            logger.info("Using pdftotext fallback extraction")
            merged_blocks = merge_continuation_blocks(fallback)
        elif PDFMINER_AVAILABLE:
            fallback = _extract_with_pdfminer(filepath, exclude_pages)
            if fallback:
                logger.info("Using pdfminer fallback extraction")
                merged_blocks = merge_continuation_blocks(fallback)

    # --- Ensure all blocks have a proper source dictionary with filename, page, and location ---
    # If PyMuPDF4LLM enhancement was applied, propagate page numbers from original blocks if missing
    if (
        merged_blocks
        and isinstance(merged_blocks[0], dict)
        and "source" in merged_blocks[0]
    ):
        # Try to propagate page numbers from original blocks to enhanced blocks if missing
        # Build a list of original page numbers for each block index
        original_pages = []
        for block in merged_blocks:
            page_val = block.get("source", {}).get("page")
            original_pages.append(page_val)
        # If any page is None, try to assign sequentially (fallback)
        for idx, block in enumerate(merged_blocks):
            if "source" not in block or not isinstance(block["source"], dict):
                block["source"] = {}
            if "filename" not in block["source"]:
                block["source"]["filename"] = os.path.basename(filepath)
            # Assign page number if missing or None
            if "page" not in block["source"] or block["source"]["page"] is None:
                if idx < len(original_pages) and original_pages[idx] is not None:
                    block["source"]["page"] = original_pages[idx]
                else:
                    block["source"]["page"] = idx + 1
            block["source"]["location"] = None
    else:
        for idx, block in enumerate(merged_blocks):
            if "source" not in block or not isinstance(block["source"], dict):
                block["source"] = {}
            if "filename" not in block["source"]:
                block["source"]["filename"] = os.path.basename(filepath)
            if "page" not in block["source"] or block["source"]["page"] is None:
                block["source"]["page"] = idx + 1
            block["source"]["location"] = None

    # Final defensive filter: remove any blocks from excluded pages
    filtered_blocks = []
    for block in merged_blocks:
        page = block.get("source", {}).get("page")
        if page is not None and page in excluded:
            continue
        filtered_blocks.append(block)

    return filtered_blocks
