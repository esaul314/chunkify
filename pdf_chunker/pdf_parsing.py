# pdf_parsing.py

import os
import re
import logging
from dataclasses import asdict
from functools import reduce
from typing import Optional, Callable, Any, Tuple, Sequence, Mapping

import fitz  # PyMuPDF
try:
    from .text_cleaning import clean_text
except Exception:

    def clean_text(text: str) -> str:
        return text


from .heading_detection import _detect_heading_fallback, TRAILING_PUNCTUATION
from .page_utils import validate_page_exclusions
from .page_artifacts import (
    is_page_artifact_text,
    remove_page_artifact_lines,
    strip_page_artifact_suffix,
)

try:
    from .extraction_fallbacks import (
        default_language,
        _assess_text_quality,
        _extract_with_pdftotext as _extract_with_pdftotext_impl,
        _extract_with_pdfminer as _extract_with_pdfminer_impl,
        PDFMINER_AVAILABLE,
    )
except Exception:

    def default_language() -> str:
        return ""

    def _assess_text_quality(text: str) -> dict[Any, Any]:
        return {}

    def _extract_with_pdftotext_impl(
        filepath: str, exclude_pages: Optional[str] = None
    ) -> list[dict[Any, Any]]:
        return []

    def _extract_with_pdfminer_impl(
        filepath: str, exclude_pages: Optional[str] = None
    ) -> list[dict[Any, Any]]:
        return []

    PDFMINER_AVAILABLE = False

from .pymupdf4llm_integration import (
    extract_with_pymupdf4llm,
    should_apply_pymupdf4llm_cleaning,
)

from .pdf_blocks import Block, merge_continuation_blocks, _extract_page_blocks

logger = logging.getLogger(__name__)


def extract_blocks_from_page(page, page_num, filename) -> list[dict]:
    """Proxy to ``_extract_page_blocks`` returning dictionaries."""
    return [asdict(b) for b in _extract_page_blocks(page, page_num, filename)]

_extract_with_pdftotext: Callable[[str, Optional[str]], list[dict[str, Any]]] = (
    _extract_with_pdftotext_impl
)
_extract_with_pdfminer: Callable[[str, Optional[str]], list[dict[str, Any]]] = (
    _extract_with_pdfminer_impl
)


def _page_count(blocks: Sequence[Mapping[str, Any]]) -> int:
    """Return the number of distinct pages represented in ``blocks``."""
    return len({
        b.get("source", {}).get("page")
        for b in blocks
        if b.get("source", {}).get("page") is not None
    })


def _assess_and_maybe_fallback(
    filepath: str,
    exclude_pages: Optional[str],
    blocks: list[dict[str, Any]],
    quality_score: float,
) -> list[dict[str, Any]]:
    """Return ``blocks`` or a higher-quality fallback extraction."""

    if quality_score >= 0.7:
        return blocks

    candidates: Sequence[tuple[str, Callable[[str, Optional[str]], list[dict[str, Any]]]]] = (
        ("pdftotext", _extract_with_pdftotext),
        ("pdfminer", _extract_with_pdfminer if PDFMINER_AVAILABLE else (lambda *_: [])),
    )

    base_pages = _page_count(blocks)
    for name, extractor in candidates:
        fallback = extractor(filepath, exclude_pages)
        if len(fallback) <= 1:
            logger.warning(
                f"{name} fallback produced {len(fallback)} block(s); keeping original extraction"
            )
            continue
        if _page_count(fallback) < base_pages:
            logger.warning(
                f"{name} fallback covered fewer pages; keeping original extraction"
            )
            continue
        score = _assess_text_quality(
            "\n".join(b.get("text", "") for b in fallback)
        )["quality_score"]
        if score <= quality_score:
            logger.warning(
                f"{name} fallback quality {score:.2f} not better than original {quality_score:.2f}"
            )
            continue
        logger.info(f"Using {name} fallback extraction")
        return [
            asdict(b)
            for b in merge_continuation_blocks(Block(**fb) for fb in fallback)
        ]

    return blocks


def extract_text_blocks_from_pdf(filepath: str, exclude_pages: Optional[str] = None) -> list[dict]:
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
        page_blocks = extract_blocks_from_page(page, page_num, os.path.basename(filepath))
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
    merged_blocks = [
        asdict(b)
        for b in merge_continuation_blocks(Block(**blk) for blk in all_blocks)
    ]

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
                        logger.warning(f"Filtering out enhanced block from excluded page {page}")
                        continue
                    filtered_enhanced_blocks.append(block)
                enhanced_blocks = filtered_enhanced_blocks

                if len(enhanced_blocks) != pre_filter_count:
                    logger.info(
                        f"Filtered out {pre_filter_count - len(enhanced_blocks)} enhanced blocks from excluded pages"
                    )

            if enhanced_blocks:
                original_page_set = {
                    p
                    for b in pre_merge_blocks
                    if (p := b.get("source", {}).get("page")) is not None and p not in excluded
                }
                enhanced_pages = {
                    b.get("source", {}).get("page")
                    for b in enhanced_blocks
                    if b.get("source", {}).get("page") is not None
                }
                missing = original_page_set - enhanced_pages
                if missing:
                    logger.warning("PyMuPDF4LLM enhancement dropped pages: %s", sorted(missing))
                    enhancement_stats["degraded"] = len(enhanced_blocks)
                    enhanced_blocks = None

            # If enhancement was successful and high quality, use enhanced blocks
            if enhanced_blocks and enhancement_stats.get("enhanced", 0) > 0:
                if pre_merge_blocks and len(pre_merge_blocks) == len(enhanced_blocks):
                    for eb, ob in zip(enhanced_blocks, pre_merge_blocks):
                        if "bbox" not in eb and ob.get("bbox"):
                            eb["bbox"] = ob["bbox"]
                if enhancement_stats.get("degraded", 0) == 0:
                    logger.info(f"PyMuPDF4LLM enhancement successful: {enhancement_stats}")
                    merged_blocks = [
                        asdict(b)
                        for b in merge_continuation_blocks(
                            Block(**blk) for blk in enhanced_blocks
                        )
                    ]
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
                            logger.debug(f"After traditional cleaning: {repr(block['text'][:50])}")
            else:
                logger.warning(
                    "PyMuPDF4LLM enhancement failed, falling back to traditional text cleaning"
                )
                logger.debug("Ensuring traditional text cleaning pipeline is used for all blocks")
                # Re-apply traditional text cleaning to all blocks
                for block in merged_blocks:
                    if "text" in block:
                        from .text_cleaning import clean_text

                        logger.debug(
                            f"Re-cleaning block text with traditional pipeline: {repr(block['text'][:50])}"
                        )
                        block["text"] = clean_text(block["text"])
                        logger.debug(f"After traditional cleaning: {repr(block['text'][:50])}")
        except Exception as e:
            logger.error(f"PyMuPDF4LLM enhancement failed with error: {e}")
            logger.info("Falling back to traditional text cleaning pipeline")
            logger.debug("Ensuring traditional text cleaning pipeline is used for all blocks")
            # Re-apply traditional text cleaning to all blocks
            for block in merged_blocks:
                if "text" in block:
                    from .text_cleaning import clean_text

                    logger.debug(
                        f"Re-cleaning block text with traditional pipeline: {repr(block['text'][:50])}"
                    )
                    block["text"] = clean_text(block["text"])
                    logger.debug(f"After traditional cleaning: {repr(block['text'][:50])}")
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

    text_blob = "\n".join(block["text"] for block in merged_blocks)
    quality = _assess_text_quality(text_blob)
    logger.debug(f"Text quality assessment: score={quality.get('quality_score', 0):.2f}")
    merged_blocks = _assess_and_maybe_fallback(
        filepath, exclude_pages, merged_blocks, quality["quality_score"]
    )

    # --- Ensure all blocks have a proper source dictionary with filename, page, and location ---
    # If PyMuPDF4LLM enhancement was applied, propagate page numbers from original blocks if missing
    if merged_blocks and isinstance(merged_blocks[0], dict) and "source" in merged_blocks[0]:
        # Try to propagate page numbers from original blocks to enhanced blocks if missing
        # Build a list of original page numbers for each block index
        original_pages: list[Optional[int]] = [
            block.get("source", {}).get("page") for block in merged_blocks
        ]
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


def legacy_extract_text_blocks_from_pdf(
    filepath: str, exclude_pages: Optional[str] = None
) -> list[dict]:
    """Shim retaining the original extraction behavior."""

    return extract_text_blocks_from_pdf(filepath, exclude_pages)


_legacy_extract_text_blocks_from_pdf = extract_text_blocks_from_pdf
