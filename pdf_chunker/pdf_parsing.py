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
from .pymupdf4llm_integration import (
    extract_with_pymupdf4llm,
    is_pymupdf4llm_available,
    PyMuPDF4LLMExtractionError
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
    if not text:
        return True

    # Check for page numbers (standalone numbers)
    if text.isdigit() and len(text) <= 3:
        return True

    # Check for common header/footer patterns
    header_footer_patterns = [
        r'^\d+$',  # Just page numbers
        r'^page\s+\d+',  # "Page 1", "Page 2", etc.
        r'^\d+\s*$',  # Page numbers with whitespace
        r'^chapter\s+\d+$',  # Standalone "Chapter X"
        r'^\d+\s+chapter',  # "1 Chapter", "2 Chapter", etc.
    ]

    text_lower = text.lower()
    for pattern in header_footer_patterns:
        if re.match(pattern, text_lower):
            return True

    # Check for very short text that might be artifacts
    if len(text.split()) <= 2 and len(text) <= 20:
        # Could be header/footer, but be conservative
        return False

    return False


def merge_continuation_blocks(blocks: list[dict]) -> list[dict]:
    """
    Merge hyphenated or split words across consecutive blocks with improved page boundary handling.
    
    This function now handles:
    - Hyphenated words split across pages
    - Sentences that span page boundaries
    - Filtering of page artifacts before merging
    - Better text flow reconstruction
    """
    if not blocks:
        return blocks

    # First pass: filter out obvious page artifacts
    filtered_blocks = []
    for block in blocks:
        page_num = block.get("source", {}).get("page", 0)
        if not is_page_artifact(block, page_num):
            filtered_blocks.append(block)

    if not filtered_blocks:
        return blocks  # Return original if all filtered out

    # Second pass: merge continuation blocks with improved logic
    merged = []
    skip_next = False

    for i in range(len(filtered_blocks) - 1):
        if skip_next:
            skip_next = False
            continue

        curr_block = filtered_blocks[i]
        next_block = filtered_blocks[i + 1]

        curr_text = curr_block["text"].strip()
        next_text = next_block["text"].strip()

        # Skip empty blocks
        if not curr_text:
            continue

        if not next_text:
            merged.append(curr_block)
            continue

        # Check for hyphenated word continuation
        should_merge = False
        merged_text = curr_text

        # Case 1: Hyphenated word at end of current block
        if curr_text.endswith("-") and next_text and next_text[0].islower():
            # Remove hyphen and merge
            merged_text = curr_text[:-1] + next_text
            should_merge = True

        # Case 2: Word appears to be split without hyphen (less common)
        elif (curr_text and next_text and 
              not curr_text.endswith(('.', '!', '?', ':', ';')) and
              next_text[0].islower() and
              len(curr_text.split()[-1]) < 4 and  # Last word is very short
              len(next_text.split()[0]) < 8):     # First word is reasonable length
            # Merge without adding space (word was split)
            last_word = curr_text.split()[-1]
            first_word = next_text.split()[0]
            rest_of_curr = ' '.join(curr_text.split()[:-1])
            rest_of_next = ' '.join(next_text.split()[1:])

            if rest_of_curr:
                merged_text = rest_of_curr + ' ' + last_word + first_word
            else:
                merged_text = last_word + first_word

            if rest_of_next:
                merged_text += ' ' + rest_of_next
            else:
                merged_text = merged_text

            should_merge = True

        # Case 3: Sentence continuation across pages (no punctuation at end)
        elif (curr_text and next_text and 
              not curr_text.endswith(('.', '!', '?')) and
              not next_text[0].isupper() and
              curr_block.get("source", {}).get("page") != next_block.get("source", {}).get("page")):
            # Merge with space for sentence continuation
            merged_text = curr_text + ' ' + next_text
            should_merge = True

        if should_merge:
            # Create merged block preserving metadata from first block
            merged_block = curr_block.copy()
            merged_block["text"] = merged_text

            # Update source information to reflect span
            curr_page = curr_block.get("source", {}).get("page")
            next_page = next_block.get("source", {}).get("page")
            if curr_page != next_page:
                merged_block["source"]["page_span"] = f"{curr_page}-{next_page}"

            merged.append(merged_block)
            skip_next = True
        else:
            merged.append(curr_block)

    # Add the last block if it wasn't merged
    if not skip_next and filtered_blocks:
        merged.append(filtered_blocks[-1])

    return merged


def extract_text_blocks_from_pdf(filepath: str, exclude_pages: str = None) -> list[dict]:
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

    # Always use traditional extraction for structural analysis
    doc = fitz.open(filepath)
    excluded = set()

    if exclude_pages:
        try:
            excluded = validate_page_exclusions(
                parse_page_ranges(exclude_pages), len(doc), os.path.basename(filepath)
            )
            logger.debug(f"Excluding pages: {sorted(excluded)}")
        except ValueError as e:
            print(f"Error parsing page exclusions: {e}", file=sys.stderr)

    all_blocks = []
    page_block_counts = {}

    for page_num, page in enumerate(doc, start=1):
        if page_num in excluded:
            logger.debug(f"Skipping excluded page {page_num}")
            continue

        page_blocks = extract_blocks_from_page(page, page_num, os.path.basename(filepath))
        page_block_counts[page_num] = len(page_blocks)
        all_blocks.extend(page_blocks)

        logger.debug(f"Page {page_num}: extracted {len(page_blocks)} blocks")

    doc.close()

    logger.debug(f"Total blocks before merging: {len(all_blocks)}")
    logger.debug(f"Page block distribution: {page_block_counts}")

    # Apply improved continuation merging with page boundary handling
    merged_blocks = merge_continuation_blocks(all_blocks)

    logger.debug(f"Total blocks after merging: {len(merged_blocks)}")

    # Log text flow analysis for debugging
    for i, block in enumerate(merged_blocks[:5]):  # Log first 5 blocks for debugging
        text_preview = block.get("text", "")[:100].replace('\n', ' ')
        page_info = block.get("source", {})
        logger.debug(f"Block {i}: page {page_info.get('page', 'unknown')}, "
                     f"type {block.get('type', 'unknown')}, "
                     f"text: '{text_preview}{'...' if len(block.get('text', '')) > 100 else ''}'")

    # Optionally enhance text quality with PyMuPDF4LLM text cleaning
    enhanced_blocks = _enhance_blocks_with_pymupdf4llm_cleaning(merged_blocks, filepath)

    logger.debug(f"Total blocks after PyMuPDF4LLM enhancement: {len(enhanced_blocks)}")

    # Assess text quality and apply fallbacks if needed
    text_blob = "\n".join(block["text"] for block in enhanced_blocks)
    quality = _assess_text_quality(text_blob)

    logger.debug(f"Text quality assessment: score={quality.get('quality_score', 0):.2f}")

    if quality["quality_score"] < 0.7:
        logger.warning(f"Low quality score ({quality['quality_score']:.2f}), attempting fallback extraction")
        fallback = _extract_with_pdftotext(filepath, exclude_pages)
        if fallback:
            logger.info("Using pdftotext fallback extraction")
            return merge_continuation_blocks(fallback)
        if PDFMINER_AVAILABLE:
            fallback = _extract_with_pdfminer(filepath, exclude_pages)
            if fallback:
                logger.info("Using pdfminer fallback extraction")
                return merge_continuation_blocks(fallback)

    return enhanced_blocks


def _enhance_blocks_with_pymupdf4llm_cleaning(blocks: list[dict], filepath: str) -> list[dict]:
    """
    Enhance text blocks with PyMuPDF4LLM text cleaning while preserving all structural metadata.
    
    Enhanced to apply cleaning after page flow reconstruction and with better artifact filtering.
    This function now uses selective cleaning based on block characteristics and includes
    quality checks to detect when PyMuPDF4LLM cleaning degrades text flow.
    
    Args:
        blocks: List of text blocks from traditional extraction (after page boundary processing)
        filepath: Path to PDF file for context
        
    Returns:
        Enhanced blocks with improved text quality but preserved structure
    """
    import os
    import logging
    
    logger = logging.getLogger(__name__)

    # Check environment variable to control PyMuPDF4LLM usage
    env_use_pymupdf4llm = os.getenv('PDF_CHUNKER_USE_PYMUPDF4LLM', '').lower()
    if env_use_pymupdf4llm in ('false', '0', 'no', 'off'):
        logger.debug("PyMuPDF4LLM enhancement disabled by environment variable")
        return blocks
    
    from .pymupdf4llm_integration import (
        is_pymupdf4llm_available, 
        clean_text_with_pymupdf4llm,
        should_apply_pymupdf4llm_cleaning,
        detect_text_flow_degradation
    )

    # If PyMuPDF4LLM is not available, return blocks unchanged
    if not is_pymupdf4llm_available():
        logger.debug("PyMuPDF4LLM not available, skipping text enhancement")
        return blocks

    enhanced_blocks = []
    enhancement_stats = {
        "enhanced": 0, 
        "failed": 0, 
        "skipped": 0, 
        "degraded": 0,
        "artifacts_filtered": 0
    }

    # Build document context for better decision making
    document_context = {
        "total_blocks": len(blocks),
        "pages": set(block.get("source", {}).get("page", 0) for block in blocks),
        "filepath": filepath
    }

    for i, block in enumerate(blocks):
        enhanced_block = block.copy()
        original_text = block.get('text', '')

        # Skip very short blocks that are likely artifacts
        if len(original_text.strip()) < 10:
            enhancement_stats["skipped"] += 1
            enhanced_blocks.append(enhanced_block)
            continue

        # Enhanced artifact detection using page context
        page_num = block.get("source", {}).get("page", 0)
        if is_page_artifact(block, page_num):
            logger.debug(f"Filtering artifact block on page {page_num}: '{original_text[:50]}...'")
            enhancement_stats["artifacts_filtered"] += 1
            enhanced_blocks.append(enhanced_block)
            continue

        # Selective cleaning based on block characteristics
        if not should_apply_pymupdf4llm_cleaning(block, document_context):
            logger.debug(f"Block {i}: skipping PyMuPDF4LLM cleaning (not beneficial)")
            enhancement_stats["skipped"] += 1
            enhanced_blocks.append(enhanced_block)
            continue

        if original_text.strip():
            try:
                # Apply PyMuPDF4LLM text cleaning
                cleaned_text = clean_text_with_pymupdf4llm(original_text, filepath)
                
                # Quality check: detect text flow degradation
                degradation_assessment = detect_text_flow_degradation(original_text, cleaned_text)
                
                if degradation_assessment['degraded']:
                    # Cleaning degraded text quality, keep original
                    logger.debug(f"Block {i}: PyMuPDF4LLM cleaning degraded text quality: {degradation_assessment['issues']}")
                    enhanced_block['text'] = original_text
                    enhancement_stats["degraded"] += 1
                    
                    if 'metadata' not in enhanced_block:
                        enhanced_block['metadata'] = {}
                    enhanced_block['metadata']['text_enhanced_with_pymupdf4llm'] = False
                    enhanced_block['metadata']['pymupdf4llm_degradation_detected'] = True
                    enhanced_block['metadata']['degradation_issues'] = degradation_assessment['issues']
                    
                elif cleaned_text and len(cleaned_text.strip()) > len(original_text.strip()) * 0.5:
                    # Cleaning was successful
                    enhanced_block['text'] = cleaned_text
                    enhancement_stats["enhanced"] += 1
                    
                    # Add metadata about text enhancement
                    if 'metadata' not in enhanced_block:
                        enhanced_block['metadata'] = {}
                    enhanced_block['metadata']['text_enhanced_with_pymupdf4llm'] = True
                    enhanced_block['metadata']['degradation_score'] = degradation_assessment['degradation_score']
                    
                    logger.debug(f"Block {i}: enhanced text from {len(original_text)} to {len(cleaned_text)} chars "
                                 f"(degradation score: {degradation_assessment['degradation_score']:.2f})")
                else:
                    # Cleaning produced poor results, keep original
                    logger.debug(f"Block {i}: PyMuPDF4LLM cleaning produced poor results, keeping original")
                    enhanced_block['text'] = original_text
                    enhancement_stats["failed"] += 1
                    
                    if 'metadata' not in enhanced_block:
                        enhanced_block['metadata'] = {}
                    enhanced_block['metadata']['text_enhanced_with_pymupdf4llm'] = False

            except Exception as e:
                # If PyMuPDF4LLM cleaning fails, keep original text
                logger.debug(f"Block {i}: PyMuPDF4LLM text cleaning failed: {e}")
                enhanced_block['text'] = original_text
                enhancement_stats["failed"] += 1

                if 'metadata' not in enhanced_block:
                    enhanced_block['metadata'] = {}
                enhanced_block['metadata']['text_enhanced_with_pymupdf4llm'] = False
                enhanced_block['metadata']['pymupdf4llm_error'] = str(e)

        enhanced_blocks.append(enhanced_block)

    logger.info(f"PyMuPDF4LLM enhancement completed: {enhancement_stats}")
    
    # Log summary of enhancement effectiveness
    total_processed = enhancement_stats["enhanced"] + enhancement_stats["failed"] + enhancement_stats["degraded"]
    if total_processed > 0:
        success_rate = enhancement_stats["enhanced"] / total_processed
        logger.info(f"PyMuPDF4LLM success rate: {success_rate:.2f} ({enhancement_stats['enhanced']}/{total_processed})")
    
    return enhanced_blocks
