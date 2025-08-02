"""
PyMuPDF4LLM Integration Module

This module provides integration with PyMuPDF4LLM for enhanced PDF extraction
with automatic heading detection and structured Markdown output. It serves as
the primary extraction method in the hybrid approach, with fallback to the
existing three-tier system when needed.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple
import logging
from . import page_artifacts

try:
    import pymupdf4llm

    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
    pymupdf4llm = None

logger = logging.getLogger(__name__)


class PyMuPDF4LLMExtractionError(Exception):
    """Exception raised when PyMuPDF4LLM extraction fails"""

    pass


def is_pymupdf4llm_available() -> bool:
    """Check if PyMuPDF4LLM is available for use"""
    available = PYMUPDF4LLM_AVAILABLE and pymupdf4llm is not None
    logger.debug(f"PyMuPDF4LLM availability check: {available}")
    return available


def extract_with_pymupdf4llm(
    pdf_path: str, exclude_pages: set = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract text using PyMuPDF4LLM with enhanced error handling and validation.
    Respects excluded pages by only processing non-excluded pages.
    """
    logger.info(f"Starting PyMuPDF4LLM extraction for: {pdf_path}")

    try:
        import fitz
        import pymupdf4llm

        # Determine which pages to process
        doc = fitz.open(pdf_path)
        all_pages = set(range(1, len(doc) + 1))
        doc.close()
        if exclude_pages:
            pages_to_include = sorted(list(all_pages - set(exclude_pages)))
        else:
            pages_to_include = sorted(list(all_pages))

        # PyMuPDF4LLM expects 0-based page indices
        if pages_to_include:
            zero_based_pages = [p - 1 for p in pages_to_include]
        else:
            zero_based_pages = []

        # Extract with PyMuPDF4LLM, passing the correct pages
        if zero_based_pages:
            md_text = pymupdf4llm.to_markdown(pdf_path, pages=zero_based_pages)
        else:
            # If all pages are excluded, return empty
            logger.warning("All pages are excluded; returning empty block list.")
            return [], {
                "enhanced": 0,
                "failed": 0,
                "skipped": 0,
                "degraded": 0,
                "artifacts_filtered": 0,
            }

        if not md_text or not md_text.strip():
            logger.warning("PyMuPDF4LLM returned empty text")
            return [], {
                "enhanced": 0,
                "failed": 1,
                "skipped": 0,
                "degraded": 0,
                "artifacts_filtered": 0,
            }

        logger.debug(f"PyMuPDF4LLM extracted {len(md_text)} characters")

        # Convert markdown to blocks
        blocks = _convert_markdown_to_blocks(md_text, pdf_path)

        if not blocks:
            logger.warning("No blocks created from PyMuPDF4LLM output")
            return [], {
                "enhanced": 0,
                "failed": 1,
                "skipped": 0,
                "degraded": 0,
                "artifacts_filtered": 0,
            }

        # Filter blocks to only include those from non-excluded pages (if possible)
        # Since markdown output may not have page info, this is a best-effort;
        # but since we only processed the allowed pages, this should be sufficient.

        # Validate enhancement quality
        quality_score = _validate_enhancement_quality(blocks)
        logger.debug(f"PyMuPDF4LLM enhancement quality score: {quality_score:.2f}")

        if quality_score < 0.5:  # Threshold for acceptable quality
            logger.warning(
                f"PyMuPDF4LLM enhancement quality too low ({quality_score:.2f}), may need fallback"
            )
            return blocks, {
                "enhanced": 0,
                "failed": 0,
                "skipped": 0,
                "degraded": len(blocks),
                "artifacts_filtered": 0,
            }

        # Apply additional cleaning to PyMuPDF4LLM output
        cleaned_blocks = []
        artifacts_filtered = 0

        for block in blocks:
            cleaned_block = _clean_pymupdf4llm_block(block)
            if cleaned_block:
                cleaned_blocks.append(cleaned_block)
            else:
                artifacts_filtered += 1

        stats = {
            "enhanced": len(cleaned_blocks),
            "failed": 0,
            "skipped": 0,
            "degraded": 0,
            "artifacts_filtered": artifacts_filtered,
        }

        logger.info(f"PyMuPDF4LLM extraction completed: {stats}")
        return cleaned_blocks, stats

    except ImportError:
        logger.error(
            "PyMuPDF4LLM not available - install with: pip install pymupdf4llm"
        )
        return [], {
            "enhanced": 0,
            "failed": 1,
            "skipped": 0,
            "degraded": 0,
            "artifacts_filtered": 0,
        }

    except Exception as e:
        logger.error(f"PyMuPDF4LLM extraction failed: {e}")
        return [], {
            "enhanced": 0,
            "failed": 1,
            "skipped": 0,
            "degraded": 0,
            "artifacts_filtered": 0,
        }


def _validate_enhancement_quality(blocks: List[Dict[str, Any]]) -> float:
    """Validate the quality of PyMuPDF4LLM enhancement and return a quality score (0-1)."""
    if not blocks:
        return 0.0

    total_score = 0.0
    total_blocks = len(blocks)

    for block in blocks:
        text = block.get("text", "")
        block_score = 0.0

        if not text.strip():
            continue

        # Score based on text characteristics
        # 1. Length (reasonable blocks get higher scores)
        length_score = min(len(text) / 100, 1.0)  # Normalize to 0-1
        block_score += length_score * 0.3

        # 2. Sentence structure (complete sentences get higher scores)
        sentences = text.count(".") + text.count("!") + text.count("?")
        if len(text) > 0:
            sentence_density = sentences / (len(text) / 100)  # Sentences per 100 chars
            sentence_score = min(sentence_density / 2, 1.0)  # Normalize
            block_score += sentence_score * 0.3

        # 3. Word boundaries (proper spacing gets higher scores)
        import re as _re1

        word_count = len(text.split())
        if word_count > 0:
            # Check for proper word spacing
            glued_words = len(_re1.findall(r"[a-z][A-Z]", text))
            gluing_penalty = min(glued_words / word_count, 0.5)
            spacing_score = 1.0 - gluing_penalty
            block_score += spacing_score * 0.2

        # 4. Quote balance (balanced quotes get higher scores)
        quote_balance = abs(text.count('"') % 2) + abs(text.count("'") % 2)
        quote_score = 1.0 if quote_balance == 0 else 0.5
        block_score += quote_score * 0.2

        total_score += block_score

    average_score = total_score / total_blocks if total_blocks > 0 else 0.0
    return min(average_score, 1.0)


def _clean_pymupdf4llm_block(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Clean and validate a block from PyMuPDF4LLM output.

    The cleaning pipeline strips markdown emphasis, removes underscore
    wrappers, fixes hyphenated line breaks, normalizes quotes, and collapses
    extraneous whitespace before validating the block.
    """
    text = block.get("text", "")

    if not text or not text.strip():
        return None

    import re as _re2
    from .text_cleaning import (
        pipe,
        remove_underscore_emphasis,
        fix_hyphenated_linebreaks,
        normalize_ligatures,
        normalize_quotes,
        remove_control_characters,
        consolidate_whitespace,
    )

    text = pipe(
        text,
        lambda s: _re2.sub(r"^#{1,6}\s*", "", s, flags=_re2.MULTILINE),
        lambda s: _re2.sub(r"\*\*(.*?)\*\*", r"\1", s),
        lambda s: _re2.sub(r"\*(.*?)\*", r"\1", s),
        lambda s: _re2.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s),
        lambda s: _re2.sub(r"\n{3,}", "\n\n", s),
        lambda s: _re2.sub(r" {2,}", " ", s),
        remove_underscore_emphasis,
        fix_hyphenated_linebreaks,
        remove_underscore_emphasis,
        normalize_ligatures,
        normalize_quotes,
        remove_control_characters,
        consolidate_whitespace,
    )

    # Skip blocks that are too short or look like artifacts
    if len(text.strip()) < 10:
        return None

    # Skip blocks that look like page numbers or headers/footers
    if _re2.match(r"^\s*\d+\s*$", text.strip()):  # Just a number
        return None

    if _re2.match(r"^\s*(page|chapter|section)\s+\d+\s*$", text.strip().lower()):
        return None

    # Use shared artifact detection for complex patterns
    if is_page_artifact_text(text, block.get("source", {}).get("page", 0)):
        logger.debug(f"Skipping PyMuPDF4LLM page artifact: {repr(text[:50])}")
        return None

    # Update the block with cleaned text
    cleaned_block = block.copy()
    cleaned_block["text"] = text.strip()

    return cleaned_block


def _convert_markdown_to_blocks(
    markdown_text: str, pdf_path: str
) -> List[Dict[str, Any]]:
    """
    Convert PyMuPDF4LLM Markdown output to structured blocks.

    This function parses the markdown text from PyMuPDF4LLM and converts it
    into structured blocks that match the expected format for the PDF chunking pipeline.

    Args:
        markdown_text: Raw markdown text from PyMuPDF4LLM
        pdf_path: Path to the source PDF file

    Returns:
        List of structured blocks with text, type, and source information
    """
    if not markdown_text or not markdown_text.strip():
        return []

    # Import text cleaning function to collapse newlines
    from .text_cleaning import collapse_single_newlines

    blocks = []
    lines = markdown_text.split("\n")
    current_block_lines = []
    current_block_type = "paragraph"

    for line in lines:
        line = line.strip()

        # Skip empty lines but use them as block separators
        if not line:
            if current_block_lines:
                # Finish current block
                block_text = "\n".join(current_block_lines).strip()
                if block_text:
                    # Apply newline collapsing to clean the text
                    cleaned_text = collapse_single_newlines(block_text)
                    blocks.append(
                        {
                            "type": current_block_type,
                            "text": cleaned_text,
                            "language": "en",  # Default language
                            "source": {
                                "filename": pdf_path,
                                "page": None,  # Page info not available from markdown
                                "location": None,
                            },
                        }
                    )
                current_block_lines = []
                current_block_type = "paragraph"
            continue

        # Check if this line is a heading
        if line.startswith("#"):
            # Finish previous block if it exists
            if current_block_lines:
                block_text = "\n".join(current_block_lines).strip()
                if block_text:
                    # Apply newline collapsing to clean the text
                    cleaned_text = collapse_single_newlines(block_text)
                    blocks.append(
                        {
                            "type": current_block_type,
                            "text": cleaned_text,
                            "language": "en",
                            "source": {
                                "filename": pdf_path,
                                "page": None,
                                "location": None,
                            },
                        }
                    )
                current_block_lines = []

            # Start new heading block
            heading_text = line.lstrip("#").strip()
            if heading_text:
                current_block_lines = [heading_text]
                current_block_type = "heading"
        else:
            # Regular text line
            current_block_lines.append(line)
            if current_block_type != "heading":
                current_block_type = "paragraph"

    # Handle final block
    if current_block_lines:
        block_text = "\n".join(current_block_lines).strip()
        if block_text:
            # Apply newline collapsing to clean the text
            cleaned_text = collapse_single_newlines(block_text)
            blocks.append(
                {
                    "type": current_block_type,
                    "text": cleaned_text,
                    "language": "en",
                    "source": {"filename": pdf_path, "page": None, "location": None},
                }
            )

    return blocks


def _convert_markdown_to_clean_text(markdown_text: str) -> str:
    """
    Convert PyMuPDF4LLM Markdown output to clean text for text cleaning purposes.

    This function strips markdown formatting while preserving text structure
    and paragraph boundaries for use in text cleaning operations.

    Args:
        markdown_text: Raw markdown text from PyMuPDF4LLM

    Returns:
        Clean text with preserved paragraph structure
    """
    from .text_cleaning import clean_text

    # Split into lines for processing
    lines = markdown_text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines but preserve paragraph breaks
        if not line:
            cleaned_lines.append("")
            continue

        # Remove markdown heading markers but keep the text
        if line.startswith("#"):
            heading_text = line.lstrip("#").strip()
            if heading_text:
                cleaned_lines.append(heading_text)
        else:
            # Regular text line - clean it
            cleaned_line = clean_text(line)
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

    # Join lines back together, preserving paragraph structure
    cleaned_text = "\n".join(cleaned_lines)

    # Clean up excessive newlines while preserving paragraph breaks
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def clean_text_with_pymupdf4llm(text: str, pdf_path: Optional[str] = None) -> str:
    """
    Clean text using PyMuPDF4LLM's superior text processing capabilities.

    This function applies PyMuPDF4LLM's text cleaning to improve ligature handling,
    word joining, and whitespace normalization while preserving text structure.

    Args:
        text: Text to clean
        pdf_path: Optional PDF path for context (not used in simplified approach)

    Returns:
        Cleaned text with improved formatting
    """
    logger.debug(f"clean_text_with_pymupdf4llm called with {len(text)} chars")
    logger.debug(f"Input text preview: {repr(text[:100])}")

    if not is_pymupdf4llm_available():
        logger.debug("PyMuPDF4LLM not available, falling back to traditional cleaning")
        # Fallback to traditional text cleaning
        from .text_cleaning import clean_paragraph

        paragraphs = (clean_paragraph(p) for p in text.split("\n\n"))
        cleaned = "\n\n".join(p for p in paragraphs if p)
        logger.debug(f"Traditional fallback result preview: {repr(cleaned[:100])}")
        return cleaned

    try:
        logger.debug("Using PyMuPDF4LLM text cleaning path")

        # Check if we need to apply newline collapsing in PyMuPDF4LLM path
        logger.debug("Checking if text contains single newlines that need collapsing")
        single_newlines_count = text.count("\n") - text.count("\n\n") * 2
        logger.debug(f"Found {single_newlines_count} single newlines in input text")

        # Apply traditional cleaning functions directly to ensure newline handling
        from .text_cleaning import (
            normalize_newlines,
            collapse_single_newlines,
            merge_spurious_paragraph_breaks,
            fix_hyphenated_linebreaks,
            normalize_ligatures,
            consolidate_whitespace,
        )

        # Apply the cleaning steps in the correct order
        from .text_cleaning import (
            collapse_spurious_double_newlines,
            collapse_inline_bullet_artifacts,
            normalize_bullet_lines,
            _apply_steps,
        )

        text = _apply_steps(
            text,
            [
                ("normalize_newlines", normalize_newlines),
                ("normalize_bullet_lines", normalize_bullet_lines),
                ("collapse_single_newlines", collapse_single_newlines),
                ("merge_spurious_paragraph_breaks", merge_spurious_paragraph_breaks),
                ("fix_hyphenated_linebreaks", fix_hyphenated_linebreaks),
                (
                    "collapse_spurious_double_newlines",
                    collapse_spurious_double_newlines,
                ),
                ("collapse_inline_bullet_artifacts", collapse_inline_bullet_artifacts),
            ],
        )

        paragraphs = [
            consolidate_whitespace(normalize_ligatures(p))
            for p in text.split("\n\n")
            if p.strip()
        ]

        cleaned = "\n\n".join(paragraphs)
        logger.debug(f"PyMuPDF4LLM cleaning result preview: {repr(cleaned[:100])}")
        return cleaned

    except Exception as e:
        logger.warning(
            f"PyMuPDF4LLM text cleaning failed: {e}. Falling back to traditional cleaning."
        )
        from .text_cleaning import clean_paragraph

        paragraphs = (clean_paragraph(p) for p in text.split("\n\n"))
        cleaned = "\n\n".join(p for p in paragraphs if p)
        logger.debug(f"Exception fallback result preview: {repr(cleaned[:100])}")
        return cleaned


def should_apply_pymupdf4llm_cleaning(blocks: list[dict]) -> bool:
    """
    Determine if PyMuPDF4LLM cleaning should be applied to improve text quality.

    Args:
        blocks: List of text blocks (dicts) from PDF extraction

    Returns:
        True if PyMuPDF4LLM cleaning should be attempted, False otherwise.
    """
    if not blocks or not isinstance(blocks, list):
        logger.debug(
            "should_apply_pymupdf4llm_cleaning: No blocks provided or not a list."
        )
        return False

    # Heuristic: If any block shows extraction issues, recommend cleaning
    problematic_count = 0
    total_blocks = len(blocks)
    for block in blocks:
        text = block.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        # Look for common extraction issues
        # 1. Ligatures
        if any(lig in text for lig in ["ﬁ", "ﬂ", "ﬀ", "ﬃ", "ﬄ"]):
            problematic_count += 1
            continue
        # 2. Word joining (lowercase followed by uppercase)
        import re

        if re.search(r"[a-z][A-Z]", text):
            problematic_count += 1
            continue
        # 3. Excessive whitespace
        if re.search(r"  +|\n{3,}", text):
            problematic_count += 1
            continue
        # 4. Hyphenation issues
        if re.search(r"-\s*\n\s*[a-z]", text):
            problematic_count += 1
            continue
        # 5. Unbalanced quotes
        if (text.count('"') % 2 != 0) or (text.count("'") % 2 != 0):
            problematic_count += 1
            continue

    # If more than 10% of blocks are problematic, recommend cleaning
    if total_blocks == 0:
        return False
    ratio = problematic_count / total_blocks
    logger.debug(
        f"should_apply_pymupdf4llm_cleaning: {problematic_count}/{total_blocks} blocks problematic (ratio={ratio:.2f})"
    )
    return ratio > 0.1


def _call_pymupdf4llm_api(pdf_path: str, pages: Optional[List[int]] = None) -> str:
    """
    Call PyMuPDF4LLM API with fallback for different method names.

    Args:
        pdf_path: Path to the PDF file
        pages: List of page numbers to process (None for all pages)

    Returns:
        Extracted markdown text

    Raises:
        PyMuPDF4LLMExtractionError: If no working API method is found
    """
    api_methods = [
        (
            "to_markdown",
            lambda: pymupdf4llm.to_markdown(pdf_path, pages=pages),
        ),
        ("extract", lambda: pymupdf4llm.extract(pdf_path, pages=pages)),
        ("convert", lambda: pymupdf4llm.convert(pdf_path, pages=pages)),
        ("parse", lambda: pymupdf4llm.parse(pdf_path, pages=pages)),
        (
            "to_markdown_simple",
            lambda: pymupdf4llm.to_markdown(pdf_path),
        ),
        ("extract_simple", lambda: pymupdf4llm.extract(pdf_path)),
        ("convert_simple", lambda: pymupdf4llm.convert(pdf_path)),
        ("parse_simple", lambda: pymupdf4llm.parse(pdf_path)),
    ]
    class_methods = ["LlamaParseReader", "PyMuPDFReader", "PDFReader", "DocumentReader"]

    for method_name, method_call in api_methods:
        try:
            if hasattr(pymupdf4llm, method_name.replace("_simple", "")):
                logger.debug(f"Trying PyMuPDF4LLM method: {method_name}")
                result = method_call()
                if isinstance(result, str) and result.strip():
                    logger.debug(f"Successfully extracted using {method_name}")
                    return result
        except Exception as e:
            logger.debug(f"Method {method_name} failed: {e}")
            continue

    for class_name in class_methods:
        if hasattr(pymupdf4llm, class_name):
            try:
                logger.debug(f"Trying PyMuPDF4LLM class: {class_name}")
                cls = getattr(pymupdf4llm, class_name)
                reader = cls()
                for method_name in [
                    "load_data",
                    "read",
                    "extract",
                    "parse",
                    "to_markdown",
                ]:
                    if hasattr(reader, method_name):
                        try:
                            method = getattr(reader, method_name)
                            result = method(pdf_path)
                            if isinstance(result, str) and result.strip():
                                logger.debug(
                                    f"Successfully extracted using {class_name}.{method_name}"
                                )
                                return result
                            elif isinstance(result, list) and result:
                                if hasattr(result[0], "text"):
                                    text = "\n".join([doc.text for doc in result])
                                else:
                                    text = "\n".join([str(doc) for doc in result])
                                if text.strip():
                                    logger.debug(
                                        f"Successfully extracted using {class_name}.{method_name}"
                                    )
                                    return text
                        except Exception as e:
                            logger.debug(f"{class_name}.{method_name} failed: {e}")
                            continue
            except Exception as e:
                logger.debug(f"Class {class_name} instantiation failed: {e}")
                continue

    available_attrs = [attr for attr in dir(pymupdf4llm) if not attr.startswith("_")]
    raise PyMuPDF4LLMExtractionError(
        f"Could not find working PyMuPDF4LLM extraction method. "
        f"Available attributes: {available_attrs}"
    )


def assess_text_cleaning_quality(
    original_text: str, cleaned_text: str
) -> Dict[str, Any]:
    """
    Assess the quality of PyMuPDF4LLM text cleaning for simple quality validation.

    Args:
        original_text: Original text before cleaning
        cleaned_text: Text after PyMuPDF4LLM cleaning

    Returns:
        Simple quality assessment metrics
    """
    if not cleaned_text or not cleaned_text.strip():
        return {
            "quality_score": 0.0,
            "has_content": False,
            "text_length": 0,
            "cleaning_effective": False,
            "issues": ["No cleaned text produced"],
        }

    issues = []
    quality_factors = []

    original_length = len(original_text.strip())
    cleaned_length = len(cleaned_text.strip())

    if cleaned_length > 0:
        if original_length > 0:
            length_ratio = cleaned_length / original_length
            if 0.8 <= length_ratio <= 1.2:
                quality_factors.append(0.5)
            elif 0.6 <= length_ratio <= 1.5:
                quality_factors.append(0.3)
            else:
                quality_factors.append(0.1)
                issues.append(f"Significant length change: {length_ratio:.2f}")
        else:
            quality_factors.append(0.5)
    else:
        issues.append("No content after cleaning")

    if cleaned_text != original_text:
        quality_factors.append(0.3)
    else:
        quality_factors.append(0.1)
        issues.append("No text cleaning applied")

    import re as _re3

    excessive_spaces = len(_re3.findall(r" {3,}", cleaned_text))
    excessive_newlines = len(_re3.findall(r"\n{3,}", cleaned_text))

    if excessive_spaces == 0 and excessive_newlines == 0:
        quality_factors.append(0.2)
    elif excessive_spaces < 5 and excessive_newlines < 5:
        quality_factors.append(0.1)
    else:
        issues.append("Excessive whitespace not cleaned")

    quality_score = min(sum(quality_factors), 1.0)

    return {
        "quality_score": quality_score,
        "has_content": len(cleaned_text.strip()) > 0,
        "text_length": len(cleaned_text),
        "cleaning_effective": cleaned_text != original_text,
        "length_ratio": cleaned_length / original_length if original_length > 0 else 0,
        "issues": issues,
    }


def detect_text_flow_degradation(
    original_text: str, cleaned_text: str
) -> Dict[str, Any]:
    """
    Detect if PyMuPDF4LLM cleaning has degraded text flow quality.

    This function specifically looks for issues that can occur when cleaning
    text that spans page boundaries or contains page artifacts.

    Args:
        original_text: Original text before cleaning
        cleaned_text: Text after PyMuPDF4LLM cleaning

    Returns:
        Dictionary with degradation assessment and specific issues found
    """
    import re as _re4

    issues = []
    degradation_score = 0.0

    if not cleaned_text or not cleaned_text.strip():
        return {
            "degraded": True,
            "degradation_score": 1.0,
            "issues": ["Text completely removed by cleaning"],
            "recommendation": "Use original text",
        }

    original_sentences = _re4.split(r"[.!?]+", original_text)
    cleaned_sentences = _re4.split(r"[.!?]+", cleaned_text)

    short_fragments = [
        s.strip() for s in cleaned_sentences if 0 < len(s.strip().split()) <= 4
    ]
    if len(short_fragments) > len(original_sentences) * 0.3:
        issues.append(
            f"Excessive sentence fragmentation: {len(short_fragments)} short fragments"
        )
        degradation_score += 0.3

    contamination_patterns = [
        r"\b\d+\s*$",
        r"^\s*\d+\b",
        r"\bchapter\s+\d+\b.*\bpage\s+\d+\b",
        r"\bfootnote\s*\d+\b",
        r"^\s*[A-Z\s]{10,}\s*$",
    ]

    contamination_count = 0
    for pattern in contamination_patterns:
        if _re4.search(pattern, cleaned_text, re.IGNORECASE):
            contamination_count += 1

    if contamination_count > 0:
        issues.append(
            f"Potential header/footer contamination: {contamination_count} patterns detected"
        )
        degradation_score += contamination_count * 0.15

    original_length = len(original_text.strip())
    cleaned_length = len(cleaned_text.strip())

    if original_length > 0:
        length_ratio = cleaned_length / original_length
        if length_ratio < 0.7:
            issues.append(f"Significant content loss: {length_ratio:.2f} length ratio")
            degradation_score += 0.4
        elif length_ratio > 1.5:
            issues.append(
                f"Unexpected content expansion: {length_ratio:.2f} length ratio"
            )
            degradation_score += 0.2

    original_words = len(original_text.split())
    cleaned_words = len(cleaned_text.split())

    if original_words > 0:
        word_ratio = cleaned_words / original_words
        if word_ratio < 0.8:
            issues.append(f"Potential word merging: {word_ratio:.2f} word count ratio")
            degradation_score += 0.2

    excessive_spaces = len(_re4.findall(r" {4,}", cleaned_text))
    excessive_newlines = len(_re4.findall(r"\n{4,}", cleaned_text))

    if excessive_spaces > 5 or excessive_newlines > 3:
        issues.append(
            f"Formatting issues: {excessive_spaces} space clusters, {excessive_newlines} newline clusters"
        )
        degradation_score += 0.1

    degradation_score = min(degradation_score, 1.0)
    degraded = degradation_score > 0.3

    recommendation = "Use original text" if degraded else "Use cleaned text"
    if degradation_score > 0.1 and not degraded:
        recommendation = "Use cleaned text with caution"

    return {
        "degraded": degraded,
        "degradation_score": degradation_score,
        "issues": issues,
        "recommendation": recommendation,
        "original_length": original_length,
        "cleaned_length": cleaned_length,
        "length_ratio": cleaned_length / original_length if original_length > 0 else 0,
    }


def should_apply_pymupdf4llm_cleaning(blocks: List[Dict[str, Any]]) -> bool:
    """Determine if PyMuPDF4LLM cleaning should be applied to improve text quality."""
    if not blocks:
        logger.debug("No blocks provided, skipping PyMuPDF4LLM cleaning")
        return False

    total_text = ""
    total_blocks = len(blocks)
    problematic_patterns = 0

    for block in blocks:
        text = block.get("text", "")
        total_text += text + " "
        if _has_extraction_issues(text):
            problematic_patterns += 1

    logger.debug(
        f"PyMuPDF4LLM analysis: {total_blocks} blocks, {problematic_patterns} with extraction issues"
    )

    should_apply = (
        problematic_patterns > 0
        or (total_blocks > 10 and len(total_text) / total_blocks < 100)
        or '"' in total_text
        or "'" in total_text
        or any(
            indicator in total_text.lower()
            for indicator in [
                "part i",
                "part ii",
                "part iii",
                "chapter",
                "section",
                "figure",
                "table",
                "appendix",
                "bibliography",
            ]
        )
        or _has_page_boundary_issues(blocks)
    )

    if should_apply:
        logger.info(
            f"PyMuPDF4LLM enhancement recommended: {problematic_patterns} problematic patterns detected"
        )
    else:
        logger.debug("PyMuPDF4LLM enhancement not needed based on text analysis")

    return should_apply


def _has_extraction_issues(text: str) -> bool:
    """Check if text shows signs of extraction issues that PyMuPDF4LLM could fix."""
    if not text:
        return False

    import re as _re5

    if _re5.search(r"[a-z][A-Z]", text):
        return True
    if (
        text.strip()
        and text.strip()[0].islower()
        and not text.strip().endswith((".", "!", "?", ":", ";"))
    ):
        return True
    if _re5.search(r"\s{3,}", text):
        return True
    quote_count = text.count('"') + text.count("'")
    if quote_count > 0 and quote_count % 2 != 0:
        return True
    if len(text.strip()) < 20 and not text.strip().endswith((".", "!", "?")):
        return True
    if _re5.search(r"[a-z][.!?][A-Z]", text):
        return True

    return False


def _has_page_boundary_issues(blocks: List[Dict[str, Any]]) -> bool:
    """Check if blocks show signs of page boundary extraction issues."""
    if len(blocks) < 2:
        return False

    page_transitions = 0
    problematic_transitions = 0

    for i in range(len(blocks) - 1):
        curr_block = blocks[i]
        next_block = blocks[i + 1]
        curr_page = curr_block.get("source", {}).get("page")
        next_page = next_block.get("source", {}).get("page")

        if curr_page != next_page:
            page_transitions += 1
            curr_text = curr_block.get("text", "").strip()
            next_text = next_block.get("text", "").strip()

            if (
                curr_text
                and next_text
                and not curr_text.endswith((".", "!", "?"))
                and next_text[0].islower()
            ):
                problematic_transitions += 1

    if page_transitions > 0:
        problem_ratio = problematic_transitions / page_transitions
        logger.debug(
            f"Page boundary analysis: {problematic_transitions}/{page_transitions} transitions problematic ({problem_ratio:.2f})"
        )
        return problem_ratio > 0.3

    return False


def is_page_artifact_text(text: str, page_num: Optional[int]) -> bool:
    """Delegate to shared page artifact detection logic."""
    return page_artifacts.is_page_artifact_text(text, page_num)


def is_text_already_clean(text: str) -> bool:
    """
    Check if text is already clean and doesn't need PyMuPDF4LLM processing.

    Args:
        text: Text to check

    Returns:
        True if text appears to already be clean
    """
    import re as _re7

    has_ligatures = any(char in text for char in ["ﬁ", "ﬂ", "ﬀ", "ﬃ", "ﬄ"])
    has_excessive_spaces = bool(_re7.search(r" {3,}", text))
    has_hyphenated_breaks = bool(_re7.search(r"-\s*\n\s*[a-z]", text))
    has_word_joining_issues = bool(_re7.search(r"[a-z][A-Z]", text))

    return not (
        has_ligatures
        or has_excessive_spaces
        or has_hyphenated_breaks
        or has_word_joining_issues
    )


def has_cleaning_opportunities(text: str) -> bool:
    """
    Check if text has characteristics that would benefit from PyMuPDF4LLM cleaning.

    Args:
        text: Text to check

    Returns:
        True if text would likely benefit from cleaning
    """
    import re as _re8

    has_ligatures = any(char in text for char in ["ﬁ", "ﬂ", "ﬀ", "ﬃ", "ﬄ"])
    has_excessive_spaces = bool(_re8.search(r" {3,}", text))
    has_hyphenated_breaks = bool(_re8.search(r"-\s*\n\s*[a-z]", text))
    has_unicode_issues = bool(_re8.search(r"[^\x00-\x7F]", text)) and not has_ligatures

    return (
        has_ligatures
        or has_excessive_spaces
        or has_hyphenated_breaks
        or has_unicode_issues
    )


def get_pymupdf4llm_info() -> Dict[str, Any]:
    """
    Get information about the PyMuPDF4LLM installation.

    Returns:
        Dictionary with installation and capability information
    """
    if not is_pymupdf4llm_available():
        return {
            "available": False,
            "error": "PyMuPDF4LLM not installed or not importable",
        }

    try:
        info = {
            "available": True,
            "version": getattr(pymupdf4llm, "__version__", "Unknown"),
            "module_file": getattr(pymupdf4llm, "__file__", "Unknown"),
            "available_attributes": [
                attr for attr in dir(pymupdf4llm) if not attr.startswith("_")
            ],
        }

        try:
            working_methods = []
            test_methods = ["to_markdown", "extract", "convert", "parse"]

            for method in test_methods:
                if hasattr(pymupdf4llm, method):
                    working_methods.append(method)
            info["working_methods"] = working_methods
            info["functional"] = len(working_methods) > 0

        except Exception as e:
            info["functional"] = False
            info["test_error"] = str(e)

        return info

    except Exception as e:
        return {"available": True, "error": f"Error getting PyMuPDF4LLM info: {str(e)}"}


def apply_pymupdf4llm_fallback(
    original_blocks: List[Dict[str, Any]], pdf_path: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply PyMuPDF4LLM selectively to problematic blocks as a fallback strategy."""
    logger.info("Applying PyMuPDF4LLM fallback strategy")

    if not original_blocks:
        return original_blocks, {
            "enhanced": 0,
            "failed": 0,
            "skipped": 0,
            "degraded": 0,
            "artifacts_filtered": 0,
        }

    problematic_blocks = []
    for i, block in enumerate(original_blocks):
        if _has_extraction_issues(block.get("text", "")):
            problematic_blocks.append(i)

    if not problematic_blocks:
        logger.debug("No problematic blocks identified for fallback enhancement")
        return original_blocks, {
            "enhanced": 0,
            "failed": 0,
            "skipped": len(original_blocks),
            "degraded": 0,
            "artifacts_filtered": 0,
        }
    logger.info(
        f"Applying fallback enhancement to {len(problematic_blocks)} problematic blocks"
    )

    try:
        pymupdf4llm_blocks, _ = extract_with_pymupdf4llm(pdf_path)
        if not pymupdf4llm_blocks:
            logger.warning("PyMuPDF4LLM fallback failed - no blocks extracted")
            return original_blocks, {
                "enhanced": 0,
                "failed": len(problematic_blocks),
                "skipped": len(original_blocks) - len(problematic_blocks),
                "degraded": 0,
                "artifacts_filtered": 0,
            }
        enhanced_blocks = _merge_extraction_results(
            original_blocks, pymupdf4llm_blocks, problematic_blocks
        )
        enhanced_count = len(
            [idx for idx in problematic_blocks if idx < len(enhanced_blocks)]
        )
        stats = {
            "enhanced": enhanced_count,
            "failed": 0,
            "skipped": len(original_blocks) - enhanced_count,
            "degraded": 0,
            "artifacts_filtered": 0,
        }
        logger.info(f"PyMuPDF4LLM fallback completed: {stats}")
        return enhanced_blocks, stats
    except Exception as e:
        logger.error(f"PyMuPDF4LLM fallback failed: {e}")
        return original_blocks, {
            "enhanced": 0,
            "failed": len(problematic_blocks),
            "skipped": len(original_blocks) - len(problematic_blocks),
            "degraded": 0,
            "artifacts_filtered": 0,
        }


def _merge_extraction_results(
    original_blocks: List[Dict[str, Any]],
    pymupdf4llm_blocks: List[Dict[str, Any]],
    problematic_indices: List[int],
) -> List[Dict[str, Any]]:
    """Merge original and PyMuPDF4LLM extraction results, using the better version for each block."""
    merged_blocks = original_blocks.copy()
    for prob_idx in problematic_indices:
        if prob_idx >= len(original_blocks):
            continue
        original_text = original_blocks[prob_idx].get("text", "")
        best_match = _find_best_matching_block(original_text, pymupdf4llm_blocks)
        if best_match and _is_text_improvement(
            original_text, best_match.get("text", "")
        ):
            logger.debug(f"Replacing block {prob_idx} with PyMuPDF4LLM version")
            improved_block = original_blocks[prob_idx].copy()
            improved_block["text"] = best_match["text"]
            merged_blocks[prob_idx] = improved_block
    return merged_blocks


def _find_best_matching_block(
    target_text: str, candidate_blocks: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Find the block that best matches the target text."""
    if not target_text or not candidate_blocks:
        return None
    target_words = set(target_text.lower().split())
    best_match = None
    best_score = 0.0
    for block in candidate_blocks:
        candidate_text = block.get("text", "")
        if not candidate_text:
            continue
        candidate_words = set(candidate_text.lower().split())
        intersection = target_words & candidate_words
        union = target_words | candidate_words
        similarity = len(intersection) / len(union) if union else 0.0
        if similarity > best_score and similarity > 0.3:
            best_score = similarity
            best_match = block
    return best_match


def _is_text_improvement(original: str, candidate: str) -> bool:
    """Determine if the candidate text is an improvement over the original."""
    if not original or not candidate:
        return False
    improvements = 0
    import re

    original_glued = len(re.findall(r"[a-z][A-Z]", original))
    candidate_glued = len(re.findall(r"[a-z][A-Z]", candidate))
    if candidate_glued < original_glued:
        improvements += 1
    original_sentences = original.count(".") + original.count("!") + original.count("?")
    candidate_sentences = (
        candidate.count(".") + candidate.count("!") + candidate.count("?")
    )
    if candidate_sentences > original_sentences:
        improvements += 1
    original_quote_balance = abs(original.count('"') % 2)
    candidate_quote_balance = abs(candidate.count('"') % 2)
    if candidate_quote_balance < original_quote_balance:
        improvements += 1
    if (
        len(candidate) > len(original) * 1.1
        and candidate.strip().endswith((".", "!", "?"))
        and not original.strip().endswith((".", "!", "?"))
    ):
        improvements += 1
    return improvements >= 2
