import os
import re
import sys
import subprocess
from langdetect import detect, LangDetectException

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams

    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

from .text_cleaning import clean_text
from .page_artifacts import remove_page_artifact_lines
from .heading_detection import _detect_heading_fallback


def _detect_language(text: str) -> str:
    """Detects language of a text block, defaults to 'un' (unknown) on failure."""
    try:
        return detect(text)
    except LangDetectException:
        return "un"


def _assess_text_quality(text: str) -> dict:
    """
    Assess the quality of extracted text to determine if fallback methods are needed.
    Returns a dict with quality metrics.
    """
    if not text or not text.strip():
        return {"avg_line_length": 0, "space_density": 0, "quality_score": 0}

    lines = text.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]

    if not non_empty_lines:
        return {"avg_line_length": 0, "space_density": 0, "quality_score": 0}

    # Calculate average line length
    avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)

    # Calculate space density (spaces per character)
    total_chars = sum(len(line) for line in non_empty_lines)
    total_spaces = sum(line.count(" ") for line in non_empty_lines)
    space_density = total_spaces / total_chars if total_chars > 0 else 0

    # Quality score: penalize very long lines and very low space density
    quality_score = 1.0
    if avg_line_length > 1000:  # Very long lines indicate poor extraction
        quality_score *= 0.3
    if space_density < 0.05:  # Very low space density indicates missing spaces
        quality_score *= 0.2

    return {
        "avg_line_length": avg_line_length,
        "space_density": space_density,
        "quality_score": quality_score,
    }


def _clean_fallback_text(text: str) -> str:
    """Remove page artifacts across pages in fallback extraction."""
    pages = text.split("\f")
    cleaned_pages = [
        remove_page_artifact_lines(page, i + 1) for i, page in enumerate(pages)
    ]
    return "\f".join(cleaned_pages)


def _extract_with_pdftotext(filepath: str, exclude_pages: str = None) -> list[dict]:
    """
    Fallback extraction using pdftotext with layout preservation.

    Args:
        filepath: Path to the PDF file
        exclude_pages: Page ranges to exclude (e.g., "1,3,5-10,15-20")
    """
    try:
        # Parse page exclusions if provided
        excluded_pages = set()
        if exclude_pages:
            from .page_utils import parse_page_ranges

            try:
                excluded_pages = parse_page_ranges(exclude_pages)
            except ValueError as e:
                print(
                    f"Error parsing page exclusions in pdftotext fallback: {e}",
                    file=sys.stderr,
                )

        # Build pdftotext command with page exclusions if needed
        cmd = ["pdftotext", "-layout"]

        # pdftotext doesn't have built-in page exclusion, so we'll extract all pages
        # and filter the results afterward
        cmd.extend([filepath, "-"])

        # Try pdftotext with -layout flag
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(
                f"pdftotext failed with return code {result.returncode}",
                file=sys.stderr,
            )
            return []

        raw_text = result.stdout
        quality = _assess_text_quality(raw_text)
        print(
            f"pdftotext extraction quality: {quality['quality_score']:.2f}",
            file=sys.stderr,
        )

        if quality["quality_score"] < 0.7:
            return []

        raw_text = _clean_fallback_text(raw_text)

        # Parse the text into structured blocks
        structured_blocks = []
        paragraphs = raw_text.split("\n\n")

        for paragraph in paragraphs:
            block_text = clean_text(paragraph)
            if block_text:
                # Simple heuristic: short paragraphs with title case might be headings
                is_heading = (
                    len(block_text.split()) < 15
                    and block_text.istitle()
                    and not block_text.endswith(".")
                )

                block_type = "heading" if is_heading else "paragraph"
                lang = _detect_language(block_text)
                structured_blocks.append(
                    {
                        "type": block_type,
                        "text": block_text,
                        "language": lang,
                        "source": {
                            "filename": os.path.basename(filepath),
                            "method": "pdftotext",
                        },
                    }
                )

        return structured_blocks

    except subprocess.TimeoutExpired:
        print("pdftotext timed out", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("pdftotext not found - install poppler-utils", file=sys.stderr)
        return []
    except Exception as e:
        print(f"pdftotext extraction failed: {e}", file=sys.stderr)
        return []


def _extract_with_pdfminer(filepath: str, exclude_pages: str = None) -> list[dict]:
    """
    Final fallback extraction using pdfminer.six with tunable LAParams.

    Args:
        filepath: Path to the PDF file
        exclude_pages: Page ranges to exclude (e.g., "1,3,5-10,15-20")
    """
    try:
        # Parse page exclusions if provided
        excluded_pages = set()
        if exclude_pages:
            from .page_utils import parse_page_ranges

            try:
                excluded_pages = parse_page_ranges(exclude_pages)
            except ValueError as e:
                print(
                    f"Error parsing page exclusions in pdfminer fallback: {e}",
                    file=sys.stderr,
                )

        # Try different LAParams configurations
        configs = [
            LAParams(char_margin=1.5, word_margin=0.5, line_margin=0.5),
            LAParams(char_margin=2.0, word_margin=0.3, line_margin=0.3),
            LAParams(char_margin=1.0, word_margin=0.8, line_margin=0.8),
        ]

        for i, laparams in enumerate(configs):
            print(f"Trying pdfminer config {i+1}/3", file=sys.stderr)
            raw_text = extract_text(filepath, laparams=laparams)

            # Apply post-processing to fix missing spaces
            repaired_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_text)
            repaired_text = _clean_fallback_text(repaired_text)

            quality = _assess_text_quality(repaired_text)
            print(
                f"pdfminer config {i+1} quality: {quality['quality_score']:.2f}",
                file=sys.stderr,
            )

            if quality["quality_score"] >= 0.7:
                # Parse the text into structured blocks
                structured_blocks = []
                paragraphs = repaired_text.split("\n\n")

                for paragraph in paragraphs:
                    block_text = clean_text(paragraph)
                    if block_text:
                        # Simple heuristic for headings
                        is_heading = (
                            len(block_text.split()) < 15
                            and block_text.istitle()
                            and not block_text.endswith(".")
                        )

                        block_type = "heading" if is_heading else "paragraph"
                        lang = _detect_language(block_text)
                        structured_blocks.append(
                            {
                                "type": block_type,
                                "text": block_text,
                                "language": lang,
                                "source": {
                                    "filename": os.path.basename(filepath),
                                    "method": "pdfminer",
                                },
                            }
                        )

                return structured_blocks

        print("All pdfminer configurations failed quality check", file=sys.stderr)
        return []

    except Exception as e:
        print(f"pdfminer extraction failed: {e}", file=sys.stderr)
        return []


def should_use_pymupdf4llm_cleaning(text: str) -> bool:
    """
    Determine if PyMuPDF4LLM text cleaning should be attempted for a text block.

    Args:
        text: Text to evaluate for cleaning

    Returns:
        True if PyMuPDF4LLM cleaning should be attempted
    """
    if not text or not text.strip():
        return False

    # Use PyMuPDF4LLM cleaning for text blocks that might benefit from it
    text_length = len(text.strip())

    # Skip very short text blocks (likely not worth cleaning)
    if text_length < 20:
        return False

    # Skip very long text blocks (might be too complex or cause performance issues)
    if text_length > 50000:
        return False

    # Check for indicators that text might benefit from cleaning
    import re

    # Look for potential ligature issues
    has_ligatures = bool(re.search(r"[ﬁﬂﬁﬃﬄﬆﬅ]", text))

    # Look for potential word joining issues
    has_joining_issues = bool(re.search(r"[a-z][A-Z]", text))

    # Look for excessive whitespace
    has_whitespace_issues = bool(re.search(r"  +|\n{3,}", text))

    # Look for hyphenation issues
    has_hyphenation_issues = bool(re.search(r"-\s*\n\s*[a-z]", text))

    # Use PyMuPDF4LLM if any potential issues are detected
    return (
        has_ligatures
        or has_joining_issues
        or has_whitespace_issues
        or has_hyphenation_issues
    )


def assess_text_cleaning_quality(original_text: str, cleaned_text: str) -> dict:
    """
    Simple quality assessment for PyMuPDF4LLM text cleaning effectiveness.

    Args:
        original_text: Original text before cleaning
        cleaned_text: Text after PyMuPDF4LLM cleaning

    Returns:
        Simple quality assessment metrics for text cleaning
    """
    if not cleaned_text or not cleaned_text.strip():
        return {
            "quality_score": 0.0,
            "has_content": False,
            "cleaning_effective": False,
            "issues": ["No cleaned text produced"],
        }

    issues = []
    quality_factors = []

    # Content preservation (60% weight)
    original_length = len(original_text.strip())
    cleaned_length = len(cleaned_text.strip())

    if cleaned_length > 0:
        if original_length > 0:
            length_ratio = cleaned_length / original_length
            if 0.8 <= length_ratio <= 1.2:  # Reasonable length preservation
                quality_factors.append(0.6)
            elif 0.6 <= length_ratio <= 1.5:  # Acceptable range
                quality_factors.append(0.4)
            else:
                quality_factors.append(0.2)
                issues.append(f"Significant length change: {length_ratio:.2f}")
        else:
            quality_factors.append(0.6)
    else:
        issues.append("No content after cleaning")

    # Text cleaning effectiveness (40% weight)
    if cleaned_text != original_text:
        quality_factors.append(0.4)
    else:
        quality_factors.append(0.2)

    # Calculate overall quality score
    quality_score = min(sum(quality_factors), 1.0)

    return {
        "quality_score": quality_score,
        "has_content": len(cleaned_text.strip()) > 0,
        "cleaning_effective": cleaned_text != original_text,
        "length_ratio": cleaned_length / original_length if original_length > 0 else 0,
        "issues": issues,
    }


def execute_fallback_extraction(
    filepath: str, exclude_pages: str = None, fallback_reason: str = None
) -> list[dict]:
    """
    Execute the traditional three-tier fallback extraction system.

    Args:
        filepath: Path to the PDF file
        exclude_pages: Page ranges to exclude (e.g., "1,3,5-10,15-20")
        fallback_reason: Reason for falling back (for logging)

    Returns:
        List of extracted text blocks using traditional methods
    """
    import logging

    logger = logging.getLogger(__name__)

    if fallback_reason:
        logger.info(
            f"Executing fallback extraction for {os.path.basename(filepath)}: {fallback_reason}"
        )
    else:
        logger.info(f"Executing fallback extraction for {os.path.basename(filepath)}")

    # Try pdftotext first
    logger.debug("Attempting pdftotext extraction")
    pdftotext_blocks = _extract_with_pdftotext(filepath, exclude_pages)
    if pdftotext_blocks:
        logger.info(f"pdftotext extraction successful: {len(pdftotext_blocks)} blocks")
        return pdftotext_blocks

    # Try pdfminer.six as final fallback
    if PDFMINER_AVAILABLE:
        logger.debug("Attempting pdfminer extraction")
        pdfminer_blocks = _extract_with_pdfminer(filepath, exclude_pages)
        if pdfminer_blocks:
            logger.info(
                f"pdfminer extraction successful: {len(pdfminer_blocks)} blocks"
            )
            return pdfminer_blocks
    else:
        logger.warning("pdfminer.six not available for fallback extraction")

    # All fallback methods failed
    logger.error("All fallback extraction methods failed")
    return []
