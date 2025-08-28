import re
from typing import Any, Dict, List, Optional, cast


TRAILING_PUNCTUATION = (".", "!", "?", ";", ":")
HEADING_STARTERS = {
    "chapter",
    "section",
    "part",
    "appendix",
    "introduction",
    "conclusion",
}


def _has_heading_starter(words: List[str]) -> bool:
    """Return True if the first word suggests a structural heading."""
    return bool(words) and words[0].lower() in HEADING_STARTERS


def _detect_heading_fallback(text: str) -> bool:
    """
    Fallback heading detection using text characteristics when font analysis fails.
    Uses heuristics like text length, capitalization patterns, and punctuation.
    """
    if not text or not text.strip():
        return False

    text = text.strip()
    words = text.split()

    # Very short text (1-3 words) without a terminal period is likely a heading.
    # Short sentences such as "It ended." should remain part of the body text
    # rather than being treated as headings.
    if len(words) <= 3 and not text.endswith(TRAILING_PUNCTUATION):
        return True

    # Text that's all uppercase might be a heading
    if text.isupper() and len(words) <= 8:
        return True

    # Title case text without ending punctuation might be a heading
    if text.istitle() and len(words) <= 10 and not text.endswith(TRAILING_PUNCTUATION):
        return True

    # Text that starts with common heading patterns
    if _has_heading_starter(words) and len(words) <= 8 and not text.endswith(TRAILING_PUNCTUATION):
        return True

    # Text that's mostly numbers (like "1.2.3 Some Topic")
    if (
        len(words) >= 2
        and re.match(r"^[\d\.\-]+$", words[0])
        and len(words) <= 8
        and not text.endswith(TRAILING_PUNCTUATION)
    ):
        return True

    return False


def detect_headings_from_pymupdf4llm_blocks(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract heading information from PyMuPDF4LLM blocks that contain Markdown headers.

    Args:
        blocks: List of text blocks from PyMuPDF4LLM extraction

    Returns:
        List of heading dictionaries with metadata
    """
    headings = []

    for block in blocks:
        metadata = block.get("metadata", {})

        # Check if this block is marked as a heading by PyMuPDF4LLM
        if metadata.get("is_heading", False):
            heading_info = {
                "text": block.get("text", "").strip(),
                "level": metadata.get("heading_level", 1),
                "source": "pymupdf4llm_markdown",
                "block_id": metadata.get("block_id"),
                "extraction_method": "pymupdf4llm",
            }

            # Add additional metadata if available
            if "heading_text" in metadata:
                heading_info["original_heading_text"] = metadata["heading_text"]

            headings.append(heading_info)

    return headings


def detect_headings_from_font_analysis(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract heading information using traditional font-based analysis."""

    def _is_heading(text: str, block_type: str) -> bool:
        declared_heading = block_type == "heading" and not text.endswith(TRAILING_PUNCTUATION)
        return declared_heading or _detect_heading_fallback(text)

    def _make_heading(i: int, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = block.get("text", "").strip()
        if not text or not _is_heading(text, block.get("type", "")):
            return None

        base = {
            "text": text,
            "level": _estimate_heading_level(text),
            "source": "font_analysis",
            "block_id": i,
            "extraction_method": block.get("source", {}).get("method", "traditional"),
        }

        return (
            base
            if "source" not in block
            else {
                **base,
                "page": block["source"].get("page"),
                "filename": block["source"].get("filename"),
            }
        )

    return [
        heading
        for heading in (_make_heading(i, block) for i, block in enumerate(blocks))
        if heading is not None
    ]


def _estimate_heading_level(text: str) -> int:
    """
    Estimate heading level based on text characteristics.

    Args:
        text: Heading text

    Returns:
        Estimated heading level (1-6)
    """
    text = text.strip()
    words = text.split()

    # Very short text (1-2 words) is likely a high-level heading
    if len(words) <= 2:
        return 1

    # All uppercase text is likely a high-level heading
    if text.isupper():
        return 1 if len(words) <= 4 else 2

    # Text starting with "Chapter" or similar is likely level 1
    first_word = words[0].lower() if words else ""
    if first_word in ["chapter", "part", "book", "volume"]:
        return 1

    # Text starting with "Section" is likely level 2
    if first_word in ["section", "appendix"]:
        return 2

    # Numbered headings (e.g., "1.2.3 Topic")
    if re.match(r"^\d+\.", text):
        # Count the number of dots to estimate level
        dots = text.split()[0].count(".")
        return min(dots + 1, 6)

    # Default to level 3 for other headings
    return 3


def detect_headings_hybrid(
    blocks: List[Dict[str, Any]], extraction_method: str = "unknown"
) -> List[Dict[str, Any]]:
    """
    Heading detection that prioritizes traditional font-based analysis.

    In the simplified PyMuPDF4LLM approach, traditional extraction handles all
    structural analysis while PyMuPDF4LLM provides text cleaning only.

    Args:
        blocks: List of text blocks from PDF extraction
        extraction_method: Method used for extraction ('traditional', 'pymupdf4llm_enhanced', etc.)

    Returns:
        List of heading dictionaries with consistent metadata
    """
    # Determine extraction method if not provided
    if extraction_method == "unknown":
        # Check if any blocks have PyMuPDF4LLM text enhancement metadata
        has_pymupdf4llm_enhancement = any(
            block.get("metadata", {}).get("text_enhanced_with_pymupdf4llm", False)
            for block in blocks
        )
        extraction_method = "pymupdf4llm_enhanced" if has_pymupdf4llm_enhancement else "traditional"

    # For PyMuPDF4LLM-enhanced extraction, use traditional font-based analysis
    # since PyMuPDF4LLM is only used for text cleaning in the simplified approach
    if extraction_method == "pymupdf4llm_enhanced":
        return detect_headings_from_font_analysis(blocks)
    elif extraction_method == "pymupdf4llm":
        # Legacy PyMuPDF4LLM extraction without structural analysis
        pymupdf4llm_headings = detect_headings_from_pymupdf4llm_blocks(blocks)

        if pymupdf4llm_headings:
            return pymupdf4llm_headings
        else:
            # Fallback to font analysis if no PyMuPDF4LLM headings found
            return detect_headings_from_font_analysis(blocks)
    else:
        # Use traditional font-based detection
        return detect_headings_from_font_analysis(blocks)


def enhance_blocks_with_heading_metadata(
    blocks: List[Dict[str, Any]], extraction_method: str = "unknown"
) -> List[Dict[str, Any]]:
    """Delegate to the registered ``heading_detect`` pass."""

    from pdf_chunker.framework import Artifact
    from pdf_chunker.passes.heading_detect import heading_detect

    artifact = Artifact(payload=blocks, meta={"extraction_method": extraction_method})
    return cast(List[Dict[str, Any]], heading_detect(artifact).payload)


def get_heading_hierarchy(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract heading hierarchy from enhanced blocks.

    Args:
        blocks: List of enhanced text blocks with heading metadata

    Returns:
        List of headings in hierarchical order with parent-child relationships
    """
    headings: list[dict] = []
    heading_stack: list[dict] = []  # Stack to track parent headings

    for block in blocks:
        if not block.get("is_heading", False):
            continue

        heading_info = {
            "text": block.get("text", "").strip(),
            "level": block.get("heading_level", 1),
            "source": block.get("heading_source", "unknown"),
            "block_index": blocks.index(block),
            "children": [],
            "parent": None,
        }

        # Determine parent-child relationships
        current_level = heading_info["level"]

        # Pop headings from stack that are at same or lower level
        while heading_stack and heading_stack[-1]["level"] >= current_level:
            heading_stack.pop()

        # Set parent relationship
        if heading_stack:
            parent = heading_stack[-1]
            heading_info["parent"] = parent["text"]
            parent["children"].append(heading_info["text"])

        # Add to stack and results
        heading_stack.append(heading_info)
        headings.append(heading_info)

    return headings


def validate_heading_consistency(
    pymupdf4llm_headings: List[Dict[str, Any]],
    traditional_headings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate consistency between PyMuPDF4LLM and traditional heading detection.

    Args:
        pymupdf4llm_headings: Headings detected by PyMuPDF4LLM
        traditional_headings: Headings detected by traditional methods

    Returns:
        Validation report with consistency metrics
    """
    # Extract heading texts for comparison
    pymupdf4llm_texts = {h["text"].strip().lower() for h in pymupdf4llm_headings}
    traditional_texts = {h["text"].strip().lower() for h in traditional_headings}

    # Calculate overlap and differences
    common_headings = pymupdf4llm_texts & traditional_texts
    pymupdf4llm_only = pymupdf4llm_texts - traditional_texts
    traditional_only = traditional_texts - pymupdf4llm_texts

    # Calculate consistency metrics
    total_unique_headings = len(pymupdf4llm_texts | traditional_texts)
    overlap_ratio = len(common_headings) / total_unique_headings if total_unique_headings > 0 else 0

    return {
        "total_pymupdf4llm_headings": len(pymupdf4llm_headings),
        "total_traditional_headings": len(traditional_headings),
        "common_headings": len(common_headings),
        "pymupdf4llm_only": len(pymupdf4llm_only),
        "traditional_only": len(traditional_only),
        "overlap_ratio": overlap_ratio,
        "consistency_score": overlap_ratio,
        "common_heading_texts": list(common_headings),
        "pymupdf4llm_only_texts": list(pymupdf4llm_only),
        "traditional_only_texts": list(traditional_only),
    }
