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

from .text_cleaning import _clean_text
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
    
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    if not non_empty_lines:
        return {"avg_line_length": 0, "space_density": 0, "quality_score": 0}
    
    # Calculate average line length
    avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
    
    # Calculate space density (spaces per character)
    total_chars = sum(len(line) for line in non_empty_lines)
    total_spaces = sum(line.count(' ') for line in non_empty_lines)
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
        "quality_score": quality_score
    }

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
                print(f"Error parsing page exclusions in pdftotext fallback: {e}", file=sys.stderr)

        # Build pdftotext command with page exclusions if needed
        cmd = ['pdftotext', '-layout']

        # pdftotext doesn't have built-in page exclusion, so we'll extract all pages
        # and filter the results afterward
        cmd.extend([filepath, '-'])

        # Try pdftotext with -layout flag
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"pdftotext failed with return code {result.returncode}", file=sys.stderr)
            return []
        
        raw_text = result.stdout
        quality = _assess_text_quality(raw_text)
        print(f"pdftotext extraction quality: {quality['quality_score']:.2f}", file=sys.stderr)
        
        if quality['quality_score'] < 0.7:
            return []
        
        # Parse the text into structured blocks
        structured_blocks = []
        paragraphs = raw_text.split('\n\n')
        
        for paragraph in paragraphs:
            block_text = _clean_text(paragraph)
            if block_text:
                # Simple heuristic: short paragraphs with title case might be headings
                is_heading = (len(block_text.split()) < 15 and 
                             block_text.istitle() and 
                             not block_text.endswith('.'))
                
                block_type = "heading" if is_heading else "paragraph"
                lang = _detect_language(block_text)
                structured_blocks.append({
                    "type": block_type,
                    "text": block_text,
                    "language": lang,
                    "source": {"filename": os.path.basename(filepath), "method": "pdftotext"}
                })
        
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
                print(f"Error parsing page exclusions in pdfminer fallback: {e}", file=sys.stderr)

        # Try different LAParams configurations
        configs = [
            LAParams(char_margin=1.5, word_margin=0.5, line_margin=0.5),
            LAParams(char_margin=2.0, word_margin=0.3, line_margin=0.3),
            LAParams(char_margin=1.0, word_margin=0.8, line_margin=0.8)
        ]
        
        
        for i, laparams in enumerate(configs):
            print(f"Trying pdfminer config {i+1}/3", file=sys.stderr)
            raw_text = extract_text(filepath, laparams=laparams)
            
            # Apply post-processing to fix missing spaces
            repaired_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_text)
            
            quality = _assess_text_quality(repaired_text)
            print(f"pdfminer config {i+1} quality: {quality['quality_score']:.2f}", file=sys.stderr)
            
            if quality['quality_score'] >= 0.7:
                # Parse the text into structured blocks
                structured_blocks = []
                paragraphs = repaired_text.split('\n\n')
                
                for paragraph in paragraphs:
                    block_text = _clean_text(paragraph)
                    if block_text:
                        # Simple heuristic for headings
                        is_heading = (len(block_text.split()) < 15 and 
                                     block_text.istitle() and 
                                     not block_text.endswith('.'))
                        
                        block_type = "heading" if is_heading else "paragraph"
                        lang = _detect_language(block_text)
                        structured_blocks.append({
                            "type": block_type,
                            "text": block_text,
                            "language": lang,
                            "source": {"filename": os.path.basename(filepath), "method": "pdfminer"}
                        })
                
                return structured_blocks
        
        print("All pdfminer configurations failed quality check", file=sys.stderr)
        return []
        
    except Exception as e:
        print(f"pdfminer extraction failed: {e}", file=sys.stderr)
        return []

def assess_pymupdf4llm_extraction_quality(blocks: list[dict], metadata: dict = None) -> dict:
    """
    Assess the quality of PyMuPDF4LLM extraction results for fallback decisions.
    Enhanced to prioritize structural fidelity including heading preservation.
    
    Args:
        blocks: List of extracted text blocks from PyMuPDF4LLM
        metadata: Optional extraction metadata from PyMuPDF4LLM
        
    Returns:
        Quality assessment metrics compatible with existing fallback logic
    """
    if not blocks:
        return {
            "quality_score": 0.0,
            "avg_line_length": 0,
            "space_density": 0,
            "has_content": False,
            "has_headings": False,
            "total_text_length": 0,
            "block_count": 0,
            "heading_count": 0,
            "structural_fidelity_score": 0.0,
            "issues": ["No content extracted from PyMuPDF4LLM"]
        }
    
    # Extract text content from blocks
    all_text = []
    heading_count = 0
    text_block_count = 0
    
    for block in blocks:
        text = block.get('text', '')
        if text:
            all_text.append(text)
            
        # Count headings with enhanced detection
        is_heading = (
            block.get('metadata', {}).get('is_heading', False) or 
            block.get('type') == 'heading' or
            block.get('is_heading', False)
        )
        
        if is_heading:
            heading_count += 1
        else:
            text_block_count += 1
    
    combined_text = '\n'.join(all_text)
    
    # Use existing quality assessment logic for base text quality
    base_quality = _assess_text_quality(combined_text)
    
    # Enhanced assessment for PyMuPDF4LLM-specific factors with structural emphasis
    issues = []
    quality_factors = []
    
    # Base content quality (40% weight - reduced to emphasize structure)
    if base_quality['quality_score'] > 0:
        quality_factors.append(base_quality['quality_score'] * 0.4)
    else:
        issues.append("Poor text extraction quality")
    
    # Structural fidelity assessment (35% weight - increased importance)
    structural_score = _assess_structural_fidelity(blocks, heading_count, text_block_count)
    quality_factors.append(structural_score * 0.35)
    
    if structural_score < 0.5:
        issues.append("Poor structural preservation")
    
    # Heading detection bonus (15% weight - maintained importance)
    if heading_count > 0:
        # Scale heading bonus based on text-to-heading ratio
        if text_block_count > 0:
            ratio = text_block_count / heading_count
            if 2 <= ratio <= 15:  # Good document structure
                quality_factors.append(0.15)
            elif ratio > 15:  # Too few headings
                quality_factors.append(0.08)
                issues.append("Insufficient heading detection")
            else:  # Too many headings
                quality_factors.append(0.10)
        else:
            quality_factors.append(0.10)
    else:
        issues.append("No headings detected")
    
    # Block structure assessment (10% weight)
    if len(blocks) > 1:
        quality_factors.append(0.1)
    else:
        issues.append("Insufficient block segmentation")
    
    # Calculate final quality score
    final_quality_score = min(sum(quality_factors), 1.0)
    
    return {
        "quality_score": final_quality_score,
        "avg_line_length": base_quality['avg_line_length'],
        "space_density": base_quality['space_density'],
        "has_content": len(combined_text.strip()) > 0,
        "has_headings": heading_count > 0,
        "total_text_length": len(combined_text),
        "block_count": len(blocks),
        "heading_count": heading_count,
        "text_block_count": text_block_count,
        "structural_fidelity_score": structural_score,
        "extraction_method": "hybrid_pymupdf4llm_with_structure",
        "issues": issues
    }

def _assess_structural_fidelity(blocks: list[dict], heading_count: int, text_block_count: int) -> float:
    """
    Assess structural fidelity of extracted blocks focusing on heading preservation
    and proper block segmentation.
    
    Args:
        blocks: List of extracted blocks
        heading_count: Number of heading blocks detected
        text_block_count: Number of text blocks detected
        
    Returns:
        Structural fidelity score (0.0 to 1.0)
    """
    if not blocks:
        return 0.0
    
    score_factors = []
    
    # Block type diversity (25% of structural score)
    if heading_count > 0 and text_block_count > 0:
        score_factors.append(0.25)
    elif heading_count > 0 or text_block_count > 0:
        score_factors.append(0.15)
    
    # Heading distribution quality (35% of structural score)
    if heading_count > 0:
        total_blocks = len(blocks)
        heading_ratio = heading_count / total_blocks
        
        if 0.05 <= heading_ratio <= 0.3:  # 5-30% headings is reasonable
            score_factors.append(0.35)
        elif 0.02 <= heading_ratio <= 0.5:  # Acceptable range
            score_factors.append(0.25)
        else:
            score_factors.append(0.1)  # Too few or too many headings
    
    # Block metadata quality (25% of structural score)
    blocks_with_metadata = sum(1 for block in blocks if block.get('metadata') or block.get('source'))
    if blocks_with_metadata > 0:
        metadata_ratio = blocks_with_metadata / len(blocks)
        score_factors.append(0.25 * metadata_ratio)
    
    # Text quality consistency across blocks (15% of structural score)
    non_empty_blocks = sum(1 for block in blocks if block.get('text', '').strip())
    if non_empty_blocks > 0:
        consistency_ratio = non_empty_blocks / len(blocks)
        score_factors.append(0.15 * consistency_ratio)
    
    return min(sum(score_factors), 1.0)

def should_fallback_from_pymupdf4llm(quality_assessment: dict, min_quality_threshold: float = 0.65) -> tuple[bool, str]:
    """
    Determine if fallback from PyMuPDF4LLM to traditional extraction methods is needed.
    Enhanced to prioritize structural fidelity and heading preservation.
    
    Args:
        quality_assessment: Quality metrics from assess_pymupdf4llm_extraction_quality
        min_quality_threshold: Minimum quality score to accept PyMuPDF4LLM results (raised to 0.65)
        
    Returns:
        Tuple of (should_fallback, reason)
    """
    if not quality_assessment.get('has_content', False):
        return True, "No content extracted"
    
    quality_score = quality_assessment.get('quality_score', 0)
    if quality_score < min_quality_threshold:
        issues = quality_assessment.get('issues', [])
        reason = f"Quality score {quality_score:.2f} below threshold {min_quality_threshold}"
        if issues:
            reason += f". Issues: {', '.join(issues)}"
        return True, reason
    
    # Enhanced structural fidelity checks
    structural_score = quality_assessment.get('structural_fidelity_score', 0)
    if structural_score < 0.4:
        return True, f"Poor structural fidelity (score: {structural_score:.2f})"
    
    # Heading preservation check
    heading_count = quality_assessment.get('heading_count', 0)
    text_block_count = quality_assessment.get('text_block_count', 0)
    
    # For documents with reasonable length, expect some headings
    total_text_length = quality_assessment.get('total_text_length', 0)
    if total_text_length > 2000 and heading_count == 0:
        return True, "No headings detected in substantial document"
    
    # Check for reasonable text-to-heading ratio
    if heading_count > 0 and text_block_count > 0:
        ratio = text_block_count / heading_count
        if ratio > 20:  # Too few headings for document size
            return True, f"Insufficient heading detection (ratio: {ratio:.1f})"
    
    # Additional checks for specific failure modes
    if quality_assessment.get('total_text_length', 0) < 100:
        return True, "Extracted text too short"
    
    if quality_assessment.get('block_count', 0) == 0:
        return True, "No text blocks extracted"
    
    # PyMuPDF4LLM passed enhanced quality checks
    return False, "Quality and structural fidelity acceptable"

def execute_fallback_extraction(filepath: str, exclude_pages: str = None, fallback_reason: str = None) -> list[dict]:
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
        logger.info(f"Executing fallback extraction for {os.path.basename(filepath)}: {fallback_reason}")
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
            logger.info(f"pdfminer extraction successful: {len(pdfminer_blocks)} blocks")
            return pdfminer_blocks
    else:
        logger.warning("pdfminer.six not available for fallback extraction")

    # All fallback methods failed
    logger.error("All fallback extraction methods failed")
    return []
