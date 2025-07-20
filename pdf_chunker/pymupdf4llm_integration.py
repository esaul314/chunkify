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
    return PYMUPDF4LLM_AVAILABLE and pymupdf4llm is not None


def extract_with_pymupdf4llm(
    pdf_path: str,
    exclude_pages: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract text from PDF using PyMuPDF4LLM with traditional font-based structural analysis.
    
    This hybrid approach combines PyMuPDF4LLM's superior text quality with the traditional
    approach's proven font-based heading detection and block preservation logic.
    
    Args:
        pdf_path: Path to the PDF file
        exclude_pages: Comma-separated string of page numbers to exclude (e.g., "1,3,5")
        
    Returns:
        Tuple of (blocks, metadata) where blocks is a list of text blocks with metadata
        and metadata contains extraction information
        
    Raises:
        PyMuPDF4LLMExtractionError: If extraction fails
    """
    if not is_pymupdf4llm_available():
        raise PyMuPDF4LLMExtractionError("PyMuPDF4LLM is not available")
    
    start_time = time.time()
    
    try:
        # Parse excluded pages
        excluded_page_numbers = set()
        if exclude_pages:
            try:
                excluded_page_numbers = {int(p.strip()) for p in exclude_pages.split(',') if p.strip()}
            except ValueError as e:
                logger.warning(f"Invalid page exclusion format: {exclude_pages}. Error: {e}")
        
        logger.debug(f"Starting hybrid extraction for: {pdf_path}")
        
        # Step 1: Extract text using PyMuPDF4LLM for superior text quality
        logger.debug("Extracting text with PyMuPDF4LLM...")
        markdown_text = _call_pymupdf4llm_api(pdf_path, None)
        
        if not markdown_text or not markdown_text.strip():
            raise PyMuPDF4LLMExtractionError("PyMuPDF4LLM returned empty text")
        
        # Step 2: Run traditional font-based analysis in parallel for structural metadata
        logger.debug("Running traditional font-based analysis for structural metadata...")
        traditional_blocks = _extract_traditional_blocks_for_structure(pdf_path, excluded_page_numbers)
        
        # Step 3: Combine PyMuPDF4LLM text quality with traditional structural analysis
        logger.debug("Combining PyMuPDF4LLM text with traditional structural metadata...")
        blocks = _convert_markdown_to_blocks_with_structure(
            markdown_text, 
            traditional_blocks,
            excluded_page_numbers
        )
        
        # Generate extraction metadata
        extraction_time = time.time() - start_time
        metadata = {
            'extraction_method': 'hybrid_pymupdf4llm_with_structure',
            'extraction_time': extraction_time,
            'markdown_length': len(markdown_text),
            'traditional_blocks_count': len(traditional_blocks),
            'total_blocks': len(blocks),
            'excluded_pages': list(excluded_page_numbers) if excluded_page_numbers else [],
            'has_headings': any(block.get('metadata', {}).get('is_heading', False) for block in blocks)
        }
        
        logger.info(f"Hybrid extraction completed: {len(blocks)} blocks in {extraction_time:.2f}s")
        
        return blocks, metadata
        
    except Exception as e:
        extraction_time = time.time() - start_time
        error_msg = f"Hybrid PyMuPDF4LLM extraction failed after {extraction_time:.2f}s: {str(e)}"
        logger.error(error_msg)
        raise PyMuPDF4LLMExtractionError(error_msg) from e


def _extract_traditional_blocks_for_structure(
    pdf_path: str, 
    excluded_page_numbers: set
) -> List[Dict[str, Any]]:
    """
    Extract blocks using traditional font-based analysis for structural metadata.
    
    Args:
        pdf_path: Path to the PDF file
        excluded_page_numbers: Set of page numbers to exclude
        
    Returns:
        List of traditional blocks with font-based structural metadata
    """
    import fitz  # PyMuPDF
    from .pdf_parsing import extract_blocks_from_page, is_artifact_block
    
    doc = fitz.open(pdf_path)
    traditional_blocks = []
    
    try:
        for page_num, page in enumerate(doc, start=1):
            if page_num in excluded_page_numbers:
                continue
            
            # Extract blocks using traditional font-based analysis
            page_blocks = extract_blocks_from_page(page, page_num, pdf_path)
            traditional_blocks.extend(page_blocks)
    
    finally:
        doc.close()
    
    return traditional_blocks


def _convert_markdown_to_blocks_with_structure(
    markdown_text: str,
    traditional_blocks: List[Dict[str, Any]],
    excluded_page_numbers: Optional[set] = None
) -> List[Dict[str, Any]]:
    """
    Convert PyMuPDF4LLM Markdown output to structured text blocks using traditional
    font-based structural analysis for heading detection and block boundaries.
    
    Args:
        markdown_text: Raw markdown text from PyMuPDF4LLM
        traditional_blocks: Blocks from traditional extraction with structural metadata
        excluded_page_numbers: Set of page numbers to exclude from blocks
        
    Returns:
        List of text blocks combining PyMuPDF4LLM text quality with traditional structure
    """
    from .text_cleaning import clean_text
    
    # Create a mapping of traditional block text to structural metadata
    traditional_structure_map = {}
    for block in traditional_blocks:
        block_text = block.get('text', '').strip()
        if block_text:
            # Use cleaned text as key for better matching
            cleaned_key = clean_text(block_text).lower()
            traditional_structure_map[cleaned_key] = {
                'type': block.get('type', 'paragraph'),
                'is_heading': block.get('type') == 'heading',
                'page': block.get('source', {}).get('page', 1),
                'language': block.get('language', 'unknown'),
                'original_text': block_text
            }
    
    # Parse PyMuPDF4LLM markdown into blocks
    blocks = []
    lines = markdown_text.split('\n')
    
    current_block_lines = []
    block_id = 0
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            if current_block_lines:
                # End current block on empty line
                block = _create_hybrid_text_block(
                    current_block_lines, 
                    block_id,
                    traditional_structure_map
                )
                if block:
                    blocks.append(block)
                    block_id += 1
                current_block_lines = []
            continue
        
        # Detect markdown headings
        if line.startswith('#'):
            # Save previous block if it exists
            if current_block_lines:
                block = _create_hybrid_text_block(
                    current_block_lines, 
                    block_id,
                    traditional_structure_map
                )
                if block:
                    blocks.append(block)
                    block_id += 1
                current_block_lines = []
            
            # Extract heading text and level
            heading_level = len(line) - len(line.lstrip('#'))
            heading_text = line.lstrip('#').strip()
            
            # Create heading block with traditional structure if available
            heading_block = _create_hybrid_text_block(
                [heading_text], 
                block_id,
                traditional_structure_map,
                is_heading=True,
                heading_level=heading_level
            )
            if heading_block:
                blocks.append(heading_block)
                block_id += 1
        else:
            # Regular text line
            current_block_lines.append(line)
    
    # Add final block if it exists
    if current_block_lines:
        block = _create_hybrid_text_block(
            current_block_lines, 
            block_id,
            traditional_structure_map
        )
        if block:
            blocks.append(block)
    
    # Apply traditional continuation merging logic
    from .pdf_parsing import merge_continuation_blocks
    merged_blocks = merge_continuation_blocks(blocks)
    
    logger.debug(f"Created {len(blocks)} blocks, merged to {len(merged_blocks)} blocks")
    
    return merged_blocks


def _create_hybrid_text_block(
    lines: List[str], 
    block_id: int,
    traditional_structure_map: Dict[str, Dict[str, Any]],
    is_heading: bool = False,
    heading_level: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Create a text block combining PyMuPDF4LLM text quality with traditional structural metadata.
    
    Args:
        lines: List of text lines for the block
        block_id: Unique block identifier
        traditional_structure_map: Mapping of text to traditional structural metadata
        is_heading: Whether this block is a heading (from PyMuPDF4LLM)
        heading_level: Heading level (1-6) for heading blocks
        
    Returns:
        Text block dictionary or None if block is empty
    """
    from .text_cleaning import clean_text
    
    if not lines:
        return None
    
    text = '\n'.join(lines).strip()
    if not text:
        return None
    
    # Clean the text using PyMuPDF4LLM's superior text quality
    cleaned_text = clean_text(text)
    
    # Look up traditional structural metadata
    lookup_key = cleaned_text.lower()
    traditional_metadata = traditional_structure_map.get(lookup_key, {})
    
    # Determine block type using traditional analysis if available, otherwise PyMuPDF4LLM
    if traditional_metadata:
        block_type = traditional_metadata.get('type', 'paragraph')
        is_heading_final = traditional_metadata.get('is_heading', False)
        page_num = traditional_metadata.get('page', 1)
        language = traditional_metadata.get('language', 'unknown')
    else:
        # Fall back to PyMuPDF4LLM analysis
        block_type = "heading" if is_heading else "paragraph"
        is_heading_final = is_heading
        page_num = 1
        language = 'unknown'
    
    # Create block structure compatible with existing pipeline
    block = {
        'type': block_type,
        'text': cleaned_text,  # Use PyMuPDF4LLM's superior text quality
        'language': language,
        'source': {
            'filename': 'hybrid_extraction',
            'page': page_num
        }
    }
    
    # Add heading-specific metadata if this is a heading
    if is_heading_final:
        block['metadata'] = {
            'block_id': block_id,
            'extraction_method': 'hybrid_pymupdf4llm_with_structure',
            'is_heading': True,
            'heading_level': heading_level if is_heading else 1,
            'heading_source': 'traditional_font_analysis' if traditional_metadata else 'pymupdf4llm_markdown'
        }
    
    return block


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
    # Try different API patterns for PyMuPDF4LLM
    api_methods = [
        # Pattern 1: Direct function calls
        ('to_markdown', lambda: pymupdf4llm.to_markdown(pdf_path, pages=pages)),
        ('extract', lambda: pymupdf4llm.extract(pdf_path, pages=pages)),
        ('convert', lambda: pymupdf4llm.convert(pdf_path, pages=pages)),
        ('parse', lambda: pymupdf4llm.parse(pdf_path, pages=pages)),
        
        # Pattern 2: Simple function calls without pages parameter
        ('to_markdown_simple', lambda: pymupdf4llm.to_markdown(pdf_path)),
        ('extract_simple', lambda: pymupdf4llm.extract(pdf_path)),
        ('convert_simple', lambda: pymupdf4llm.convert(pdf_path)),
        ('parse_simple', lambda: pymupdf4llm.parse(pdf_path)),
    ]
    
    # Try class-based APIs
    class_methods = [
        'LlamaParseReader',
        'PyMuPDFReader', 
        'PDFReader',
        'DocumentReader'
    ]
    
    # Try direct function calls first
    for method_name, method_call in api_methods:
        try:
            if hasattr(pymupdf4llm, method_name.replace('_simple', '')):
                logger.debug(f"Trying PyMuPDF4LLM method: {method_name}")
                result = method_call()
                if isinstance(result, str) and result.strip():
                    logger.debug(f"Successfully extracted using {method_name}")
                    return result
        except Exception as e:
            logger.debug(f"Method {method_name} failed: {e}")
            continue
    
    # Try class-based APIs
    for class_name in class_methods:
        if hasattr(pymupdf4llm, class_name):
            try:
                logger.debug(f"Trying PyMuPDF4LLM class: {class_name}")
                cls = getattr(pymupdf4llm, class_name)
                reader = cls()
                
                # Try different method names on the class
                for method_name in ['load_data', 'read', 'extract', 'parse', 'to_markdown']:
                    if hasattr(reader, method_name):
                        try:
                            method = getattr(reader, method_name)
                            result = method(pdf_path)
                            
                            # Handle different return types
                            if isinstance(result, str) and result.strip():
                                logger.debug(f"Successfully extracted using {class_name}.{method_name}")
                                return result
                            elif isinstance(result, list) and result:
                                # Handle list of documents
                                if hasattr(result[0], 'text'):
                                    text = '\n'.join([doc.text for doc in result])
                                else:
                                    text = '\n'.join([str(doc) for doc in result])
                                
                                if text.strip():
                                    logger.debug(f"Successfully extracted using {class_name}.{method_name}")
                                    return text
                        except Exception as e:
                            logger.debug(f"{class_name}.{method_name} failed: {e}")
                            continue
            except Exception as e:
                logger.debug(f"Class {class_name} instantiation failed: {e}")
                continue
    
    # If we get here, no method worked
    available_attrs = [attr for attr in dir(pymupdf4llm) if not attr.startswith('_')]
    raise PyMuPDF4LLMExtractionError(
        f"Could not find working PyMuPDF4LLM extraction method. "
        f"Available attributes: {available_attrs}"
    )


def _convert_markdown_to_blocks(
    markdown_text: str, 
    excluded_page_numbers: Optional[set] = None
) -> List[Dict[str, Any]]:
    """
    Convert PyMuPDF4LLM Markdown output to structured text blocks.
    
    Args:
        markdown_text: Raw markdown text from PyMuPDF4LLM
        excluded_page_numbers: Set of page numbers to exclude from blocks
        
    Returns:
        List of text blocks with metadata compatible with existing pipeline
    """
    blocks = []
    lines = markdown_text.split('\n')
    
    current_block_lines = []
    current_heading = None
    block_id = 0
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            if current_block_lines:
                # End current block on empty line
                block = _create_text_block(
                    current_block_lines, 
                    current_heading, 
                    block_id,
                    is_heading=False
                )
                if block:
                    blocks.append(block)
                    block_id += 1
                current_block_lines = []
            continue
        
        # Detect markdown headings
        if line.startswith('#'):
            # Save previous block if it exists
            if current_block_lines:
                block = _create_text_block(
                    current_block_lines, 
                    current_heading, 
                    block_id,
                    is_heading=False
                )
                if block:
                    blocks.append(block)
                    block_id += 1
                current_block_lines = []
            
            # Extract heading text and level
            heading_level = len(line) - len(line.lstrip('#'))
            heading_text = line.lstrip('#').strip()
            
            # Create heading block
            heading_block = _create_text_block(
                [heading_text], 
                heading_text, 
                block_id,
                is_heading=True,
                heading_level=heading_level
            )
            if heading_block:
                blocks.append(heading_block)
                block_id += 1
            
            # Update current heading context
            current_heading = heading_text
        else:
            # Regular text line
            current_block_lines.append(line)
    
    # Add final block if it exists
    if current_block_lines:
        block = _create_text_block(
            current_block_lines, 
            current_heading, 
            block_id,
            is_heading=False
        )
        if block:
            blocks.append(block)
    
    # Filter out blocks from excluded pages if needed
    # Note: PyMuPDF4LLM doesn't provide page information in its output,
    # so we can't filter by page number. This is a limitation of the approach.
    if excluded_page_numbers:
        logger.warning(
            "Page exclusion requested but PyMuPDF4LLM doesn't provide page information. "
            "All extracted content will be included."
        )
    
    return blocks


def _create_text_block(
    lines: List[str], 
    current_heading: Optional[str], 
    block_id: int,
    is_heading: bool = False,
    heading_level: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Create a text block with metadata compatible with existing pipeline.
    
    Args:
        lines: List of text lines for the block
        current_heading: Current section heading context
        block_id: Unique block identifier
        is_heading: Whether this block is a heading
        heading_level: Heading level (1-6) for heading blocks
        
    Returns:
        Text block dictionary or None if block is empty
    """
    if not lines:
        return None
    
    text = '\n'.join(lines).strip()
    if not text:
        return None
    
    # Create block metadata compatible with existing pipeline
    metadata = {
        'block_id': block_id,
        'extraction_method': 'pymupdf4llm',
        'is_heading': is_heading,
        'source': 'pymupdf4llm_markdown'
    }
    
    # Add heading-specific metadata
    if is_heading:
        metadata.update({
            'heading_level': heading_level,
            'heading_text': text
        })
    else:
        # Add context metadata for regular text blocks
        if current_heading:
            metadata['section_heading'] = current_heading
    
    # Create block structure compatible with existing pipeline
    block = {
        'text': text,
        'metadata': metadata
    }
    
    return block


def assess_pymupdf4llm_quality(blocks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess the quality of PyMuPDF4LLM extraction results with emphasis on structural fidelity.
    
    Args:
        blocks: List of extracted text blocks
        metadata: Extraction metadata
        
    Returns:
        Quality assessment metrics compatible with enhanced fallback logic
    """
    if not blocks:
        return {
            'quality_score': 0.0,
            'has_content': False,
            'has_headings': False,
            'avg_block_length': 0,
            'total_text_length': 0,
            'block_count': 0,
            'heading_count': 0,
            'text_block_count': 0,
            'structural_fidelity_score': 0.0,
            'issues': ['No content extracted']
        }
    
    # Calculate basic metrics
    total_text_length = sum(len(block.get('text', '')) for block in blocks)
    
    # Enhanced heading detection
    heading_blocks = []
    text_blocks = []
    
    for block in blocks:
        is_heading = (
            block.get('metadata', {}).get('is_heading', False) or
            block.get('type') == 'heading' or
            block.get('is_heading', False)
        )
        
        if is_heading:
            heading_blocks.append(block)
        else:
            text_blocks.append(block)
    
    heading_count = len(heading_blocks)
    text_block_count = len(text_blocks)
    avg_block_length = total_text_length / len(blocks) if blocks else 0
    
    # Assess quality factors with structural emphasis
    issues = []
    quality_factors = []
    
    # Content presence (30% weight - reduced)
    if total_text_length > 0:
        quality_factors.append(0.3)
    else:
        issues.append('No text content')
    
    # Structural fidelity (40% weight - increased)
    structural_score = _assess_block_structural_fidelity(blocks, heading_count, text_block_count)
    quality_factors.append(structural_score * 0.4)
    
    if structural_score < 0.5:
        issues.append('Poor structural preservation')
    
    # Heading detection quality (20% weight - maintained)
    if heading_blocks:
        # Assess heading quality based on distribution and metadata
        heading_quality = _assess_heading_quality(heading_blocks, text_blocks)
        quality_factors.append(heading_quality * 0.2)
        
        if heading_quality < 0.5:
            issues.append('Poor heading detection quality')
    else:
        issues.append('No headings detected')
    
    # Block structure consistency (10% weight)
    if len(blocks) > 1:
        consistency_score = _assess_block_consistency(blocks)
        quality_factors.append(consistency_score * 0.1)
    
    # Calculate overall quality score
    quality_score = sum(quality_factors)
    
    return {
        'quality_score': min(quality_score, 1.0),
        'has_content': total_text_length > 0,
        'has_headings': len(heading_blocks) > 0,
        'avg_block_length': avg_block_length,
        'total_text_length': total_text_length,
        'block_count': len(blocks),
        'heading_count': heading_count,
        'text_block_count': text_block_count,
        'structural_fidelity_score': structural_score,
        'extraction_time': metadata.get('extraction_time', 0),
        'extraction_method': 'hybrid_pymupdf4llm_with_structure',
        'issues': issues
    }

def _assess_block_structural_fidelity(blocks: List[Dict[str, Any]], heading_count: int, text_block_count: int) -> float:
    """
    Assess the structural fidelity of extracted blocks.
    
    Args:
        blocks: List of extracted blocks
        heading_count: Number of heading blocks
        text_block_count: Number of text blocks
        
    Returns:
        Structural fidelity score (0.0 to 1.0)
    """
    if not blocks:
        return 0.0
    
    score_components = []
    
    # Block type diversity
    if heading_count > 0 and text_block_count > 0:
        score_components.append(0.3)
    elif heading_count > 0 or text_block_count > 0:
        score_components.append(0.15)
    
    # Metadata preservation
    blocks_with_metadata = sum(1 for block in blocks if 
                              block.get('metadata') or 
                              block.get('source') or 
                              block.get('type'))
    if blocks_with_metadata > 0:
        metadata_ratio = blocks_with_metadata / len(blocks)
        score_components.append(0.3 * metadata_ratio)
    
    # Text distribution quality
    non_empty_blocks = sum(1 for block in blocks if block.get('text', '').strip())
    if non_empty_blocks > 0:
        text_ratio = non_empty_blocks / len(blocks)
        score_components.append(0.2 * text_ratio)
    
    # Heading distribution
    if heading_count > 0:
        heading_ratio = heading_count / len(blocks)
        if 0.05 <= heading_ratio <= 0.3:  # Reasonable heading density
            score_components.append(0.2)
        elif 0.02 <= heading_ratio <= 0.5:  # Acceptable range
            score_components.append(0.1)
    
    return min(sum(score_components), 1.0)

def _assess_heading_quality(heading_blocks: List[Dict[str, Any]], text_blocks: List[Dict[str, Any]]) -> float:
    """
    Assess the quality of heading detection and metadata.
    
    Args:
        heading_blocks: List of detected heading blocks
        text_blocks: List of text blocks
        
    Returns:
        Heading quality score (0.0 to 1.0)
    """
    if not heading_blocks:
        return 0.0
    
    quality_components = []
    
    # Heading metadata quality
    headings_with_levels = sum(1 for block in heading_blocks if 
                              block.get('metadata', {}).get('heading_level') or
                              block.get('heading_level'))
    if headings_with_levels > 0:
        level_ratio = headings_with_levels / len(heading_blocks)
        quality_components.append(0.4 * level_ratio)
    
    # Heading text quality (not too short, not too long)
    appropriate_length_headings = 0
    for block in heading_blocks:
        text = block.get('text', '').strip()
        word_count = len(text.split())
        if 1 <= word_count <= 15:  # Reasonable heading length
            appropriate_length_headings += 1
    
    if appropriate_length_headings > 0:
        length_ratio = appropriate_length_headings / len(heading_blocks)
        quality_components.append(0.3 * length_ratio)
    
    # Text-to-heading ratio
    if text_blocks:
        ratio = len(text_blocks) / len(heading_blocks)
        if 2 <= ratio <= 15:  # Good document structure
            quality_components.append(0.3)
        elif 1 <= ratio <= 20:  # Acceptable structure
            quality_components.append(0.2)
        else:
            quality_components.append(0.1)
    
    return min(sum(quality_components), 1.0)

def _assess_block_consistency(blocks: List[Dict[str, Any]]) -> float:
    """
    Assess consistency of block structure and metadata.
    
    Args:
        blocks: List of blocks to assess
        
    Returns:
        Consistency score (0.0 to 1.0)
    """
    if not blocks:
        return 0.0
    
    # Check for consistent metadata structure
    blocks_with_type = sum(1 for block in blocks if block.get('type'))
    blocks_with_source = sum(1 for block in blocks if block.get('source'))
    blocks_with_text = sum(1 for block in blocks if block.get('text', '').strip())
    
    type_consistency = blocks_with_type / len(blocks)
    source_consistency = blocks_with_source / len(blocks)
    text_consistency = blocks_with_text / len(blocks)
    
    # Average consistency across metadata fields
    consistency_score = (type_consistency + source_consistency + text_consistency) / 3
    
    return consistency_score

def get_pymupdf4llm_info() -> Dict[str, Any]:
    """
    Get information about the PyMuPDF4LLM installation.
    
    Returns:
        Dictionary with installation and capability information
    """
    if not is_pymupdf4llm_available():
        return {
            'available': False,
            'error': 'PyMuPDF4LLM not installed or not importable'
        }
    
    try:
        info = {
            'available': True,
            'version': getattr(pymupdf4llm, '__version__', 'Unknown'),
            'module_file': getattr(pymupdf4llm, '__file__', 'Unknown'),
            'available_attributes': [attr for attr in dir(pymupdf4llm) if not attr.startswith('_')]
        }
        
        # Test basic functionality
        try:
            # Try to identify working extraction methods
            working_methods = []
            test_methods = ['to_markdown', 'extract', 'convert', 'parse']
            
            for method in test_methods:
                if hasattr(pymupdf4llm, method):
                    working_methods.append(method)
            
            info['working_methods'] = working_methods
            info['functional'] = len(working_methods) > 0
            
        except Exception as e:
            info['functional'] = False
            info['test_error'] = str(e)
        
        return info
        
    except Exception as e:
        return {
            'available': True,
            'error': f'Error getting PyMuPDF4LLM info: {str(e)}'
        }
