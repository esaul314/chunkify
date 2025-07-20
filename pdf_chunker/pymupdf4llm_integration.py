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
    Extract text from PDF using PyMuPDF4LLM with automatic heading detection.
    
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
        
        # Determine pages to process
        pages_to_process = None
        if excluded_page_numbers:
            # PyMuPDF4LLM expects a list of page numbers to include, not exclude
            # We'll need to determine the total pages first, then create inclusion list
            # For now, we'll extract all pages and filter afterward
            logger.info(f"Page exclusion requested: {excluded_page_numbers}")
        
        # Extract using PyMuPDF4LLM
        logger.debug(f"Starting PyMuPDF4LLM extraction for: {pdf_path}")
        
        # Try different API patterns for PyMuPDF4LLM
        markdown_text = _call_pymupdf4llm_api(pdf_path, pages_to_process)
        
        if not markdown_text or not markdown_text.strip():
            raise PyMuPDF4LLMExtractionError("PyMuPDF4LLM returned empty text")
        
        # Convert Markdown to structured blocks
        blocks = _convert_markdown_to_blocks(markdown_text, excluded_page_numbers)
        
        # Generate extraction metadata
        extraction_time = time.time() - start_time
        metadata = {
            'extraction_method': 'pymupdf4llm',
            'extraction_time': extraction_time,
            'markdown_length': len(markdown_text),
            'total_blocks': len(blocks),
            'excluded_pages': list(excluded_page_numbers) if excluded_page_numbers else [],
            'has_headings': any(block.get('metadata', {}).get('is_heading', False) for block in blocks)
        }
        
        logger.info(f"PyMuPDF4LLM extraction completed: {len(blocks)} blocks in {extraction_time:.2f}s")
        
        return blocks, metadata
        
    except Exception as e:
        extraction_time = time.time() - start_time
        error_msg = f"PyMuPDF4LLM extraction failed after {extraction_time:.2f}s: {str(e)}"
        logger.error(error_msg)
        raise PyMuPDF4LLMExtractionError(error_msg) from e


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
    # Try different API patterns
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
    Assess the quality of PyMuPDF4LLM extraction results.
    
    Args:
        blocks: List of extracted text blocks
        metadata: Extraction metadata
        
    Returns:
        Quality assessment metrics
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
            'issues': ['No content extracted']
        }
    
    # Calculate basic metrics
    total_text_length = sum(len(block.get('text', '')) for block in blocks)
    heading_blocks = [b for b in blocks if b.get('metadata', {}).get('is_heading', False)]
    text_blocks = [b for b in blocks if not b.get('metadata', {}).get('is_heading', False)]
    
    avg_block_length = total_text_length / len(blocks) if blocks else 0
    
    # Assess quality factors
    issues = []
    quality_factors = []
    
    # Content presence
    if total_text_length > 0:
        quality_factors.append(0.3)  # Base score for having content
    else:
        issues.append('No text content')
    
    # Heading detection
    if heading_blocks:
        quality_factors.append(0.2)  # Bonus for heading detection
    else:
        issues.append('No headings detected')
    
    # Block structure
    if len(blocks) > 1:
        quality_factors.append(0.2)  # Bonus for multiple blocks
    
    # Average block length (prefer reasonable-sized blocks)
    if 100 <= avg_block_length <= 2000:
        quality_factors.append(0.2)  # Bonus for reasonable block sizes
    elif avg_block_length < 50:
        issues.append('Blocks too short')
    elif avg_block_length > 5000:
        issues.append('Blocks too long')
    
    # Text-to-heading ratio
    if text_blocks and heading_blocks:
        ratio = len(text_blocks) / len(heading_blocks)
        if 2 <= ratio <= 10:
            quality_factors.append(0.1)  # Bonus for good structure
    
    # Calculate overall quality score
    quality_score = sum(quality_factors)
    
    return {
        'quality_score': min(quality_score, 1.0),
        'has_content': total_text_length > 0,
        'has_headings': len(heading_blocks) > 0,
        'avg_block_length': avg_block_length,
        'total_text_length': total_text_length,
        'block_count': len(blocks),
        'heading_count': len(heading_blocks),
        'text_block_count': len(text_blocks),
        'extraction_time': metadata.get('extraction_time', 0),
        'issues': issues
    }


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
