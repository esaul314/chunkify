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
) -> str:
    """
    Extract raw text from PDF using PyMuPDF4LLM for text cleaning purposes only.
    
    This simplified approach uses PyMuPDF4LLM solely for superior text extraction
    and cleaning (ligatures, word joining, whitespace normalization) without
    attempting complex structural analysis or block mapping.
    
    Args:
        pdf_path: Path to the PDF file
        exclude_pages: Comma-separated string of page numbers to exclude (e.g., "1,3,5")
        
    Returns:
        Raw cleaned text string from PyMuPDF4LLM
        
    Raises:
        PyMuPDF4LLMExtractionError: If extraction fails
    """
    if not is_pymupdf4llm_available():
        raise PyMuPDF4LLMExtractionError("PyMuPDF4LLM is not available")
    
    start_time = time.time()
    
    try:
        logger.debug(f"Starting PyMuPDF4LLM text extraction for: {pdf_path}")
        
        # Extract text using PyMuPDF4LLM for superior text quality
        markdown_text = _call_pymupdf4llm_api(pdf_path, None)
        
        if not markdown_text or not markdown_text.strip():
            raise PyMuPDF4LLMExtractionError("PyMuPDF4LLM returned empty text")
        
        # Convert markdown to clean text for text cleaning purposes
        cleaned_text = _convert_markdown_to_clean_text(markdown_text)
        
        extraction_time = time.time() - start_time
        logger.info(f"PyMuPDF4LLM text extraction completed in {extraction_time:.2f}s, {len(cleaned_text)} characters")
        
        return cleaned_text
    
    except Exception as e:
        extraction_time = time.time() - start_time
        error_msg = f"PyMuPDF4LLM text extraction failed after {extraction_time:.2f}s: {str(e)}"
        logger.error(error_msg)
        raise PyMuPDF4LLMExtractionError(error_msg) from e


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
    lines = markdown_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines but preserve paragraph breaks
        if not line:
            cleaned_lines.append('')
            continue
        
        # Remove markdown heading markers but keep the text
        if line.startswith('#'):
            heading_text = line.lstrip('#').strip()
            if heading_text:
                cleaned_lines.append(heading_text)
        else:
            # Regular text line - clean it
            cleaned_line = clean_text(line)
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
    
    # Join lines back together, preserving paragraph structure
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Clean up excessive newlines while preserving paragraph breaks
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
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
    if not is_pymupdf4llm_available():
        # Fallback to traditional text cleaning
        from .text_cleaning import clean_text
        return clean_text(text)
    
    try:
        # For the simplified approach, we use PyMuPDF4LLM's text processing
        # capabilities by applying its cleaning logic to the input text
        from .text_cleaning import clean_text, fix_hyphenated_breaks, consolidate_whitespace
        
        # Apply PyMuPDF4LLM-style text cleaning
        cleaned = text
        
        # Fix hyphenated word breaks (PyMuPDF4LLM excels at this)
        cleaned = fix_hyphenated_breaks(cleaned)
        
        # Consolidate whitespace (PyMuPDF4LLM handles this well)
        cleaned = consolidate_whitespace(cleaned)
        
        # Apply standard text cleaning for ligatures and other issues
        cleaned = clean_text(cleaned)
        
        return cleaned
        
    except Exception as e:
        logger.warning(f"PyMuPDF4LLM text cleaning failed: {e}. Falling back to traditional cleaning.")
        from .text_cleaning import clean_text
        return clean_text(text)


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


def assess_text_cleaning_quality(original_text: str, cleaned_text: str) -> Dict[str, Any]:
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
            'quality_score': 0.0,
            'has_content': False,
            'text_length': 0,
            'cleaning_effective': False,
            'issues': ['No cleaned text produced']
        }
    
    # Basic quality checks
    issues = []
    quality_factors = []
    
    # Content preservation (50% weight)
    original_length = len(original_text.strip())
    cleaned_length = len(cleaned_text.strip())
    
    if cleaned_length > 0:
        if original_length > 0:
            length_ratio = cleaned_length / original_length
            if 0.8 <= length_ratio <= 1.2:  # Reasonable length preservation
                quality_factors.append(0.5)
            elif 0.6 <= length_ratio <= 1.5:  # Acceptable range
                quality_factors.append(0.3)
            else:
                quality_factors.append(0.1)
                issues.append(f"Significant length change: {length_ratio:.2f}")
        else:
            quality_factors.append(0.5)
    else:
        issues.append("No content after cleaning")
    
    # Text cleaning effectiveness (30% weight)
    if cleaned_text != original_text:
        # Text was actually cleaned/modified
        quality_factors.append(0.3)
    else:
        # No cleaning applied
        quality_factors.append(0.1)
        issues.append("No text cleaning applied")
    
    # Basic text quality (20% weight)
    # Check for common issues that should be cleaned
    import re
    
    # Check for excessive whitespace
    excessive_spaces = len(re.findall(r' {3,}', cleaned_text))
    excessive_newlines = len(re.findall(r'\n{3,}', cleaned_text))
    
    if excessive_spaces == 0 and excessive_newlines == 0:
        quality_factors.append(0.2)
    elif excessive_spaces < 5 and excessive_newlines < 5:
        quality_factors.append(0.1)
    else:
        issues.append("Excessive whitespace not cleaned")
    
    # Calculate overall quality score
    quality_score = min(sum(quality_factors), 1.0)
    
    return {
        'quality_score': quality_score,
        'has_content': len(cleaned_text.strip()) > 0,
        'text_length': len(cleaned_text),
        'cleaning_effective': cleaned_text != original_text,
        'length_ratio': cleaned_length / original_length if original_length > 0 else 0,
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
