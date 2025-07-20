# text_cleaning.py

import re
from functools import reduce
import ftfy


def normalize_ligatures(text: str) -> str:
    """
    Normalize Unicode ligatures and other text issues using ftfy.
    
    Args:
        text: Input text that may contain Unicode ligatures and encoding issues
        
    Returns:
        Text with ligatures and encoding issues fixed
    """
    if not text:
        return text
    
    # Use ftfy to fix Unicode issues including ligatures
    return ftfy.fix_text(text)


def remove_special_chars(text: str) -> str:
    """Remove BOM and zero-width/special characters."""
    return re.sub(r'[\ufeff\u200b\u008b\u0089\u0097\u0002\u0004]', '', text)


def fix_hyphenated_breaks(text: str) -> str:
    """
    Merge hyphenated splits into single words, whether the break
    was newline-based or space-based (e.g. 'fea-\ntures' or 'fea- tures').
    """
    patterns = [
        # hyphen + optional whitespace & newline + lowercase → merge
        (r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2"),
        # hyphen + space(s) + lowercase → merge
        (r"([A-Za-z])-\s+([a-z])",  r"\1\2"),
    ]
    return reduce(lambda acc, pr: re.sub(pr[0], pr[1], acc),
                  patterns,
                  text)


#def fix_hyphenated_breaks(text: str) -> str:
#    """
#    Merge hyphenated splits across line breaks into single words.
#    E.g., 'fea-\ntures' or 'fea-  \ntures' -> 'features'
#    """
#    # Pattern: letter, '-', optional whitespace/newlines, lowercase letter
#    return re.sub(r"([A-Za-z])\-\s*\n\s*([a-z])", r"\1\2", text)


#def fix_hyphenated_breaks(text: str) -> str:
#    """Fix hyphenated word breaks from PDF line wrapping."""
#    text = re.sub(r'(\w)-\n+', r'\1', text)
#    text = re.sub(r'(\w)-\s*\n+\s*', r'\1', text)
#    # Fix hyphen + space + lowercase continuation (e.g., "charac- teristics" -> "characteristics")
#    text = re.sub(r'(\w)-\s+([a-z])', r'\1\2', text)
#    return text


def consolidate_whitespace(text: str) -> str:
    """Consolidate all whitespace into single spaces and trim."""
    return re.sub(r'\s+', ' ', text).strip()


def cleanup_residual_continuations(text: str) -> str:
    """Merge residual paragraph breaks that likely represent continuations."""
    return re.sub(r'([a-zA-Z]+)\n\n([a-z]+)', r'\1\2', text)


def clean_paragraph(paragraph: str) -> str:
    """
    Clean a single paragraph: remove special chars, normalize ligatures,
    fix hyphen breaks, and consolidate whitespace.
    """
    transformations = [
        remove_special_chars,
        normalize_ligatures,
        fix_hyphenated_breaks,
        consolidate_whitespace,
    ]
    return reduce(lambda txt, fn: fn(txt), transformations, paragraph)


def clean_text(text: str, use_pymupdf4llm: bool = True) -> str:
    """
    Clean a text block by paragraph, preserving meaningful breaks and
    cleaning residual continuations. Optionally uses PyMuPDF4LLM for
    enhanced text normalization.
    
    Args:
        text: Text to clean
        use_pymupdf4llm: Whether to use PyMuPDF4LLM for enhanced cleaning
        
    Returns:
        Cleaned text with improved formatting
    """
    if not text or not text.strip():
        return ''

    # Try PyMuPDF4LLM enhanced cleaning if available and requested
    if use_pymupdf4llm:
        try:
            from .pymupdf4llm_integration import is_pymupdf4llm_available, clean_text_with_pymupdf4llm
            
            if is_pymupdf4llm_available():
                return clean_text_with_pymupdf4llm(text)
        except ImportError:
            # PyMuPDF4LLM integration not available, fall back to traditional cleaning
            pass
        except Exception:
            # PyMuPDF4LLM cleaning failed, fall back to traditional cleaning
            pass
    
    # Traditional text cleaning approach
    paragraphs = (clean_paragraph(p) for p in text.split('\n\n'))
    cleaned = '\n\n'.join(p for p in paragraphs if p)
    return cleanup_residual_continuations(cleaned)


def clean_text_traditional(text: str) -> str:
    """
    Clean text using only traditional methods without PyMuPDF4LLM.
    
    This function provides a way to explicitly use traditional text cleaning
    without attempting PyMuPDF4LLM integration, useful for fallback scenarios
    or when PyMuPDF4LLM-specific behavior is not desired.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text using traditional methods only
    """
    return clean_text(text, use_pymupdf4llm=False)


# Alias original underscored names for backward compatibility
_normalize_ligatures = normalize_ligatures
_remove_special_chars = remove_special_chars
_fix_hyphenated_breaks = fix_hyphenated_breaks
_consolidate_whitespace = consolidate_whitespace
_cleanup_residual_continuations = cleanup_residual_continuations
_clean_paragraph = clean_paragraph
_clean_text = clean_text
