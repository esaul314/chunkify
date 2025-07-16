import re

# Unicode ligature to ASCII mapping for common typographic ligatures
LIGATURE_MAP = {
    # Latin ligatures
    '\ufb01': 'fi',    # ﬁ (U+FB01)
    '\ufb02': 'fl',    # ﬂ (U+FB02)
    '\ufb03': 'ffi',   # ﬃ (U+FB03)
    '\ufb04': 'ffl',   # ﬄ (U+FB04)
    '\ufb00': 'ff',    # ﬀ (U+FB00)
    '\ufb05': 'ft',    # ﬅ (U+FB05)
    '\ufb06': 'st',    # ﬆ (U+FB06)

    # Additional common ligatures
    '\u0152': 'OE',    # Œ (U+0152)
    '\u0153': 'oe',    # œ (U+0153)
    '\u00c6': 'AE',    # Æ (U+00C6)
    '\u00e6': 'ae',    # æ (U+00E6)
    '\u0132': 'IJ',    # Ĳ (U+0132)
    '\u0133': 'ij',    # ĳ (U+0133)

    # German eszett (though not technically a ligature, often needs normalization)
    '\u00df': 'ss',    # ß (U+00DF)
}

def _normalize_ligatures(text: str) -> str:
    """
    Normalize Unicode ligatures to their ASCII equivalents.
    
    Args:
        text: Input text that may contain Unicode ligatures
        
    Returns:
        Text with ligatures replaced by ASCII equivalents
    """
    if not text:
        return text

    # Apply ligature replacements
    normalized_text = text
    for ligature, replacement in LIGATURE_MAP.items():
        normalized_text = normalized_text.replace(ligature, replacement)

    return normalized_text

def _clean_paragraph(paragraph: str) -> str:
    """
    Replaces all whitespace characters with a single space and removes the BOM character.
    Also fixes hyphenated word breaks from PDF line wrapping and normalizes Unicode ligatures.
    """
    # Remove the BOM character (U+FEFF) which can appear in source files

    cleaned_text = paragraph.replace('\ufeff', '').replace('\u200b', '').replace('\u008b', '').replace('\u0089', '').replace('\u0097', '')
    

    # Normalize Unicode ligatures to ASCII equivalents
    cleaned_text = _normalize_ligatures(cleaned_text)


    # Fix hyphenated word breaks (e.g., "itera-tion" -> "iteration")
    # Enhanced pattern to handle hyphens followed by various newline combinations
    cleaned_text = re.sub(r'(\w)-\n+', r'\1', cleaned_text)
    cleaned_text = re.sub(r'(\w)-\s*\n+\s*', r'\1', cleaned_text)
    

    # Consolidate all other whitespace into single spaces
    return re.sub(r'\s+', ' ', cleaned_text).strip()

def _clean_text(text: str) -> str:
    """
    Cleans a block of text by preserving paragraph breaks and cleaning each paragraph.
    This function is designed to be pure and declarative.
    """
    if not text or not text.strip():
        return ""

    # Split by paragraph, clean each one, filter out empty ones, and rejoin.

    paragraphs = text.split('\n\n')
    cleaned_paragraphs = (_clean_paragraph(p) for p in paragraphs)
    cleaned_text = '\n\n'.join(p for p in cleaned_paragraphs if p)
    # Post-processing cleanup for residual continuation patterns
    return _cleanup_residual_continuations(cleaned_text)
    


def _cleanup_residual_continuations(text: str) -> str:
    """
    Clean up residual patterns like 'word\n\nword' that likely represent
    continuations from line breaks rather than actual paragraph breaks.

    This is a conservative cleanup that only merges when the pattern
    strongly suggests a continuation.

    Args:
        text: Input text that may contain residual continuation patterns

    Returns:
        Text with likely continuation patterns cleaned up
    """
    if not text:
        return text

    # Pattern: lowercase word + \n\n + lowercase word (likely continuation)
    # Only merge if both sides are lowercase words (conservative approach)
    cleaned_text = re.sub(r'([a-z]+)\n\n([a-z]+)', r'\1\2', text)

    # Pattern: word ending + \n\n + word beginning (more conservative)
    # Only merge if the second part starts with lowercase and is short
    cleaned_text = re.sub(r'([a-zA-Z]+)\n\n([a-z]{1,8}(?:\s|$))', r'\1\2', cleaned_text)

    return cleaned_text
    
