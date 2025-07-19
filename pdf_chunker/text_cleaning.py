# text_cleaning.py

import re
from functools import reduce

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

    # Unicode hyphens to ASCII hyphen
    '\u00AD': '-',   # ­ (U+00AD) Soft hyphen
    '\u2010': '-',     # ‐ (U+2010) Hyphen
    '\u2011': '-',     # ‑ (U+2011) Non-breaking hyphen
    '\u2012': '-',     # ‒ (U+2012) Figure dash
    '\u2013': '-',     # – (U+2013) En dash
    '\u2014': '-',     # — (U+2014) Em dash
}


def normalize_ligatures(text: str) -> str:
    return reduce(lambda acc, kv: acc.replace(kv[0], kv[1]),
                  LIGATURE_MAP.items(),
                  text)




#def normalize_ligatures(text: str) -> str:
#    """
#    Normalize Unicode ligatures to their ASCII equivalents.
#    
#    Args:
#        text: Input text that may contain Unicode ligatures
#        
#    Returns:
#        Text with ligatures replaced by ASCII equivalents
#    """
#    if not text:
#        return text
#    
#    # Apply ligature replacements
#    normalized_text = text
#    for ligature, replacement in LIGATURE_MAP.items():
#        normalized_text = normalized_text.replace(ligature, replacement)
#    
#    return normalized_text


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


def clean_text(text: str) -> str:
    """
    Clean a text block by paragraph, preserving meaningful breaks and
    cleaning residual continuations.
    """
    if not text or not text.strip():
        return ''

    paragraphs = (clean_paragraph(p) for p in text.split('\n\n'))
    cleaned = '\n\n'.join(p for p in paragraphs if p)
    return cleanup_residual_continuations(cleaned)

# Alias original underscored names for backward compatibility
_normalize_ligatures = normalize_ligatures
_remove_special_chars = remove_special_chars
_fix_hyphenated_breaks = fix_hyphenated_breaks
_consolidate_whitespace = consolidate_whitespace
_cleanup_residual_continuations = cleanup_residual_continuations
_clean_paragraph = clean_paragraph
_clean_text = clean_text
