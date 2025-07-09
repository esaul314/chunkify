import re

def _clean_paragraph(paragraph: str) -> str:
    """
    Replaces all whitespace characters with a single space and removes the BOM character.
    Also fixes hyphenated word breaks from PDF line wrapping.
    """
    # Remove the BOM character (U+FEFF) which can appear in source files
    cleaned_text = paragraph.replace('\ufeff', '').replace('\u200b', '')
    
    # Fix hyphenated word breaks (e.g., "itera-tion" -> "iteration")
    # Pattern matches: word character + hyphen + whitespace
    cleaned_text = re.sub(r'(\w)-\s+', r'\1', cleaned_text)
    
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
    return '\n\n'.join(p for p in cleaned_paragraphs if p)
