import re

def _detect_heading_fallback(text: str) -> bool:
    """
    Fallback heading detection using text characteristics when font analysis fails.
    Uses heuristics like text length, capitalization patterns, and punctuation.
    """
    if not text or not text.strip():
        return False
    
    text = text.strip()
    words = text.split()
    
    # Very short text (1-3 words) is likely a heading
    if len(words) <= 3:
        return True
    
    # Text that's all uppercase might be a heading
    if text.isupper() and len(words) <= 8:
        return True
    
    # Title case text without ending punctuation might be a heading
    if (text.istitle() and 
        len(words) <= 10 and 
        not text.endswith(('.', '!', '?', ';', ':'))):
        return True
    
    # Text that starts with common heading patterns
    heading_starters = ['chapter', 'section', 'part', 'appendix', 'introduction', 'conclusion']
    first_word = words[0].lower() if words else ''
    if first_word in heading_starters and len(words) <= 8:
        return True
    
    # Text that's mostly numbers (like "1.2.3 Some Topic")
    if len(words) >= 2:
        first_part = words[0]
        if re.match(r'^[\d\.\-]+$', first_part) and len(words) <= 8:
            return True
    
    return False
