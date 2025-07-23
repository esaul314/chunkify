# text_cleaning.py

import re
import logging
from typing import List, Tuple, Set
from functools import reduce
import ftfy

logger = logging.getLogger(__name__)


def normalize_ligatures(text: str) -> str:
    """
    Normalize Unicode ligatures and other text issues using ftfy.
    
    Args:
        text: Input text that may contain Unicode ligatures and encoding issues
        
    Returns:
        Text with ligatures and encoding issues fixed
    """
    # Use ftfy to fix Unicode issues including ligatures
    return ftfy.fix_text(text)


def normalize_quotes(text: str) -> str:
    """Normalize quotes to standard ASCII quotes and fix common issues."""
    if not text:
        return text
    
    import re
    
    double_quote_count = text.count('"')
    single_quote_count = text.count("'")
    logger.debug(f"Quote normalization started: {double_quote_count + single_quote_count} quote characters")
    
    # Step 1: Convert smart quotes to standard ASCII quotes
    # Double quotes
    text = text.replace('“', '"')  # Left double quotation mark
    text = text.replace('”', '"')  # Right double quotation mark
    text = text.replace('„', '"')  # Double low-9 quotation mark
    text = text.replace('‚', '"')  # Single low-9 quotation mark (sometimes used as double)
    
    # Single quotes / apostrophes
    text = text.replace('‘', "'")  # Left single quotation mark
    text = text.replace('’', "'")  # Right single quotation mark
    text = text.replace('`', "'")  # Grave accent (sometimes used as quote)
    
    # Step 2: Fix spacing around quotes
    # Add space before opening quotes when needed
    text = re.sub(r'(\w)"([A-Z])', r'\1 "\2', text)
    
    # Add space after closing quotes when needed
    text = re.sub(r'"(\w)', r'" \1', text)
    
    # Step 3: Fix common quote sequence issues
    # Multiple quotes in a row
    text = re.sub(r'"{2,}', '"', text)  # Multiple double quotes
    text = re.sub(r"'{2,}", "'", text)  # Multiple single quotes
    
    # Step 4: Handle quotes at word boundaries more carefully
    # Fix quotes that got separated from words
    text = re.sub(r'\s+"([^"]*?)"\s+', r' "\1" ', text)
    
    # Step 5: Ensure consistent quote style within the same text
    # This is a simple heuristic - could be enhanced
    double_quote_count = text.count('"')
    single_quote_count = text.count("'")
    
    if double_quote_count > 0 and single_quote_count > 0:
        logger.debug(f"Mixed quote styles detected: {double_quote_count} double, {single_quote_count} single")
    

    logger.debug(f"Quote normalization complete: {double_quote_count + single_quote_count} quote characters")
    
    
    return text


def validate_json_safety(text: str) -> Tuple[bool, List[str]]:
    """Validate if text can be safely serialized to JSON and return issues found."""
    if not text:
        return True, []
    
    import json
    import re
    
    issues = []
    
    # Test 1: Basic JSON serialization
    try:
        json.dumps({"test": text}, ensure_ascii=False)
    except (json.JSONEncodeError, UnicodeEncodeError) as e:
        issues.append(f"JSON serialization failed: {str(e)}")
    
    # Test 2: Check for problematic characters
    control_chars = re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text)
    if control_chars:
        issues.append(f"Control characters found: {len(control_chars)} instances")
    
    # Test 3: Check for unbalanced quotes
    double_quotes = text.count('"')
    if double_quotes % 2 != 0:
        issues.append(f"Unbalanced double quotes: {double_quotes} total")
    
    # Test 4: Check for problematic quote patterns
    if re.search(r'^[",]', text.strip()):
        issues.append("Text starts with problematic punctuation")
    
    if re.search(r'[",]$', text.strip()):
        issues.append("Text ends with problematic punctuation")
    
    # Test 5: Check for encoding issues
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeError:
        issues.append("Unicode encoding issues detected")
    
    is_safe = len(issues) == 0
    return is_safe, issues


def clean_text(text: str, preserve_structure: bool = True) -> str:
    """Clean and normalize text extracted from PDF."""
    if not text:
        return text
    
    original_length = len(text)
    logger.debug(f"Text cleaning started: {original_length} characters")
    logger.debug(f"Input preview: '{text[:100].replace(chr(10), '\\n')}'...")
    
    # Apply cleaning steps
    cleaned = text
    
    # Fix hyphenated line breaks
    before_hyphen = len(cleaned)
    cleaned = fix_hyphenated_breaks(cleaned)
    after_hyphen = len(cleaned)
    if before_hyphen != after_hyphen:
        logger.debug(f"Hyphen fixing: {before_hyphen} → {after_hyphen} chars")
    
    # Fix word gluing issues
    before_gluing = len(cleaned)
    cleaned = detect_and_fix_word_gluing(cleaned)
    after_gluing = len(cleaned)
    if before_gluing != after_gluing:
        logger.debug(f"Word gluing fixes: {before_gluing} → {after_gluing} chars")
    
    # Normalize quotes
    before_quotes = len(cleaned)
    cleaned = normalize_quotes(cleaned)
    after_quotes = len(cleaned)
    if before_quotes != after_quotes:
        logger.debug(f"Quote normalization: {before_quotes} → {after_quotes} chars")
    
    # Consolidate whitespace
    before_whitespace = len(cleaned)
    cleaned = consolidate_whitespace(cleaned, preserve_structure)
    after_whitespace = len(cleaned)
    if before_whitespace != after_whitespace:
        logger.debug(f"Whitespace consolidation: {before_whitespace} → {after_whitespace} chars")
    
    # Validate JSON safety and apply fixes if needed
    is_safe, issues = validate_json_safety(cleaned)
    if not is_safe:
        logger.warning(f"JSON safety issues detected: {issues}")
        # Apply additional cleaning for JSON safety
        before_json_fixes = len(cleaned)
        cleaned = _apply_json_safety_fixes(cleaned)
        after_json_fixes = len(cleaned)
        if before_json_fixes != after_json_fixes:
            logger.debug(f"JSON safety fixes: {before_json_fixes} → {after_json_fixes} chars")
        
        # Re-validate after fixes
        is_safe_after, remaining_issues = validate_json_safety(cleaned)
        if not is_safe_after:
            logger.warning(f"JSON safety issues remain after fixes: {remaining_issues}")
    
    final_length = len(cleaned)
    logger.debug(f"Text cleaning complete: {original_length} → {final_length} characters")
    logger.debug(f"Output preview: '{cleaned[:100].replace(chr(10), '\\n')}'...")
    
    return cleaned
    # Use ftfy to fix Unicode issues including ligatures
    return ftfy.fix_text(text)


def _apply_json_safety_fixes(text: str) -> str:
    """Apply fixes to make text safe for JSON serialization."""
    if not text:
        return text
    
    import re
    
    fixed = text
    
    # Fix 1: Remove control characters
    fixed = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', fixed)
    
    # Fix 2: Handle problematic quote patterns
    # Remove quotes at the very beginning if they look like fragments
    if fixed.startswith('", '):
        fixed = fixed[3:]
    elif fixed.startswith('"'):
        # Check if this looks like a fragment
        if len(fixed) > 1 and fixed[1].islower():
            fixed = fixed[1:]
    
    # Fix 3: Handle problematic endings
    if fixed.endswith(', "') or fixed.endswith(',"'):
        fixed = fixed[:-2]
    
    # Fix 4: Ensure proper Unicode handling
    try:
        fixed = fixed.encode('utf-8', errors='replace').decode('utf-8')
    except UnicodeError:
        # Fallback: remove non-ASCII characters
        fixed = ''.join(char for char in fixed if ord(char) < 128)
    
    # Fix 5: Final quote balancing attempt
    double_quotes = fixed.count('"')
    if double_quotes % 2 != 0:
        # Simple fix: remove the last quote if it's at the end
        if fixed.endswith('"'):
            fixed = fixed[:-1]
        elif fixed.startswith('"'):
            fixed = fixed[1:]
    
    return fixed


def remove_special_chars(text: str) -> str:
    """Remove BOM and zero-width/special characters."""
    return re.sub(r'[\ufeff\u200b\u008b\u0089\u0097\u0002\u0004]', '', text)


def fix_hyphenated_breaks(text: str) -> str:
    """Fix hyphenated line breaks in text."""
    if not text:
        return text
    
    import re
    
    # Pattern for hyphenated line breaks: word- followed by newline and lowercase word
    hyphen_pattern = r'(\w)-\s*\n\s*(\w)'
    
    matches = list(re.finditer(hyphen_pattern, text))
    if matches:
        logger.debug(f"Found {len(matches)} hyphenated line breaks to fix")
        for i, match in enumerate(matches[:3]):  # Log first 3 matches
            before = match.group(0).replace('\n', '\\n')
            after = match.group(1) + match.group(2)
            logger.debug(f"  Hyphen fix {i+1}: '{before}' → '{after}'")
    
    # Replace hyphenated line breaks
    fixed_text = re.sub(hyphen_pattern, r'\1\2', text)
    
    return fixed_text


def remove_special_chars(text: str) -> str:
    """Remove BOM and zero-width/special characters."""
    return re.sub(r'[\ufeff\u200b\u008b\u0089\u0097\u0002\u0004]', '', text)


def fix_hyphenated_breaks(text: str) -> str:
    """
    Merge hyphenated splits into single words, whether the break
    was newline-based or space-based (e.g. 'fea-
    tures' or 'fea- tures').
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
#    E.g., 'fea-
#tures' or 'fea-  \
#tures' -> 'features'
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


def clean_text(text: str, use_pymupdf4llm: bool = False) -> str:
    """
    Clean a text block by paragraph, preserving meaningful breaks and
    cleaning residual continuations. Optionally uses PyMuPDF4LLM for
    enhanced text normalization.
    
    Args:
        text: Text to clean
        use_pymupdf4llm: Whether to use PyMuPDF4LLM for enhanced cleaning.
                        Defaults to False for safer behavior. Can be overridden
                        by setting PDF_CHUNKER_USE_PYMUPDF4LLM environment variable.
        
    Returns:
        Cleaned text with improved formatting
    """
    import os
    
    if not text or not text.strip():
        return ''
    
    # Check environment variable to override default behavior
    env_use_pymupdf4llm = os.getenv('PDF_CHUNKER_USE_PYMUPDF4LLM', '').lower()
    if env_use_pymupdf4llm in ('true', '1', 'yes', 'on'):
        use_pymupdf4llm = True
    elif env_use_pymupdf4llm in ('false', '0', 'no', 'off'):
        use_pymupdf4llm = False
    # If environment variable is not set or invalid, use the parameter value

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


def detect_and_fix_word_gluing(text: str) -> str:
    """Detect and fix word gluing issues using multiple heuristics."""
    if not text:
        return text
    
    logger.debug(f"Word gluing detection started: {len(text)} characters")
    original_text = text
    
    # Apply multiple detection and repair strategies
    text = _fix_case_transition_gluing(text)
    text = _fix_page_boundary_gluing(text)
    text = _fix_line_break_gluing(text)
    text = _fix_quote_boundary_gluing(text)
    
    if text != original_text:
        changes = len(original_text) - len(text)
        logger.debug(f"Word gluing fixes applied: {changes} character change")
        logger.debug(f"Sample fix: '{original_text[:100]}' → '{text[:100]}'")
    
    return text


def _fix_case_transition_gluing(text: str) -> str:
    """Fix word gluing at lowercase-to-uppercase transitions."""
    # Pattern: lowercase letter followed immediately by uppercase letter
    # But exclude legitimate cases like "iPhone", "McDonald", etc.
    
    # Common legitimate patterns to preserve
    legitimate_patterns = {
        r'\b[a-z]+[A-Z][a-z]*\b',  # camelCase words
        r'\b[A-Z][a-z]*[A-Z][a-z]*\b',  # PascalCase words
        r'\b[a-z]*[A-Z]{2,}\b',  # Words with acronyms like "iPhone"
    }
    
    # Find potential gluing: lowercase followed by uppercase
    gluing_pattern = r'([a-z])([A-Z][a-z])'
    
    def should_split(match):
        """Determine if a case transition should be split."""
        full_match = match.group(0)
        before_char = match.group(1)
        after_chars = match.group(2)
        
        # Get surrounding context
        start_pos = match.start()
        end_pos = match.end()
        
        # Extend to find word boundaries
        word_start = start_pos
        word_end = end_pos
        
        while word_start > 0 and text[word_start - 1].isalnum():
            word_start -= 1
        while word_end < len(text) and text[word_end].isalnum():
            word_end += 1
        
        full_word = text[word_start:word_end]
        
        # Check against legitimate patterns
        for pattern in legitimate_patterns:
            if re.match(pattern, full_word):
                return False
        
        # Check if this looks like two separate words
        before_word = text[word_start:start_pos + 1]
        after_word = text[start_pos + 1:word_end]
        
        # Simple heuristic: if both parts are reasonable length and 
        # the transition doesn't look like a compound word
        if (len(before_word) >= 2 and len(after_word) >= 2 and
            not _is_likely_compound_word(before_word, after_word)):
            return True
        
        return False
    
    def replace_gluing(match):
        if should_split(match):
            return match.group(1) + ' ' + match.group(2)
        return match.group(0)
    
    fixed_text = re.sub(gluing_pattern, replace_gluing, text)
    
    if fixed_text != text:
        logger.debug(f"Case transition fixes: {len(re.findall(gluing_pattern, text))} potential, applied fixes")
    
    return fixed_text


def _fix_page_boundary_gluing(text: str) -> str:
    """Fix word gluing that occurs at page boundaries."""
    # Look for patterns that suggest page boundary issues:
    # 1. Word ending followed immediately by capitalized word (new sentence/paragraph)
    # 2. Incomplete words at what might be page breaks
    
    # Pattern: word character followed by uppercase letter that starts a new sentence
    page_boundary_pattern = r'([a-z])([A-Z][a-z]+(?:\s+[A-Z]|[.!?]))'
    
    def fix_page_boundary(match):
        before = match.group(1)
        after = match.group(2)
        
        # Check if this looks like a page boundary issue
        # Heuristic: if the "after" part starts what looks like a new sentence
        if (after[0].isupper() and 
            (len(after) > 3 or after.endswith(('.', '!', '?')))):
            return before + ' ' + after
        
        return match.group(0)
    
    fixed_text = re.sub(page_boundary_pattern, fix_page_boundary, text)
    
    return fixed_text


def _fix_line_break_gluing(text: str) -> str:
    """Fix word gluing that occurs at line breaks, especially in indented quotes."""
    # Pattern: word followed immediately by another word, where the break
    # might have occurred at a line boundary
    
    # Look for missing spaces in quoted text or indented content
    # This is more conservative - only fix obvious cases
    
    # Pattern: letter followed by letter with no space, but not at start of word
    line_break_pattern = r'([a-z])([a-z][a-z]+)'
    
    def fix_line_break(match):
        before = match.group(1)
        after = match.group(2)
        combined = before + after
        
        # Only fix if the combined word seems too long or unusual
        if (len(combined) > 12 and  # Unusually long word
            not _is_valid_word(combined) and  # Not a real word
            _is_valid_word(after)):  # But the second part is valid
            return before + ' ' + after
        
        return match.group(0)
    
    # Apply conservatively
    fixed_text = re.sub(line_break_pattern, fix_line_break, text)
    
    return fixed_text


def _fix_quote_boundary_gluing(text: str) -> str:
    """Fix word gluing that occurs around quotation marks."""
    # Pattern: word glued to quote mark or quote mark glued to word
    
    # Fix missing spaces before opening quotes
    text = re.sub(r'([a-z])"([A-Z])', r'\1 "\2', text)
    
    # Fix missing spaces after closing quotes
    text = re.sub(r'"([a-z])', r'" \1', text)
    
    # Fix missing spaces around quotes in general
    text = re.sub(r'([a-z])"([a-z])', r'\1" \2', text)
    
    return text


def _is_likely_compound_word(word1: str, word2: str) -> bool:
    """Heuristic to determine if two word parts form a legitimate compound word."""
    # Simple heuristics for compound words
    compound_indicators = [
        (word1.lower().endswith(('over', 'under', 'out', 'up', 'down')), True),
        (word2.lower().startswith(('able', 'ing', 'ed', 'er', 'est')), True),
        (word1.lower() in {'web', 'data', 'file', 'text', 'code'}, True),
        (word2.lower() in {'base', 'set', 'type', 'name', 'path'}, True),
    ]
    
    for condition, is_compound in compound_indicators:
        if condition:
            return is_compound
    
    return False


def _is_valid_word(word: str) -> bool:
    """Simple heuristic to check if a word looks valid."""
    if not word or len(word) < 2:
        return False
    
    consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}', word)
    if consonant_clusters:
        return False
    
    if re.match(r'^[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]+$', word) and len(word) > 3:
        return False
    
    vowel_count = len(re.findall(r'[aeiouAEIOU]', word))
    consonant_count = len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWLMNPQRSTVWXYZ]', word))
    
    if vowel_count == 0 and consonant_count > 2:
        return False
    
    return True
