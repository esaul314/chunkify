import re
import os
import logging
import json
import ftfy
from typing import List, Callable, Tuple

logger = logging.getLogger(__name__)

def pipe(value, *funcs):
    for fn in funcs:
        value = fn(value)
    return value

# Patterns
PARAGRAPH_BREAK = re.compile(r'\n{2,}')
CONTROL_CHARS = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')

# Quote normalization patterns
QUOTE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'(\w)"([A-Z])'), r'\1 "\2'),
    (re.compile(r'"(\w)'), r'" \1'),
    (re.compile(r'"{2,}'), '"'),
    (re.compile(r"'{2,}"), "'"),
    (re.compile(r'\s+"([^\"]*?)"\s+'), r' "\1" '),
]

# Hyphenation (handles soft and unicode hyphens across line breaks)
HYPHEN_CHARS = "-\u2010\u2011"


def fix_hyphenated_linebreaks(text: str) -> str:
    """Join words split across lines by hyphen-like characters."""

    pattern_break = rf"(\w)[{HYPHEN_CHARS}]\s*\n\s*(\w)"
    text = re.sub(pattern_break, r"\1\2", text)

    pattern_space = rf"(\w)[{HYPHEN_CHARS}]\s+([a-z])"
    text = re.sub(pattern_space, r"\1\2", text)

    text = re.sub(r"[\u00ad\u2010\u2011]", "", text)

    return text

def collapse_artifact_breaks(text: str) -> str:
    # Remove unwanted breaks after ., _, etc. (e.g., systems._\nThis → systems. This)
    return re.sub(r'([._])\n(\w)', r'\1 \2', text)

def collapse_single_newlines(text: str) -> str:
    logger.debug(f"collapse_single_newlines called with {len(text)} chars")
    logger.debug(f"Input text preview: {repr(text[:100])}")

    # First, protect paragraph breaks (2+ newlines) by replacing with placeholder
    text = re.sub(r'\n{2,}', '[[PARAGRAPH_BREAK]]', text)
    # Replace all remaining single newlines with spaces
    text = text.replace('\n', ' ')
    # Restore paragraph breaks
    text = text.replace('[[PARAGRAPH_BREAK]]', '\n\n')

    logger.debug(f"Output text preview: {repr(text[:100])}")
    return text

def normalize_ligatures(text: str) -> str:
    return ftfy.fix_text(text)

def normalize_quotes(text: str) -> str:
    if not text:
        return text
    mapping = {
        '“': '"', '”': '"', '„': '"', '‚': '"',
        '‘': "'", '’': "'", '`': "'"
    }
    for smart, ascii_q in mapping.items():
        text = text.replace(smart, ascii_q)
    for pattern, repl in QUOTE_PATTERNS:
        text = pattern.sub(repl, text)
    return text

def normalize_newlines(text: str) -> str:
    # Convert all CRLF and CR to LF, and unicode separators to LF as well
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\u2028', '\n').replace('\u2029', '\n')
    return text

def remove_control_characters(text: str) -> str:
    return CONTROL_CHARS.sub('', text)

def consolidate_whitespace(text: str) -> str:
    return re.sub(r'[ \t\r\f\v]+', ' ', text).strip()

def validate_json_safety(text: str) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    try:
        json.dumps({"text": text}, ensure_ascii=False)
    except (TypeError, UnicodeEncodeError) as e:
        issues.append(f"JSON serialization failed: {e}")
    found = CONTROL_CHARS.findall(text)
    if found:
        issues.append(f"Control characters found: {len(found)}")
    dq = text.count('"')
    if dq % 2 != 0:
        issues.append(f"Unbalanced double quotes: {dq}")
    if re.search(r'^[\",]', text.strip()):
        issues.append("Text starts with problematic punctuation")
    if re.search(r'[\",]$', text.strip()):
        issues.append("Text ends with problematic punctuation")
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeError:
        issues.append("Unicode encoding issues detected")
    return (len(issues) == 0, issues)

def apply_json_safety_fixes(text: str) -> str:
    fixed = CONTROL_CHARS.sub('', text)
    if fixed.startswith('", '):
        fixed = fixed[3:]
    elif fixed.startswith('"') and len(fixed) > 1 and fixed[1].islower():
        fixed = fixed[1:]
    if fixed.endswith(', "') or fixed.endswith(',"'):
        fixed = fixed[:-2]
    try:
        fixed = fixed.encode('utf-8', errors='replace').decode('utf-8')
    except UnicodeError:
        fixed = ''.join(ch for ch in fixed if ord(ch) < 128)
    if fixed.count('"') % 2 != 0:
        if fixed.endswith('"'):
            fixed = fixed[:-1]
        elif fixed.startswith('"'):
            fixed = fixed[1:]
    return fixed

def clean_paragraph(paragraph: str) -> str:
    """
    Cleans a single paragraph: removes mid-line hyphens, artifacts,
    collapses all newlines (if present) to spaces, and normalizes.
    """
    return pipe(
        paragraph,
        fix_hyphenated_linebreaks,
        collapse_artifact_breaks,
        lambda t: t.replace('\n', ' '),    # any internal newlines to space
        normalize_ligatures,
        normalize_quotes,
        remove_control_characters,
        consolidate_whitespace,
    )

def clean_text(text: str) -> str:
    """
    Cleans multi-paragraph text, preserving paragraph breaks,
    using clean_paragraph() for each paragraph.
    """
    if not text or not text.strip():
        return ''

    logger.debug(f"clean_text called with {len(text)} chars")
    logger.debug(f"Input text preview: {repr(text[:100])}")

    from .env_utils import use_pymupdf4llm as _use_pymupdf4llm

    enabled = _use_pymupdf4llm()
    logger.debug(
        f"PDF_CHUNKER_USE_PYMUPDF4LLM environment variable: {os.getenv('PDF_CHUNKER_USE_PYMUPDF4LLM', 'not set')}"
    )
    logger.debug(f"use_pymupdf4llm evaluated to: {enabled}")
    
    if enabled:
        logger.debug("Using PyMuPDF4LLM text cleaning path")
        try:
            from .pymupdf4llm_integration import is_pymupdf4llm_available, clean_text_with_pymupdf4llm
            if is_pymupdf4llm_available():
                logger.debug("PyMuPDF4LLM is available, calling clean_text_with_pymupdf4llm")
                result = clean_text_with_pymupdf4llm(text)
                logger.debug(f"PyMuPDF4LLM result preview: {repr(result[:100])}")
                return result
            else:
                logger.debug("PyMuPDF4LLM not available, falling back to traditional")
        except Exception as e:
            logger.debug(f"PyMuPDF4LLM cleaning failed with exception: {e}, falling back to traditional")
            pass

    logger.debug("Using traditional text cleaning path")

    # Normalize newlines first
    logger.debug("Calling normalize_newlines")
    text = normalize_newlines(text)
    logger.debug(f"After normalize_newlines: {repr(text[:100])}")
    
    # Collapse single line breaks except paragraph breaks
    logger.debug("Calling collapse_single_newlines")
    text = collapse_single_newlines(text)
    logger.debug(f"After collapse_single_newlines: {repr(text[:100])}")

    # Split on paragraph breaks, clean each
    paragraphs = [p for p in PARAGRAPH_BREAK.split(text) if p.strip()]
    logger.debug(f"Split into {len(paragraphs)} paragraphs")
    cleaned_paragraphs = [clean_paragraph(p) for p in paragraphs]
    result = "\n\n".join(cleaned_paragraphs)

    # Final JSON safety check
    safe, issues = validate_json_safety(result)
    if not safe:
        logger.warning(f"JSON safety issues detected: {issues}")
        result = apply_json_safety_fixes(result)

    logger.debug(f"Final clean_text result preview: {repr(result[:100])}")
    return result
