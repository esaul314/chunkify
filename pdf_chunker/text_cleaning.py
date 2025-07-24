import re
import os
import logging
import json
import ftfy
from typing import List, Callable, Tuple

logger = logging.getLogger(__name__)

# Simple pipe utility to apply a series of functions to a value
def pipe(value, *funcs):
    """
    Thread `value` through each function in `funcs` sequentially.
    """
    for fn in funcs:
        value = fn(value)
    return value

# Patterns for splitting and cleaning
PARAGRAPH_SPLIT = re.compile(r'\n{2,}')
SINGLE_LINE_BREAK = re.compile(r'(?<!\n)\n(?!\n)')
CONTROL_CHARS = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')

# Hyphenation break fix patterns
HYPHEN_BREAK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'([A-Za-z])\-\s*\n\s*([a-z])'), r'\1\2'),
    (re.compile(r'([A-Za-z])\-\s+([a-z])'),  r'\1\2'),
]

# Quote normalization patterns
QUOTE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'(\w)"([A-Z])'),      r'\1 "\2'),
    (re.compile(r'"(\w)'),             r'" \1'),
    (re.compile(r'"{2,}'),              '"'),
    (re.compile(r"'{2,}"),             "'"),
    (re.compile(r'\s+"([^\"]*?)"\s+'), r' "\1" '),
]


def normalize_ligatures(text: str) -> str:
    """
    Fix Unicode ligatures and encoding issues via ftfy.
    """
    return ftfy.fix_text(text)


def normalize_quotes(text: str) -> str:
    """
    Convert smart quotes to ASCII and fix spacing/patterns.
    """
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


def fix_hyphenated_breaks(text: str) -> str:
    """
    Merge hyphenated splits across line breaks or spaces.
    """
    for pattern, repl in HYPHEN_BREAK_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def remove_control_characters(text: str) -> str:
    """
    Strip out ASCII control characters that break JSON.
    """
    return CONTROL_CHARS.sub('', text)


def collapse_single_line_breaks(text: str) -> str:
    """
    Replace single newlines (not paragraphs) with spaces.
    """
    return SINGLE_LINE_BREAK.sub(' ', text)


def consolidate_whitespace(text: str) -> str:
    """
    Collapse spaces/tabs into single spaces and trim edges.
    """
    return re.sub(r'[ \t\r\f\v]+', ' ', text).strip()


def validate_json_safety(text: str) -> Tuple[bool, List[str]]:
    """
    Check JSON serialization, control chars, and quote balance.
    """
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
    """
    Remove/fix chars and quote fragments to satisfy JSON safety.
    """
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
    Clean a paragraph: control chars, ligatures, hyphens,
    collapse lines, normalize quotes, and whitespace.
    """
    pipeline: List[Callable[[str], str]] = [
        remove_control_characters,
        normalize_ligatures,
        fix_hyphenated_breaks,
        collapse_single_line_breaks,
        normalize_quotes,
        consolidate_whitespace,
    ]
    return pipe(paragraph, *pipeline)


def clean_text(text: str) -> str:
    """
    Clean multi-paragraph text, preserving paragraph breaks and
    normalizing content before final JSON safety.
    """
    # Short-circuit empty input
    if not text or not text.strip():
        return ''

    # Delegate to PyMuPDF4LLM if enabled
    use_pymupdf4llm = os.getenv('PDF_CHUNKER_USE_PYMUPDF4LLM', '').lower() in ('true', '1', 'yes', 'on')
    if use_pymupdf4llm:
        try:
            from .pymupdf4llm_integration import is_pymupdf4llm_available, clean_text_with_pymupdf4llm
            if is_pymupdf4llm_available():
                return clean_text_with_pymupdf4llm(text)
        except Exception:
            pass

    # Split into paragraphs on two-or-more consecutive newlines
    paragraphs = [p for p in PARAGRAPH_SPLIT.split(text) if p.strip()]

    # Clean each paragraph and rejoin with two real newlines
    cleaned_paragraphs = [clean_paragraph(p) for p in paragraphs]
    result = "\n\n".join(cleaned_paragraphs)

    # Ensure JSON-safe output
    safe, issues = validate_json_safety(result)
    if not safe:
        logger.warning(f"JSON safety issues detected: {issues}")
        result = apply_json_safety_fixes(result)

    return result

