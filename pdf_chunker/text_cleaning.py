"""text_cleaning

Public API (stable):
- fix_hyphenated_linebreaks
- rejoin_hyphenated_words
- collapse_artifact_breaks
- remove_stray_bullet_lines
- cleanup_bullet_fragments
- merge_number_suffix_lines
- insert_numbered_list_newlines
- collapse_single_newlines
- normalize_ligatures
- normalize_quotes
- remove_underscore_emphasis
- strip_underscore_wrapping
- normalize_newlines
- remove_control_characters
- consolidate_whitespace
- strip_headers_and_footers
- merge_spurious_paragraph_breaks
- validate_json_safety
- apply_json_safety_fixes
- clean_paragraph
- clean_text

Notes:
- Functions are pure (no side effects) except for logging.
- Regexes are precompiled and grouped for readability.
- Composition uses `pipe()` for a declarative flow.
- External behaviour preserved. Any minor internal reorganizations were
  done to improve clarity and maintainability.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import reduce
from typing import Callable, List, Match, Tuple, TypeVar

import ftfy
from wordfreq import zipf_frequency

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

PREVIEW_LEN = 100


def pipe(value: T, *funcs: Callable[[T], T]) -> T:
    """Left-to-right function composition for a single value."""
    for fn in funcs:
        value = fn(value)
    return value


def _preview(s: str, n: int = PREVIEW_LEN) -> str:
    """Return a safe preview slice for debug logs."""
    return repr(s[:n])


# ---------------------------------------------------------------------------
# Patterns & Constants
# ---------------------------------------------------------------------------

# Paragraphs & control characters
PARAGRAPH_BREAK = re.compile(r"\n{2,}")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u202d\u202c]")

# Quote normalization helpers
SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‚": '"',
    "\x84": '"',
    "‘": "'",
    "’": "'",
    "`": "'",
}

QUOTE_SPACING_PATTERNS: List[Tuple[re.Pattern[str], str]] = [

    # ensure space before an opening quote stuck to previous text (letters only)
    (re.compile(r'(?<=[A-Za-z])"(?=\w)'), r' "'),
    # ensure space after a closing quote stuck to a word character
    (re.compile(r'(?<=[A-Za-z])"(?=[A-Za-z])'), r'" '),
    (re.compile(r'"{2,}'), '"'),
    (re.compile(r"'{2,}"), "'"),
]

# Hyphenation (handles soft and unicode hyphens across line breaks)
# Keep character set stable; duplicates are harmless in a character class.
_HYPHEN_CHARS = "\u2010\u2011\u002d\u00ad\u1400\ufe63‐-"
HYPHEN_CHARS_ESC = re.escape(_HYPHEN_CHARS)
SOFT_HYPHEN_RE = re.compile("\u00ad")

# Bullets
BULLET_CHARS_ESC = re.escape("*•")

# Terminal punctuation for quoted sentence continuation
END_PUNCT = ".!?…"

# Inline artifacts
# Avoid collapsing list markers like "2.\n3." by skipping digits after the break
COLLAPSE_ARTIFACT_BREAKS_RE = re.compile(r"([._])\n(?!\d)(\w)")
PIPE_RE = re.compile(r"\|")
UNDERSCORE_WRAP_RE = re.compile(r"_{1,2}([^_]+?)_{1,2}")
DANGLING_UNDERSCORE_RE = re.compile(r"(?<!\w)_+|_+(?!\w)")

# Stray bullet variants
STRAY_BULLET_SOLO_RE = re.compile(rf"\n[{BULLET_CHARS_ESC}](?:\n+|$)")
# Guard against collapsing legitimate list items (e.g., after colons)
STRAY_BULLET_AFTER_NEWLINE_RE = re.compile(
    rf"(?<![\n:{BULLET_CHARS_ESC}])\n[{BULLET_CHARS_ESC}]\s+(?=[a-z0-9])"
)
STRAY_BULLET_INLINE_RE = re.compile(rf"(?<=\S)[ \t][{BULLET_CHARS_ESC}]\s+(?=[a-z0-9])")

# Numbered list helpers
NUMBER_SUFFIX_LINE_RE = re.compile(r"\n(\d+\.)(\s*)")
NUMBERED_AFTER_COLON_RE = re.compile(r":\s*(?!\n)(\d{1,3}[.)])")
NUMBERED_INLINE_CANDIDATE_RE = re.compile(
    r"(\d{1,3}[.)](?:[^\n]|\n(?!\n|\d))*?)\s+(?=(\d{1,3}[.)]))"
)
NUMBERED_END_RE = re.compile(
    rf"(\d{{1,3}}[.)][^\n]+?)"
    rf"(?<![{re.escape(END_PUNCT)}]\")"
    rf"(?=\s+(?:[{BULLET_CHARS_ESC}]|[A-Z][a-z]+\b(?!\s+\d)|$))"
)

# List break preservation
LIST_BREAK_RE = re.compile(rf"\n(?=\s*(?:[{BULLET_CHARS_ESC}]|-\s|\d+[.)]))")
COLON_BULLET_START_RE = re.compile(rf":\s*(?=-|[{BULLET_CHARS_ESC}])")

# Newline/split heuristics
DOUBLE_NEWLINE_RE = re.compile(r"([A-Za-z]+)\n{2,}\s*([a-z][A-Za-z]+)")
SPLIT_WORD_RE = re.compile(r"([A-Za-z]{2,})(?:\n|\s{2,}|\u00A0)([a-z]{2,})")

STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "of",
        "for",
        "and",
        "with",
        "from",
        "this",
        "that",
        "is",
        "to",
        "in",
        "on",
        "as",
        "by",
        "then",
        "up",
    }
)

# Footnote handling
FOOTNOTE_BRACKETED_RE = re.compile(rf"\[\d+\](?:[{re.escape(END_PUNCT)}])?$")
FOOTNOTE_DOTTED_RE = re.compile(r"\.(\d+)$")
FOOTNOTE_PLAIN_RE = re.compile(r"(?<=[^\s\d])(\d+)$")
SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_MAP = str.maketrans(SUPERSCRIPT_DIGITS, "0123456789")
_SUP_DIGITS_ESC = re.escape(SUPERSCRIPT_DIGITS)
INLINE_FOOTNOTE_RE = re.compile(rf"(?<!\d)\.([0-9{_SUP_DIGITS_ESC}]+)(\s|$)")

# Hyphenated word joiners (compiled with constants above)
_HYPHEN_BULLET_OPT = rf"(?:[{BULLET_CHARS_ESC}]\s*)?"
HYPHEN_BREAK_RE = re.compile(rf"(\w)[{HYPHEN_CHARS_ESC}]\s*\n\s*{_HYPHEN_BULLET_OPT}([A-Za-z]+)")
HYPHEN_SPACE_RE = re.compile(rf"(\w)[{HYPHEN_CHARS_ESC}]\s+{_HYPHEN_BULLET_OPT}([A-Za-z]+)")


# ---------------------------------------------------------------------------
# Hyphenation and word glue fixes
# ---------------------------------------------------------------------------


def _join_hyphenated_words(text: str) -> str:
    """Merge words broken with hyphenation across line breaks."""

    def repl(match: Match[str]) -> str:
        head, tail = match.group(1), match.group(2)
        tail = tail[1:] if tail and tail[0].lower() == head.lower() else tail
        return f"{head}{tail}"

    return HYPHEN_SPACE_RE.sub(repl, HYPHEN_BREAK_RE.sub(repl, text))


def _remove_soft_hyphens(text: str) -> str:
    return SOFT_HYPHEN_RE.sub("", text)


def fix_hyphenated_linebreaks(text: str) -> str:
    """Join words split across lines without removing valid hyphens."""
    return pipe(text, _join_hyphenated_words, _remove_soft_hyphens)


def rejoin_hyphenated_words(text: str) -> str:
    """Public helper that reuses ``fix_hyphenated_linebreaks``."""
    return fix_hyphenated_linebreaks(text)


def _maybe_join_words(head: str, tail: str) -> str:
    """Return head+tail when the combined form is more plausible than separate words."""

    if head.lower() in STOPWORDS or tail.lower() in STOPWORDS:
        return f"{head} {tail}"

    head_freq = zipf_frequency(head, "en")
    tail_freq = zipf_frequency(tail, "en")
    combined = head + tail
    combined_freq = zipf_frequency(combined, "en")

    if combined_freq > max(head_freq, tail_freq):
        return combined

    if head[-1].lower() == tail[0].lower():
        dedup = head + tail[1:]
        if zipf_frequency(dedup, "en") > max(head_freq, tail_freq):
            return dedup

    return f"{head} {tail}"


def _fix_double_newlines(text: str) -> str:
    """Resolve words or phrases separated by double newlines using word heuristics."""
    return DOUBLE_NEWLINE_RE.sub(lambda m: _maybe_join_words(m.group(1), m.group(2)), text)


def _fix_split_words(text: str) -> str:
    """Join words erroneously split by whitespace using word heuristics."""
    return SPLIT_WORD_RE.sub(lambda m: _maybe_join_words(m.group(1), m.group(2)), text)


# ---------------------------------------------------------------------------
# Quote & ligature normalization
# ---------------------------------------------------------------------------


def _map_smart_quotes(text: str) -> str:
    """Map smart quotes to their ASCII equivalents."""
    return "".join(SMART_QUOTES.get(ch, ch) for ch in text)


def _fix_quote_spacing(text: str) -> str:
    """Apply spacing and duplication fixes around quotes."""
    return reduce(lambda s, p: p[0].sub(p[1], s), QUOTE_SPACING_PATTERNS, text)


def normalize_quotes(text: str) -> str:
    """Map smart quotes to ASCII without altering spacing."""
    return text if not text else _map_smart_quotes(text)


def normalize_ligatures(text: str) -> str:
    return ftfy.fix_text(text)


# ---------------------------------------------------------------------------
# Newline & list handling
# ---------------------------------------------------------------------------


def collapse_artifact_breaks(text: str) -> str:
    """Remove unwanted breaks after ., _, etc. (e.g., systems._\nThis → systems. This)"""
    return COLLAPSE_ARTIFACT_BREAKS_RE.sub(r"\1 \2", text)


def merge_number_suffix_lines(text: str) -> str:
    """Join lines where a terminal number is split onto its own line."""

    def repl(match: Match[str]) -> str:
        start = match.start()
        prev = text[text.rfind("\n", 0, start) + 1 : start].strip()
        last = prev.split()[-1].lower() if prev else ""
        if (
            not prev
            or prev.endswith(":")
            or re.match(r"\d+[.)]", prev)
            or re.search(r"[.!?]$", prev)
            or last == "chapter"
        ):
            return match.group(0)
        return f" {match.group(1)}{match.group(2)}"

    return NUMBER_SUFFIX_LINE_RE.sub(repl, text)


def drop_spurious_number_markers(text: str) -> str:
    """Remove numbered markers created by hyphenated splits or isolated lines."""
    return pipe(
        text,
        lambda t: re.sub(r"-\n\d+[.)]\s*", "-", t),
        lambda t: re.sub(r"\n\d+[.)]\n", "\n", t),
        lambda t: re.sub(r"\n\d+[.)]\s+(?=[A-Z]{2,}\b)", "\n", t),
        lambda t: re.sub(r"(?<=\b[a-z])\s+\d+[.)]\s+(?=[a-z])", " ", t),
        lambda t: re.sub("(?<=[a-z])\\s+\\d+[.)](?:['\"\\u2019])?(?=\\s*$)", "", t),
        lambda t: re.sub(r"\n{2,}(\d+[.)])", r"\n\1", t),
    )


def remove_stray_bullet_lines(text: str) -> str:
    """Collapse stray bullet markers while keeping legitimate list breaks intact."""
    return pipe(
        text,
        lambda t: STRAY_BULLET_SOLO_RE.sub("\n", t),
        lambda t: STRAY_BULLET_AFTER_NEWLINE_RE.sub(" ", t),
        lambda t: STRAY_BULLET_INLINE_RE.sub(" ", t),
        lambda t: re.sub(rf"\n+(?=[{BULLET_CHARS_ESC}])", "\n", t),
    )


def cleanup_bullet_fragments(text: str) -> str:
    """Public helper that delegates to ``remove_stray_bullet_lines``."""
    return remove_stray_bullet_lines(text)


def strip_headers_and_footers(text: str) -> str:
    """Remove simple header/footer lines containing ``|`` separators."""
    lines = text.splitlines()
    filtered = (ln for ln in lines if "|" not in ln)
    return "\n".join(filtered)


def _split_inline_numbered(match: Match[str]) -> str:
    """Insert a newline before the next item unless the preceding token is title-cased and inline."""
    head, tail = match.groups()
    tokens = head.rstrip().split()
    last = tokens[-1] if tokens else ""
    return (
        match.group(0)
        if last.istitle() and tail.rstrip(".)").isdigit() and "\n" not in head
        else f"{head}\n"
    )


def _apply_inline_numbered(text: str) -> str:
    """Split inline numbered items, handling sequential overlaps in one pass."""
    return NUMBERED_INLINE_CANDIDATE_RE.sub(_split_inline_numbered, text)


def insert_numbered_list_newlines(text: str) -> str:
    """Insert newlines around numbered list items and terminate the list with a paragraph break."""
    text = NUMBERED_AFTER_COLON_RE.sub(r":\n\1", text)
    text = _apply_inline_numbered(text)
    return NUMBERED_END_RE.sub(r"\1\n\n", text)


def _preserve_list_newlines(text: str) -> str:
    """Keep newlines that precede bullets or enumerated items."""
    placeholder = "[[LIST_BREAK]]"
    return LIST_BREAK_RE.sub(placeholder, text).replace("\n", " ").replace(placeholder, "\n")


def collapse_single_newlines(text: str) -> str:
    logger.debug(f"collapse_single_newlines called with {len(text)} chars")
    logger.debug(f"Input text preview: {_preview(text)}")

    list_break, para_break = "[[LIST_BREAK]]", "[[PARAGRAPH_BREAK]]"

    result = pipe(
        text,
        merge_number_suffix_lines,
        lambda t: COLON_BULLET_START_RE.sub(":\n", t),
        lambda t: LIST_BREAK_RE.sub(list_break, t),
        lambda t: PARAGRAPH_BREAK.sub(para_break, t),
        lambda t: t.replace("\n", " "),
        lambda t: t.replace(para_break, "\n\n").replace(list_break, "\n"),
        _fix_quote_spacing,
    )

    logger.debug(f"Output text preview: {_preview(result)}")
    return result


# ---------------------------------------------------------------------------
# Heading/list detection & footnotes
# ---------------------------------------------------------------------------


def _starts_list_item(line: str) -> bool:
    return bool(re.match(rf"([{BULLET_CHARS_ESC}]|\d+[.)])\s", line))


def _starts_new_list_item(text: str) -> bool:
    return _starts_list_item(text.lstrip())


def _is_probable_heading(text: str) -> bool:
    """Heuristically determine whether a line looks like a heading."""
    stripped = text.strip()
    if not stripped or len(stripped) > 80:
        return False

    words = [w for w in re.split(r"\s+", stripped) if w]

    # Lines ending with a bare vertical bar are usually headers or footers.
    if stripped.endswith("|"):
        return False

    # Short phrases with at least one capitalized word are often headings.
    if 1 < len(words) <= 3 and stripped[0].isupper() and not re.search(r"[.!?]$", stripped):
        return True

    # Quoted fragments are likely part of sentences, not headings
    opens = stripped.startswith(('"', "'"))
    closes = stripped.endswith(('"', "'"))
    if opens != closes:
        return False

    # Bulleted or UPPER CASE
    if re.match(r"^[\-\u2022*]\s", stripped):
        return True
    if stripped.isupper():
        return True

    # Headings often contain a colon or digits without terminal punctuation
    if ":" in stripped and not re.search(r"[.!?]$", stripped):
        return True

    # Short phrases ending with ! or ? as headings
    if re.search(r"[!?]$", stripped):
        word_count = len(words)
        if 1 < word_count <= 6 and len(stripped) <= 60:
            return True

    if not words:
        return False

    alpha_words = [w for w in words if w[0].isalpha()]
    if not alpha_words:
        return False

    upper_ratio = sum(w[0].isupper() for w in alpha_words) / len(alpha_words)
    return upper_ratio >= 0.5 and not re.search(r"[.!?]$", stripped)


def _has_unbalanced_quotes(text: str) -> bool:
    """Return True if the text contains an odd number of quotes."""
    return text.count('"') % 2 == 1 or text.count("'") % 2 == 1


def _ends_with_footnote(text: str) -> bool:
    stripped = text.strip()
    return bool(
        FOOTNOTE_BRACKETED_RE.search(stripped)
        or FOOTNOTE_DOTTED_RE.search(stripped)
        or FOOTNOTE_PLAIN_RE.search(stripped)
    )


def _normalize_trailing_footnote(text: str) -> str:
    stripped = text.rstrip()
    match = FOOTNOTE_BRACKETED_RE.search(stripped)
    if match:
        return stripped
    match = FOOTNOTE_DOTTED_RE.search(stripped)
    if match:
        num = match.group(1)
        return FOOTNOTE_DOTTED_RE.sub(f"[{num}].", stripped)
    match = FOOTNOTE_PLAIN_RE.search(stripped)
    if match:
        num = match.group(1)
        return FOOTNOTE_PLAIN_RE.sub(f"[{num}]", stripped)
    return stripped


def _normalize_inline_footnotes(text: str) -> str:
    def repl(match: Match[str]) -> str:
        digits = match.group(1).translate(_SUPERSCRIPT_MAP)
        return f"[{digits}].{match.group(2)}"

    return INLINE_FOOTNOTE_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Safety & whitespace utilities
# ---------------------------------------------------------------------------


def normalize_newlines(text: str) -> str:
    """Convert CRLF/CR and unicode separators to LF."""
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )


def remove_control_characters(text: str) -> str:
    return CONTROL_CHARS.sub("", text)


def consolidate_whitespace(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def replace_pipes(text: str) -> str:
    """Normalize stray pipe characters to colons."""
    return PIPE_RE.sub(":", text)


def remove_underscore_emphasis(text: str) -> str:
    """Remove single/double underscore emphasis markers."""
    return UNDERSCORE_WRAP_RE.sub(r"\1", text)


def strip_underscore_wrapping(text: str) -> str:
    """Public helper that removes underscore emphasis wrappers."""
    return remove_underscore_emphasis(text)


def remove_dangling_underscores(text: str) -> str:
    """Remove underscores that don't join word characters."""
    return DANGLING_UNDERSCORE_RE.sub("", text)


# ---------------------------------------------------------------------------
# Paragraph merge logic
# ---------------------------------------------------------------------------


def merge_spurious_paragraph_breaks(text: str) -> str:
    parts = [p for p in PARAGRAPH_BREAK.split(text) if p.strip()]
    merged: List[str] = []
    for part in parts:
        if merged:
            prev = merged[-1]
            author_line = part.lstrip()
            if author_line.startswith("—"):
                merged[-1] = f"{prev.rstrip()} {author_line}"
                continue
            last_line = prev.strip().splitlines()[-1]
            if _starts_list_item(last_line):
                if _ends_with_footnote(prev) and not _starts_new_list_item(part):
                    normalized = _normalize_trailing_footnote(prev)
                    merged[-1] = f"{normalized} {part.lstrip()}"
                else:
                    merged.append(part)
                continue
            if _ends_with_footnote(prev) and not _starts_new_list_item(part):
                normalized = _normalize_trailing_footnote(prev)
                merged[-1] = f"{normalized} {part.lstrip()}"
                continue
            if not any(_is_probable_heading(seg) for seg in (prev, part)):
                if _has_unbalanced_quotes(prev) and not _has_unbalanced_quotes(prev + part):
                    merged[-1] = f"{prev.rstrip()} {part.lstrip()}"
                    continue
                if len(prev) < 60 or not prev.rstrip().endswith((".", "?", "!")):
                    merged[-1] = f"{prev.rstrip()} {part.lstrip()}"
                    continue
        merged.append(part)
    return "\n\n".join(merged)


# ---------------------------------------------------------------------------
# JSON safety helpers
# ---------------------------------------------------------------------------


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
    if re.search(r'^[",]', text.strip()):
        issues.append("Text starts with problematic punctuation")
    if re.search(r'[",]$', text.strip()):
        issues.append("Text ends with problematic punctuation")
    try:
        text.encode("utf-8").decode("utf-8")
    except UnicodeError:
        issues.append("Unicode encoding issues detected")
    return (len(issues) == 0, issues)


def apply_json_safety_fixes(text: str) -> str:
    fixed = CONTROL_CHARS.sub("", text)
    if fixed.startswith('", '):
        fixed = fixed[3:]
    elif fixed.startswith('"') and len(fixed) > 1 and fixed[1].islower():
        fixed = fixed[1:]
    try:
        fixed = fixed.encode("utf-8", errors="replace").decode("utf-8")
    except UnicodeError:
        fixed = "".join(ch for ch in fixed if ord(ch) < 128)
    if fixed.count('"') % 2 != 0:
        if fixed.endswith('"'):
            fixed = fixed[:-1]
        elif fixed.startswith('"'):
            fixed = fixed[1:]
    return fixed


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def clean_paragraph(paragraph: str) -> str:
    """
    Cleans a single paragraph: removes mid-line hyphens, artifacts,
    collapses all newlines (if present) to spaces, and normalizes.
    """
    return pipe(
        paragraph,
        rejoin_hyphenated_words,
        strip_headers_and_footers,
        replace_pipes,
        collapse_artifact_breaks,
        cleanup_bullet_fragments,
        _preserve_list_newlines,
        remove_control_characters,
        normalize_ligatures,
        consolidate_whitespace,
    )


def _clean_text_impl(text: str) -> str:
    """
    Cleans multi-paragraph text, preserving paragraph breaks,
    using clean_paragraph() for each paragraph.
    """
    if not text or not text.strip():
        return ""

    logger.debug(f"clean_text called with {len(text)} chars")
    logger.debug(f"Input text preview: {_preview(text)}")

    # Optional strategy via env
    from .env_utils import use_pymupdf4llm as _use_pymupdf4llm

    env_flag = os.getenv("PDF_CHUNKER_USE_PYMUPDF4LLM")
    enabled = bool(env_flag) and _use_pymupdf4llm()
    logger.debug(f"PDF_CHUNKER_USE_PYMUPDF4LLM environment variable: {env_flag or 'not set'}")
    logger.debug(f"use_pymupdf4llm evaluated to: {enabled}")

    if enabled:
        logger.debug("Using PyMuPDF4LLM text cleaning path")
        try:
            from .pymupdf4llm_integration import (
                is_pymupdf4llm_available,
                clean_text_with_pymupdf4llm,
            )

            if is_pymupdf4llm_available():
                logger.debug("PyMuPDF4LLM is available, calling clean_text_with_pymupdf4llm")
                result = clean_text_with_pymupdf4llm(text)
                logger.debug(f"PyMuPDF4LLM result preview: {_preview(result)}")
                return result
            else:
                logger.debug("PyMuPDF4LLM not available, falling back to traditional")
        except Exception as e:  # noqa: BLE001 - keep behaviour identical
            logger.debug(
                f"PyMuPDF4LLM cleaning failed with exception: {e}, falling back to traditional"
            )
            pass

    logger.debug("Using traditional text cleaning path")

    # Normalize newlines and fix broken words before other cleanup
    logger.debug("Calling normalize_newlines")
    text = normalize_newlines(text)
    logger.debug(f"After normalize_newlines: {_preview(text)}")

    logger.debug("Calling fix_hyphenated_linebreaks")
    text = fix_hyphenated_linebreaks(text)
    logger.debug(f"After fix_hyphenated_linebreaks: {_preview(text)}")

    logger.debug("Calling normalize_quotes")
    text = normalize_quotes(text)
    logger.debug(f"After normalize_quotes: {_preview(text)}")

    logger.debug("Calling remove_control_characters")
    text = remove_control_characters(text)
    logger.debug(f"After remove_control_characters: {_preview(text)}")

    logger.debug("Calling _fix_double_newlines")
    text = _fix_double_newlines(text)
    logger.debug(f"After _fix_double_newlines: {_preview(text)}")

    logger.debug("Calling drop_spurious_number_markers")
    text = drop_spurious_number_markers(text)
    logger.debug(f"After drop_spurious_number_markers: {_preview(text)}")

    logger.debug("Calling insert_numbered_list_newlines")
    text = insert_numbered_list_newlines(text)
    logger.debug(f"After insert_numbered_list_newlines: {_preview(text)}")

    # Collapse single line breaks except paragraph breaks
    logger.debug("Calling collapse_single_newlines")
    text = collapse_single_newlines(text)
    logger.debug(f"After collapse_single_newlines: {_preview(text)}")

    logger.debug("Calling merge_spurious_paragraph_breaks")
    text = merge_spurious_paragraph_breaks(text)
    logger.debug(f"After merge_spurious_paragraph_breaks: {_preview(text)}")

    logger.debug("Calling _normalize_inline_footnotes")
    text = _normalize_inline_footnotes(text)
    logger.debug(f"After _normalize_inline_footnotes: {_preview(text)}")

    logger.debug("Calling _fix_split_words")
    text = _fix_split_words(text)
    logger.debug(f"After _fix_split_words: {_preview(text)}")

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

    logger.debug(f"Final clean_text result preview: {_preview(result)}")
    return result


def clean_text(text: str) -> str:
    """Shim maintaining legacy API while delegating to the text_clean pass."""
    from pdf_chunker.framework import Artifact
    from pdf_chunker.passes.text_clean import text_clean as _text_clean

    return _text_clean(Artifact(payload=text)).payload


__all__ = [
    # public API
    "fix_hyphenated_linebreaks",
    "rejoin_hyphenated_words",
    "collapse_artifact_breaks",
    "remove_stray_bullet_lines",
    "cleanup_bullet_fragments",
    "merge_number_suffix_lines",
    "drop_spurious_number_markers",
    "insert_numbered_list_newlines",
    "collapse_single_newlines",
    "normalize_ligatures",
    "normalize_quotes",
    "remove_underscore_emphasis",
    "strip_underscore_wrapping",
    "remove_dangling_underscores",
    "normalize_newlines",
    "remove_control_characters",
    "consolidate_whitespace",
    "merge_spurious_paragraph_breaks",
    "validate_json_safety",
    "apply_json_safety_fixes",
    "strip_headers_and_footers",
    "clean_paragraph",
    "clean_text",
]
