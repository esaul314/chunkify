import re
import os
import logging
import json
import ftfy
from functools import reduce
from typing import Callable, List, Tuple, TypeVar
from wordfreq import zipf_frequency

logger = logging.getLogger(__name__)

T = TypeVar("T")


def pipe(value: T, *funcs: Callable[[T], T]) -> T:
    for fn in funcs:
        value = fn(value)
    return value


# Patterns
PARAGRAPH_BREAK = re.compile(r"\n{2,}")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u202d\u202c]")

# Quote normalization helpers
SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‚": '"',
    "‘": "'",
    "’": "'",
    "`": "'",
}

QUOTE_SPACING_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'(?<!\s)"(?=[A-Z])'), r' "'),
    (re.compile(r'(?<=\w)"(?=\w)'), r'" '),
    (re.compile(r'"{2,}'), '"'),
    (re.compile(r"'{2,}"), "'"),
    (re.compile(r'\s+"([^\"]*?)"\s+'), r' "\1" '),
]

# Hyphenation (handles soft and unicode hyphens across line breaks)
HYPHEN_CHARS_ESC = re.escape("\u2010\u2011\u002d\u00ad\u1400\ufe63‐-")
# HYPHEN_CHARS = "\u2010\u2011\u002d\u00ad\u1400\ufe63-"
# HYPHEN_CHARS = "-\u2010\u2011\u002d\u00ad\u1400\ufe63"
# HYPHEN_CHARS = "\u2010\u2011\u002d\u00ad\u1400\ufe63"

SOFT_HYPHEN_RE = re.compile("\u00ad")


BULLET_CHARS_ESC = re.escape("*•")


def _join_hyphenated_words(text: str) -> str:
    """Merge words broken with hyphenation across line breaks."""

    bullet_opt = rf"(?:[{BULLET_CHARS_ESC}]\s*)?"
    pattern_break = re.compile(
        rf"(\w)[{HYPHEN_CHARS_ESC}]\s*\n\s*{bullet_opt}([A-Za-z]+)"
    )
    pattern_space = re.compile(rf"(\w)[{HYPHEN_CHARS_ESC}]\s+{bullet_opt}([A-Za-z]+)")

    def repl(match: re.Match) -> str:
        head, tail = match.group(1), match.group(2)
        tail = tail[1:] if tail and tail[0].lower() == head.lower() else tail
        return f"{head}{tail}"

    return pattern_space.sub(repl, pattern_break.sub(repl, text))


DOUBLE_NEWLINE_RE = re.compile(r"([A-Za-z]+)\n{2,}\s*([a-z][A-Za-z]+)")

SPLIT_WORD_RE = re.compile(r"([A-Za-z]{2,})(?:\n|\s{2,}|\u00A0)([a-z]{2,})")
STOPWORDS = {
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


def _maybe_join_words(head: str, tail: str) -> str:
    """Return head+tail when the combined form is more plausible than separate words."""

    if head.lower() in STOPWORDS or tail.lower() in STOPWORDS:
        return f"{head} {tail}"

    head_freq, tail_freq = map(lambda w: zipf_frequency(w, "en"), (head, tail))
    word = head + tail
    combined_freq = zipf_frequency(word, "en")

    if combined_freq > max(head_freq, tail_freq):
        return word

    if head[-1].lower() == tail[0].lower():
        dedup = head + tail[1:]
        if zipf_frequency(dedup, "en") > max(head_freq, tail_freq):
            return dedup

    return f"{head} {tail}"


def _fix_double_newlines(text: str) -> str:
    """Resolve words or phrases separated by double newlines using word heuristics."""

    return DOUBLE_NEWLINE_RE.sub(
        lambda m: _maybe_join_words(m.group(1), m.group(2)), text
    )


def _fix_split_words(text: str) -> str:
    """Join words erroneously split by whitespace using word heuristics."""

    return SPLIT_WORD_RE.sub(lambda m: _maybe_join_words(m.group(1), m.group(2)), text)


def _remove_soft_hyphens(text: str) -> str:
    return SOFT_HYPHEN_RE.sub("", text)


def fix_hyphenated_linebreaks(text: str) -> str:
    """Join words split across lines without removing valid hyphens."""

    return pipe(text, _join_hyphenated_words, _remove_soft_hyphens)


def collapse_artifact_breaks(text: str) -> str:
    # Remove unwanted breaks after ., _, etc. (e.g., systems._\nThis → systems. This)
    return re.sub(r"([._])\n(\w)", r"\1 \2", text)


STRAY_BULLET_RE = re.compile(rf"\n[{BULLET_CHARS_ESC}](?:\n+|$)")


NUMBER_SUFFIX_LINE_RE = re.compile(r"\n(\d+\.)(\s*)")


def merge_number_suffix_lines(text: str) -> str:
    """Join lines where a terminal number is split onto its own line."""

    def repl(match: re.Match[str]) -> str:
        start = match.start()
        prev = text[text.rfind("\n", 0, start) + 1 : start].strip()
        if (
            not prev
            or prev.endswith(":")
            or re.match(r"\d+[.)]", prev)
            or re.search(r"[.!?]$", prev)
        ):
            return match.group(0)
        return f" {match.group(1)}{match.group(2)}"

    return NUMBER_SUFFIX_LINE_RE.sub(repl, text)


def remove_stray_bullet_lines(text: str) -> str:
    """Collapse bullet markers that appear alone or mid-item while preserving line breaks."""
    text = STRAY_BULLET_RE.sub("\n", text)
    text = re.sub(rf"\n[{BULLET_CHARS_ESC}]\s+(?=[a-z0-9])", " ", text)
    text = re.sub(rf"(?<=\S)[ \t][{BULLET_CHARS_ESC}]\s+(?=[a-z0-9])", " ", text)
    return re.sub(rf"\n+(?=[{BULLET_CHARS_ESC}])", "\n", text)


NUMBERED_AFTER_COLON_RE = re.compile(r":\s*(?!\n)(\d{1,3}[.)])")
NUMBERED_INLINE_RE = re.compile(r"(\d{1,3}[.)][^\n]+?)\s+(?=\d{1,3}[.)])")
# Avoid inserting paragraph breaks when a numbered item ends with a quoted
# question or exclamation that continues the same sentence.
NUMBERED_END_RE = re.compile(
    rf"(\d{{1,3}}[.)][^\n]+?)(?<![?!]\")(?=\s+(?:[{BULLET_CHARS_ESC}]|[A-Z][a-z]+\b|$))"
)


def insert_numbered_list_newlines(text: str) -> str:
    """Insert newlines around numbered list items and terminate the list with a paragraph break."""
    text = NUMBERED_AFTER_COLON_RE.sub(r":\n\1", text)
    text = NUMBERED_INLINE_RE.sub(r"\1\n", text)
    return NUMBERED_END_RE.sub(r"\1\n\n", text)


def _preserve_list_newlines(text: str) -> str:
    """Keep newlines that precede bullets or enumerated items."""
    placeholder = "[[LIST_BREAK]]"
    pattern = rf"\n(?=\s*(?:[{BULLET_CHARS_ESC}]|-\s|\d+[.)]))"
    return (
        re.sub(pattern, placeholder, text).replace("\n", " ").replace(placeholder, "\n")
    )


def collapse_single_newlines(text: str) -> str:
    logger.debug(f"collapse_single_newlines called with {len(text)} chars")
    logger.debug(f"Input text preview: {repr(text[:100])}")

    list_break = "[[LIST_BREAK]]"
    para_break = "[[PARAGRAPH_BREAK]]"
    list_re = rf"\n(?=\s*(?:[{BULLET_CHARS_ESC}]|-\s|\d+[.)]))"

    # Normalize colon bullet starts and protect paragraph and list breaks
    text = merge_number_suffix_lines(text)
    text = re.sub(rf":\s*(?=-|[{BULLET_CHARS_ESC}])", ":\n", text)
    text = re.sub(list_re, list_break, text)
    text = re.sub(r"\n{2,}", para_break, text)
    text = text.replace("\n", " ")

    # Restore preserved breaks
    text = text.replace(para_break, "\n\n").replace(list_break, "\n")

    logger.debug(f"Output text preview: {repr(text[:100])}")
    return text


def normalize_ligatures(text: str) -> str:
    return ftfy.fix_text(text)


def _map_smart_quotes(text: str) -> str:
    """Map smart quotes to their ASCII equivalents."""

    return "".join(SMART_QUOTES.get(ch, ch) for ch in text)


def _fix_quote_spacing(text: str) -> str:
    """Apply spacing and duplication fixes around quotes."""

    return reduce(lambda s, p: p[0].sub(p[1], s), QUOTE_SPACING_PATTERNS, text)


def normalize_quotes(text: str) -> str:
    return text if not text else pipe(text, _map_smart_quotes, _fix_quote_spacing)


def remove_underscore_emphasis(text: str) -> str:
    """Remove single/double underscore emphasis markers and stray edges."""

    cleaned = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    return cleaned.strip("_")


def normalize_newlines(text: str) -> str:
    # Convert all CRLF and CR to LF, and unicode separators to LF as well
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    return text


def remove_control_characters(text: str) -> str:
    return CONTROL_CHARS.sub("", text)


def consolidate_whitespace(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _is_probable_heading(text: str) -> bool:
    """Heuristically determine whether a line looks like a heading."""
    stripped = text.strip()
    if not stripped or len(stripped) > 80:
        return False

    words = [w for w in re.split(r"\s+", stripped) if w]

    # Lines ending with a bare vertical bar are usually headers or footers,
    # not actual headings. Treat them as non-headings so they remain with the
    # preceding body text rather than being attached to the next chunk.
    if stripped.endswith("|"):
        return False

    # Short phrases with at least one capitalized word are often headings even
    # without terminal punctuation. This helps catch cases like "Assimilate and
    # expand" that follow removed footers.
    # Allow short 2-3 word phrases with an initial capital as possible headings
    if (
        1 < len(words) <= 3
        and stripped[0].isupper()
        and not re.search(r"[.!?]$", stripped)
    ):
        return True

    # Quoted fragments are likely part of sentences, not headings
    opens = stripped.startswith(('"', "'"))
    closes = stripped.endswith(('"', "'"))
    if opens != closes:
        return False

    # Bulleted or explicitly upper-cased lines are strong indicators
    if re.match(r"^[\-\u2022*]\s", stripped):
        return True
    if stripped.isupper():
        return True

    # Headings often contain a colon or digits without terminal punctuation
    if ":" in stripped and not re.search(r"[.!?]$", stripped):
        return True

    # Short phrases ending with ! or ? are often enthusiastic or question
    # style headings that should accompany the following section.
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
            if re.match(rf"([{BULLET_CHARS_ESC}]|\d+[.)])\s", last_line):
                merged.append(part)
                continue
            if not any(_is_probable_heading(seg) for seg in (prev, part)):
                if _has_unbalanced_quotes(prev) and not _has_unbalanced_quotes(
                    prev + part
                ):
                    merged[-1] = f"{prev.rstrip()} {part.lstrip()}"
                    continue
                if len(prev) < 60 or not prev.rstrip().endswith((".", "?", "!")):
                    merged[-1] = f"{prev.rstrip()} {part.lstrip()}"
                    continue
        merged.append(part)
    return "\n\n".join(merged)


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
    if re.search(r"^[\",]", text.strip()):
        issues.append("Text starts with problematic punctuation")
    if re.search(r"[\",]$", text.strip()):
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
    if fixed.endswith(', "') or fixed.endswith(',"'):
        fixed = fixed[:-2]
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


def clean_paragraph(paragraph: str) -> str:
    """
    Cleans a single paragraph: removes mid-line hyphens, artifacts,
    collapses all newlines (if present) to spaces, and normalizes.
    """
    return pipe(
        paragraph,
        fix_hyphenated_linebreaks,
        collapse_artifact_breaks,
        remove_stray_bullet_lines,
        _preserve_list_newlines,
        normalize_quotes,
        remove_control_characters,
        consolidate_whitespace,
        normalize_ligatures,
        consolidate_whitespace,
    )


def clean_text(text: str) -> str:
    """
    Cleans multi-paragraph text, preserving paragraph breaks,
    using clean_paragraph() for each paragraph.
    """
    if not text or not text.strip():
        return ""

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
            from .pymupdf4llm_integration import (
                is_pymupdf4llm_available,
                clean_text_with_pymupdf4llm,
            )

            if is_pymupdf4llm_available():
                logger.debug(
                    "PyMuPDF4LLM is available, calling clean_text_with_pymupdf4llm"
                )
                result = clean_text_with_pymupdf4llm(text)
                logger.debug(f"PyMuPDF4LLM result preview: {repr(result[:100])}")
                return result
            else:
                logger.debug("PyMuPDF4LLM not available, falling back to traditional")
        except Exception as e:
            logger.debug(
                f"PyMuPDF4LLM cleaning failed with exception: {e}, falling back to traditional"
            )
            pass

    logger.debug("Using traditional text cleaning path")

    # Normalize newlines and fix broken words before other cleanup
    logger.debug("Calling normalize_newlines")
    text = normalize_newlines(text)
    logger.debug(f"After normalize_newlines: {repr(text[:100])}")

    logger.debug("Calling fix_hyphenated_linebreaks")
    text = fix_hyphenated_linebreaks(text)
    logger.debug(f"After fix_hyphenated_linebreaks: {repr(text[:100])}")

    logger.debug("Calling _fix_double_newlines")
    text = _fix_double_newlines(text)
    logger.debug(f"After _fix_double_newlines: {repr(text[:100])}")

    logger.debug("Calling insert_numbered_list_newlines")
    text = insert_numbered_list_newlines(text)
    logger.debug(f"After insert_numbered_list_newlines: {repr(text[:100])}")

    # Collapse single line breaks except paragraph breaks
    logger.debug("Calling collapse_single_newlines")
    text = collapse_single_newlines(text)
    logger.debug(f"After collapse_single_newlines: {repr(text[:100])}")

    logger.debug("Calling merge_spurious_paragraph_breaks")
    text = merge_spurious_paragraph_breaks(text)
    logger.debug(f"After merge_spurious_paragraph_breaks: {repr(text[:100])}")

    logger.debug("Calling _fix_split_words")
    text = _fix_split_words(text)
    logger.debug(f"After _fix_split_words: {repr(text[:100])}")

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
