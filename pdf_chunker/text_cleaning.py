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


def _sub_with_log(pattern: re.Pattern, repl: str, text: str, label: str) -> str:
    """Wrapper for re.sub that logs substitution count when non-zero."""
    new_text, count = pattern.subn(repl, text)
    if count:
        logger.debug(f"{label}: removed {count} occurrence{'s' if count > 1 else ''}")
    return new_text


# Patterns
PARAGRAPH_BREAK = re.compile(r"\n{2,}")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Quote normalization patterns
QUOTE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'(\w)"([A-Z])'), r'\1 "\2'),
    (re.compile(r'"(\w)'), r'" \1'),
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


BULLET_CHARS_ESC = re.escape("*•◦▪‣·●◉○‧")


def _join_broken_words(text: str) -> str:
    bullet_opt = rf"(?:[{BULLET_CHARS_ESC}]\s*)?"
    pattern_break = rf"(\w)[{HYPHEN_CHARS_ESC}]\s*\n\s*{bullet_opt}(\w)"
    pattern_space = rf"(\w)[{HYPHEN_CHARS_ESC}]\s+{bullet_opt}([a-z])"
    text = re.sub(pattern_break, r"\1\2", text)
    return re.sub(pattern_space, r"\1\2", text)


def _remove_soft_hyphens(text: str) -> str:
    return SOFT_HYPHEN_RE.sub("", text)


def fix_hyphenated_linebreaks(text: str) -> str:
    """Join words split across lines without removing valid hyphens."""

    return pipe(text, _join_broken_words, _remove_soft_hyphens)


def collapse_artifact_breaks(text: str) -> str:
    # Remove unwanted breaks after ., _, etc. (e.g., systems._\nThis → systems. This)
    return re.sub(r"([._])\n(\w)", r"\1 \2", text)


def collapse_single_newlines(text: str) -> str:
    logger.debug(f"collapse_single_newlines called with {len(text)} chars")
    logger.debug(f"Input text preview: {repr(text[:100])}")

    # First, protect paragraph breaks (2+ newlines) by replacing with placeholder
    text = re.sub(r"\n{2,}", "[[PARAGRAPH_BREAK]]", text)
    # Replace all remaining single newlines with spaces
    text = text.replace("\n", " ")
    # Restore paragraph breaks
    text = text.replace("[[PARAGRAPH_BREAK]]", "\n\n")

    logger.debug(f"Output text preview: {repr(text[:100])}")
    return text


def normalize_ligatures(text: str) -> str:
    return ftfy.fix_text(text)


def normalize_quotes(text: str) -> str:
    if not text:
        return text
    mapping = {"“": '"', "”": '"', "„": '"', "‚": '"', "‘": "'", "’": "'", "`": "'"}
    for smart, ascii_q in mapping.items():
        text = text.replace(smart, ascii_q)
    for pattern, repl in QUOTE_PATTERNS:
        text = pattern.sub(repl, text)
    return text


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
        if merged and not any(_is_probable_heading(seg) for seg in (merged[-1], part)):
            prev = merged[-1]
            if _has_unbalanced_quotes(prev) and not _has_unbalanced_quotes(prev + part):
                merged[-1] = f"{prev.rstrip()} {part.lstrip()}"
                continue
            if len(prev) < 60 or not prev.rstrip().endswith((".", "?", "!")):
                merged[-1] = f"{prev.rstrip()} {part.lstrip()}"
                continue
        merged.append(part)
    return "\n\n".join(merged)


# Any lowercase letter following a double newline likely means the break was
# introduced mid-sentence. Allow a handful of punctuation characters before the
# break so constructs like "words)\n\ncontinue" also collapse correctly.
PUNCT_BEFORE_BREAK = r"[A-Za-z0-9,;:\)\]\"'”’]"
SPURIOUS_DOUBLE_NL = re.compile(rf"(?<={PUNCT_BEFORE_BREAK})\n{{2}}(?=[a-z])")


def collapse_spurious_double_newlines(text: str) -> str:
    """Collapse double newlines that interrupt clauses."""

    return SPURIOUS_DOUBLE_NL.sub(" ", text)


BULLET_LINEBREAK_RE = re.compile(
    rf"(?<=[{HYPHEN_CHARS_ESC}])\n\s*[{BULLET_CHARS_ESC}]\s*(?=\w)"
)
LINE_START_BULLET_RE = re.compile(
    rf"(?<={PUNCT_BEFORE_BREAK})\n+\s*[{BULLET_CHARS_ESC}]\s*(?=\w)"
)
INLINE_BULLET_RE = re.compile(rf"(?<!\n)(?<!\A)\s*[{BULLET_CHARS_ESC}]\s*(?=\w)")


def collapse_inline_bullet_artifacts(text: str) -> str:
    """Remove stray bullet markers that interrupt sentences."""
    logger.debug("collapse_inline_bullet_artifacts invoked")
    text = _sub_with_log(LINE_START_BULLET_RE, " ", text, "line_start_bullet")
    text = _sub_with_log(BULLET_LINEBREAK_RE, " ", text, "bullet_linebreak")
    return _sub_with_log(INLINE_BULLET_RE, " ", text, "inline_bullet")


def _apply_steps(text: str, steps: List[Tuple[str, Callable[[str], str]]]) -> str:
    """Run cleaning steps sequentially with logging."""
    for name, fn in steps:
        logger.debug(f"Applying {name}")
        new_text = fn(text)
        if new_text != text:
            logger.debug(f"After {name}: {repr(new_text[:100])}")
        text = new_text
    return text


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
        lambda t: t.replace("\n", " "),  # any internal newlines to space
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

    text = _apply_steps(
        text,
        [
            ("normalize_newlines", normalize_newlines),
            ("collapse_single_newlines", collapse_single_newlines),
            ("merge_spurious_paragraph_breaks", merge_spurious_paragraph_breaks),
            ("collapse_spurious_double_newlines", collapse_spurious_double_newlines),
            ("collapse_inline_bullet_artifacts", collapse_inline_bullet_artifacts),
        ],
    )

    paragraphs = [p for p in PARAGRAPH_BREAK.split(text) if p.strip()]
    logger.debug(f"Split into {len(paragraphs)} paragraphs")
    result = "\n\n".join(clean_paragraph(p) for p in paragraphs)

    # Final JSON safety check
    safe, issues = validate_json_safety(result)
    if not safe:
        logger.warning(f"JSON safety issues detected: {issues}")
        result = apply_json_safety_fixes(result)

    logger.debug(f"Final clean_text result preview: {repr(result[:100])}")
    return result
