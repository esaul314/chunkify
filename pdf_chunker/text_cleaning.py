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
from contextlib import contextmanager
from contextvars import ContextVar
from functools import reduce
from typing import Callable, Iterator, List, Mapping, Match, Optional, Sequence, Tuple, TypeVar

import ftfy
from wordfreq import zipf_frequency

from pdf_chunker.page_artifacts import _drop_trailing_bullet_footers

logger = logging.getLogger(__name__)

_BULLET_TRACE_ENV = "PDF_CHUNKER_TRACE_BULLETS"
_BULLET_TRACE_TARGETS = frozenset({"sample_book-footer.pdf", "sample-local-pdf.pdf"})
_BULLET_TRACE_CONTEXT: ContextVar[Optional[dict[str, object]]] = ContextVar(
    "_BULLET_TRACE_CONTEXT",
    default=None,
)

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


def _stabilize(value: T, transform: Callable[[T], T], *, limit: int = 3) -> T:
    """Apply ``transform`` until a fixpoint is reached or ``limit`` iterations pass."""

    for _ in range(limit):
        updated = transform(value)
        if updated == value:
            return value
        value = updated
    return value


def _preview(s: str, n: int = PREVIEW_LEN) -> str:
    """Return a safe preview slice for debug logs."""
    return repr(s[:n])


def _should_trace_bullets(source: Optional[Mapping[str, object]]) -> bool:
    if not os.environ.get(_BULLET_TRACE_ENV):
        return False
    if not source:
        return False
    filename = source.get("filename")
    if not filename:
        return False
    return os.path.basename(str(filename)) in _BULLET_TRACE_TARGETS


@contextmanager
def bullet_trace_scope(
    source: Optional[Mapping[str, object]],
    *,
    stage: str,
    **metadata: object,
) -> Iterator[None]:
    if not _should_trace_bullets(source):
        yield
        return

    filename = os.path.basename(str(source.get("filename"))) if source else ""
    context = {
        "filename": filename,
        "stage": stage,
        **{key: value for key, value in metadata.items() if value is not None},
    }
    token = _BULLET_TRACE_CONTEXT.set(context)
    try:
        yield
    finally:
        _BULLET_TRACE_CONTEXT.reset(token)


def emit_bullet_trace(
    event: str,
    *,
    before: str,
    after: str,
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    context = _BULLET_TRACE_CONTEXT.get()
    if context is None:
        return

    payload: dict[str, object] = {
        "event": event,
        "before_preview": _preview(before),
        "after_preview": _preview(after),
        "changed": before != after,
        **context,
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if v is not None})

    logger.debug("bullet-trace %s", json.dumps(payload, ensure_ascii=False))


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

WINDOWS_1252_QUOTE_TRANSLATION = str.maketrans(
    {
        "\x82": "'",
        "\x91": "'",
        "\x92": "'",
        "\x93": '"',
        "\x94": '"',
    }
)
NBSP = "\u00a0"
_NBSP_EQUIVALENTS = {
    NBSP,
    "\u1680",  # ogham space mark
    "\u2000",  # en quad
    "\u2001",  # em quad
    "\u2002",  # en space
    "\u2003",  # em space
    "\u2004",  # three-per-em space
    "\u2005",  # four-per-em space
    "\u2006",  # six-per-em space
    "\u2007",  # figure space
    "\u2008",  # punctuation space
    "\u2009",  # thin space
    "\u200a",  # hair space
    "\u202f",  # narrow no-break space
    "\u205f",  # medium mathematical space
    "\u3000",  # ideographic space
}
_CIRCUMFLEX_WHITESPACE_ARTIFACT_RE = re.compile(r"(?:(?<=\s)|^)\u00C2(?=\s)")
_NBSP_TRANSLATION = str.maketrans({ch: " " for ch in _NBSP_EQUIVALENTS})

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
_ADDRESS_BULLET_PREFIXES: Tuple[str, ...] = (
    "directed to",
    "addressed to",
    "ship to",
    "shipped to",
    "sent to",
    "consigned to",
)
_ADDRESS_PREFIX_PATTERN = "|".join(map(re.escape, _ADDRESS_BULLET_PREFIXES))
ADDRESS_BULLET_RE = re.compile(
    rf"^[{BULLET_CHARS_ESC}]\s*(?:({_ADDRESS_PREFIX_PATTERN})\s+)?"
    r"[A-Z0-9][\w.'-]*(?:\s+[A-Z0-9][\w.'-]*)*,\s*"
    r"[A-Z][\w.'-]*(?:\s+[A-Za-z][\w.'-]*)*,\s*"
    r"[A-Z][A-Za-z.'-]+$",
    re.IGNORECASE,
)
_ADDRESS_TRAIT_RE = re.compile(
    r"(\d{1,6}\b|\b(?:street|st\.|road|rd\.|avenue|ave\.|boulevard|blvd\.|suite|ste\.|floor|fl\.|building|bldg|box)\b)",
    re.IGNORECASE,
)


def _should_strip_address_bullet(line: str) -> bool:
    match = ADDRESS_BULLET_RE.match(line)
    if not match:
        return False
    if match.group(1):
        return True
    return bool(_ADDRESS_TRAIT_RE.search(line))


BULLET_CONTINUATION_RE = re.compile(
    rf"((?:^|\n)\s*(?:[{BULLET_CHARS_ESC}]|-\s|\d+[.)])[^\n]*?)\n(?=\s*\S)(?!\s*(?:[{BULLET_CHARS_ESC}]|-\s|\d+[.)]))"
)

# Terminal punctuation for quoted sentence continuation
END_PUNCT = ".!?…"
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_CLOSING_QUOTE_CHARS = frozenset({'"', "'", "”", "’", "›", "»"})
_EM_DASH_CHARS = frozenset("—")
_STOPWORD_LOWERCASE_CHARS = frozenset(",;:'\"-")
_LIST_MARKER_RE = re.compile(rf"^\s*(?:[{BULLET_CHARS_ESC}]|-\s|\d+[.)])\s*")
_LEADING_WORD_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")

# Inline artifacts
# Avoid collapsing list markers like "2.\n3." by skipping digits after the break
COLLAPSE_ARTIFACT_BREAKS_RE = re.compile(r"([._])\n(?!\d)(\w)")
PIPE_RE = re.compile(r"\|")
UNDERSCORE_WRAP_RE = re.compile(
    r"(?<!\w)(?P<wrap>_{1,2})(?P<content>[^_]+?)(?P=wrap)(?!\w)"
)
_SIMPLE_LEADING_UNDERSCORE_RE = re.compile(
    r"(?<!\w)_(?P<word>[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*)"
)
_SIMPLE_TRAILING_UNDERSCORE_RE = re.compile(
    r"(?P<word>[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*)_+(?!\w)"
)
_PRESERVE_MULTIWORD_UNDERSCORE_RE = re.compile(
    r"(?<!\w)_(?P<content>[^_\n]*\s[^_]*)_(?!\w)"
)
DANGLING_UNDERSCORE_RE = re.compile(r"(?<!\w)_+|_+(?!\w)")

# Stray bullet variants
# Match stray bullet markers that occupy a full line, tolerating trailing spaces.
STRAY_BULLET_SOLO_RE = re.compile(rf"\n[{BULLET_CHARS_ESC}]\s*(?=\n|$)")
# Guard against collapsing legitimate list items (e.g., after colons)
STRAY_BULLET_AFTER_NEWLINE_RE = re.compile(
    rf"(?<![\n:{BULLET_CHARS_ESC}])\n[{BULLET_CHARS_ESC}]\s+(?=[a-z0-9])"
)
STRAY_BULLET_INLINE_RE = re.compile(rf"(?<=\S)[ \t][{BULLET_CHARS_ESC}]\s+(?=[a-z0-9])")


def _strip_address_bullet_line(line: str) -> str:
    """Remove bullet markers from address-style lines while preserving indentation."""

    stripped = line.lstrip()
    if not stripped:
        return line

    if _should_strip_address_bullet(stripped):
        prefix = line[: len(line) - len(stripped)]
        content = re.sub(rf"^[{BULLET_CHARS_ESC}]\s*", "", stripped)
        return f"{prefix}{content}"
    return line


def _strip_address_bullet_lines(text: str) -> str:
    return "\n".join(map(_strip_address_bullet_line, text.splitlines()))

# Numbered list helpers
NUMBER_SUFFIX_LINE_RE = re.compile(r"\n(\d+\.)(\s*)")
NUMBERED_AFTER_COLON_RE = re.compile(r":\s*(?!\n)(\d{1,3}[.)])")
NUMBERED_INLINE_CANDIDATE_RE = re.compile(
    r"(\d{1,3}[.)](?:[^\n]|\n(?!\n|\d))*?)\s+(?=(\d{1,3}[.)]))"
)
NUMBERED_END_RE = re.compile(
    rf"(\d{{1,3}}[.)][^\n]*[{re.escape(END_PUNCT)}])"
    rf"(?<![{re.escape(END_PUNCT)}]\")"
    rf"(?=\n(?:\s*(?:[{BULLET_CHARS_ESC}]|\d+[.)])|\n|$))"
)
NUMBERED_CONTINUATION_RE = re.compile(rf"(\d{{1,3}}[.)][^\n]*[{re.escape(END_PUNCT)}])\n(?=[^\n])")

# List break preservation
LIST_BREAK_RE = re.compile(rf"\n(?=\s*(?:[{BULLET_CHARS_ESC}]|-\s|\d+[.)]|.*\.\.))")
LIST_BREAK_SPAN_RE = re.compile(
    rf"(?P<break>\n{{1,}})(?=(?P<context>\s*(?:[{BULLET_CHARS_ESC}]|-\s|\d+[.)]|.*\.\.)))"
)
LIST_BREAK_SENTINEL_RE = re.compile(r"\[\[LIST_BREAK_(?P<idx>\d+)\]\]")
COLON_BULLET_START_RE = re.compile(rf":\s*(?=-|[{BULLET_CHARS_ESC}])")

# Newline/split heuristics
DOUBLE_NEWLINE_RE = re.compile(r"([A-Za-z]+)\n{2,}\s*([a-z][A-Za-z]+)")
COLON_LIST_BREAK_RE = re.compile(r":\n{2,}(?=\s*(?:[•\-]|\d))")


def _guarded_sub(pattern: re.Pattern[str], repl: str, text: str) -> str:
    """Apply ``pattern.sub`` only when the pattern appears."""

    return pattern.sub(repl, text) if pattern.search(text) else text
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
        "when",
        "then",
        "up",
    }
)

_BULLET_STOPWORD_TITLES = tuple(sorted({word.title() for word in STOPWORDS}))
_BULLET_STOPWORD_PATTERN = re.compile(rf"\b({'|'.join(_BULLET_STOPWORD_TITLES)})\b")

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
HYPHEN_BREAK_RE = re.compile(
    rf"([A-Za-z]+)[{HYPHEN_CHARS_ESC}]\s*\n\s*{_HYPHEN_BULLET_OPT}([A-Za-z]+)"
)
HYPHEN_SPACE_RE = re.compile(rf"([A-Za-z]+)[{HYPHEN_CHARS_ESC}]\s+{_HYPHEN_BULLET_OPT}([A-Za-z]+)")

_LINEBREAK_HYPHEN_MARGIN = 1.5
_LINEBREAK_LOWER_SUFFIXES = frozenset(
    {
        "ing",
        "ings",
        "ed",
        "er",
        "ers",
        "est",
        "ly",
        "ally",
        "ment",
        "ments",
        "ness",
        "less",
        "ful",
        "ity",
        "ities",
        "ous",
        "ive",
        "tion",
        "tions",
        "sion",
        "sions",
        "able",
        "ible",
        "ance",
        "ances",
        "ence",
        "ences",
        "ism",
        "ist",
        "ists",
        "ship",
        "ships",
        "hood",
        "out",
    }
)

_LINEBREAK_TITLE_JOIN_HEADS = frozenset(
    word.title()
    for word in (
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    )
)


# ---------------------------------------------------------------------------
# Hyphenation and word glue fixes
# ---------------------------------------------------------------------------


def _hyphenation_scores(head: str, tail: str) -> Tuple[str, str, float, float]:
    """Return join candidates and their corpus frequencies."""

    joined = head + tail
    hyphenated = f"{head}-{tail}"
    joined_freq = zipf_frequency(joined.lower(), "en")
    hyphen_freq = zipf_frequency(hyphenated.lower(), "en")
    return joined, hyphenated, joined_freq, hyphen_freq


def _choose_hyphenation(head: str, tail: str) -> str:
    """Return the most plausible join of ``head`` and ``tail``.

    Prefers the hyphenated form when its corpus frequency is at least as high
    as the unhyphenated join. This preserves legitimate hyphenated compounds
    while still repairing line-break hyphenation artifacts.
    """

    joined, hyphenated, joined_freq, hyphen_freq = _hyphenation_scores(head, tail)
    return hyphenated if hyphen_freq > joined_freq else joined


def _linebreak_prefers_join(
    head: str, tail: str, joined_freq: float, token: str, prefix: str
) -> bool:
    """Return ``True`` when newline hyphenation hints at a simple join."""

    suffix = token.split("\n", 1)[-1]
    shares_hyphen = any(char in _HYPHEN_CHARS for char in suffix)
    resumes_lower = tail[:1].islower()
    head_is_title = head in _LINEBREAK_TITLE_JOIN_HEADS
    joined_missing = joined_freq <= 0
    return shares_hyphen or (
        not prefix.strip()
        and resumes_lower
        and head_is_title
        and joined_missing
    )


def _should_keep_linebreak_hyphen(
    head: str, tail: str, joined_freq: float, hyphen_freq: float
) -> bool:
    """Decide whether a newline-spanning hyphen should be preserved."""

    if joined_freq <= 0:
        return True
    if len(head) <= 2 or len(tail) <= 2:
        return hyphen_freq > joined_freq
    return hyphen_freq - joined_freq >= _LINEBREAK_HYPHEN_MARGIN


def _normalize_linebreak_join_case(head: str, tail: str, joined: str) -> str:
    """Lower-case obvious continuations after dropping a hyphen."""

    if not tail or not tail.isalpha():
        return joined
    if tail == tail.lower():
        return joined
    tail_lower = tail.lower()
    prefix = joined[: -len(tail)] if len(tail) < len(joined) else ""
    if head.islower():
        return prefix + tail_lower
    if tail_lower in _LINEBREAK_LOWER_SUFFIXES:
        return prefix + tail_lower
    return joined


def _join_hyphenated_words(text: str) -> str:
    """Merge words broken with hyphenation across line breaks."""

    def _hyphen_from_token(token: str) -> str:
        return next((char for char in token if char in _HYPHEN_CHARS), "-")

    def _replace_with_original(joined: str, hyphen: str) -> str:
        return joined.replace("-", hyphen, 1) if "-" in joined else joined

    def choose_for_break(match: Match[str]) -> str:
        head, tail = match.group(1), match.group(2)
        hyphen = _hyphen_from_token(match.group(0))
        joined, hyphenated, joined_freq, hyphen_freq = _hyphenation_scores(head, tail)
        prefer_join = _linebreak_prefers_join(
            head, tail, joined_freq, match.group(0), match.string[: match.start()]
        )
        if hyphen_freq > joined_freq and _should_keep_linebreak_hyphen(
            head, tail, joined_freq, hyphen_freq
        ):
            return (
                _normalize_linebreak_join_case(head, tail, joined)
                if prefer_join
                else _replace_with_original(hyphenated, hyphen)
            )
        return _normalize_linebreak_join_case(head, tail, joined)

    def choose_hyphenation(match: Match[str]) -> str:
        head, tail = match.group(1), match.group(2)
        joined, hyphenated, joined_freq, hyphen_freq = _hyphenation_scores(head, tail)
        if hyphen_freq > joined_freq and _should_keep_linebreak_hyphen(
            head, tail, joined_freq, hyphen_freq
        ):
            hyphen = _hyphen_from_token(match.group(0))
            return hyphenated.replace("-", hyphen, 1)
        return _normalize_linebreak_join_case(head, tail, joined)

    return pipe(
        text,
        lambda value: HYPHEN_BREAK_RE.sub(choose_for_break, value),
        lambda value: HYPHEN_SPACE_RE.sub(choose_hyphenation, value),
    )


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


def _collapse_colon_list_breaks(text: str) -> str:
    """Collapse paragraph gaps after colons that precede list markers."""

    return _guarded_sub(COLON_LIST_BREAK_RE, ":\n", text)


def _fix_double_newlines(text: str) -> str:
    """Resolve words or phrases separated by double newlines using word heuristics."""

    normalized = _collapse_colon_list_breaks(text)
    return DOUBLE_NEWLINE_RE.sub(
        lambda m: _maybe_join_words(m.group(1), m.group(2)), normalized
    )


def _fix_split_words(text: str) -> str:
    """Join words erroneously split by whitespace using word heuristics."""
    return SPLIT_WORD_RE.sub(lambda m: _maybe_join_words(m.group(1), m.group(2)), text)


# ---------------------------------------------------------------------------
# Footer clean-up
# ---------------------------------------------------------------------------


def _remove_trailing_bullet_footers(text: str) -> str:
    """Prune footer-style bullet runs at the end of paragraphs."""

    if not text:
        return text

    paragraphs = (
        tuple(filter(None, text.split("\n\n")))
        if "\n" in text
        else (text,)
    )
    pruned = (
        "\n".join(_drop_trailing_bullet_footers(paragraph.split("\n")))
        for paragraph in paragraphs
    )
    compact = tuple(paragraph for paragraph in pruned if paragraph)
    return "\n\n".join(compact)


# ---------------------------------------------------------------------------
# Quote & ligature normalization
# ---------------------------------------------------------------------------


def _map_smart_quotes(text: str) -> str:
    """Map smart quotes to their ASCII equivalents."""
    return "".join(SMART_QUOTES.get(ch, ch) for ch in text)


def _fix_quote_spacing(text: str) -> str:
    """Apply spacing and duplication fixes around quotes."""
    return reduce(lambda s, p: p[0].sub(p[1], s), QUOTE_SPACING_PATTERNS, text)


def _normalize_windows_1252_quote_bytes(text: str) -> str:
    """Replace stray Windows-1252 quote bytes with ASCII fallbacks."""

    return text.translate(WINDOWS_1252_QUOTE_TRANSLATION) if text else text


def normalize_windows_1252_quotes(text: str) -> str:
    """Public wrapper ensuring callers share the same byte normalization."""

    return _normalize_windows_1252_quote_bytes(text)


def _normalize_circumflex_whitespace_artifacts(text: str) -> str:
    """Collapse stray ``Â`` prefixes that merely guard whitespace artifacts."""

    return (
        _CIRCUMFLEX_WHITESPACE_ARTIFACT_RE.sub(" ", text)
        if "\u00C2" in text
        else text
    )


def normalize_non_breaking_spaces(text: str) -> str:
    """Collapse NBSP-style separators to regular spaces for stable downstream handling."""

    if not text or not any(ch in text for ch in _NBSP_EQUIVALENTS):
        return text
    return text.translate(_NBSP_TRANSLATION)


def normalize_quotes(text: str) -> str:
    """Normalize smart quotes and repair missing surrounding spaces."""
    return (
        text
        if not text
        else pipe(text, normalize_windows_1252_quotes, _map_smart_quotes, _fix_quote_spacing)
    )


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


_TRAILING_NUMBER_EXCEPTIONS = frozenset(
    {"chapter", "section", "part", "figure", "table", "appendix"}
)
_TRAILING_NUMBER_MARKER_RE = re.compile(
    r"(?P<prefix>\b[A-Za-z]+)(?P<sep>\s+)(?P<marker>\d+[.)](?:['\"\u2019])?)(?P<trail>\s*)$",
    re.IGNORECASE,
)


def _strip_trailing_number_marker(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        if prefix and prefix.lower() in _TRAILING_NUMBER_EXCEPTIONS:
            return match.group(0)
        return f"{prefix}{match.group('trail')}"

    return _TRAILING_NUMBER_MARKER_RE.sub(_replace, text)


def drop_spurious_number_markers(text: str) -> str:
    """Remove numbered markers created by hyphenated splits or isolated lines."""
    return pipe(
        text,
        lambda t: re.sub(r"-\n\d+[.)]\s*", "-", t),
        lambda t: re.sub(r"\n\d+[.)]\n", "\n", t),
        lambda t: re.sub(r"\n\d+[.)]\s+(?=[A-Z]{2,}\b)", "\n", t),
        lambda t: re.sub(r"(?<=\b[a-z])\s+\d+[.)]\s+(?=[a-z])", " ", t),
        _strip_trailing_number_marker,
        lambda t: re.sub(r"\n{2,}(\d+[.)])", r"\n\1", t),
    )


def _remove_stray_bullet_lines_once(text: str) -> str:
    """Perform a single pass of stray bullet cleanup."""

    return pipe(
        text,
        lambda t: STRAY_BULLET_SOLO_RE.sub("\n", t),
        lambda t: STRAY_BULLET_AFTER_NEWLINE_RE.sub(" ", t),
        lambda t: STRAY_BULLET_INLINE_RE.sub(" ", t),
        _strip_address_bullet_lines,
        lambda t: re.sub(rf"\n+(?=[{BULLET_CHARS_ESC}])", "\n", t),
    )


def remove_stray_bullet_lines(text: str) -> str:
    """Collapse stray bullet markers while keeping legitimate list breaks intact."""

    cleaned = _stabilize(text, _remove_stray_bullet_lines_once)
    emit_bullet_trace(
        "remove_stray_bullet_lines",
        before=text,
        after=cleaned,
        extra={"stabilized": cleaned != text},
    )
    return cleaned


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
    return pipe(
        text,
        lambda t: NUMBERED_AFTER_COLON_RE.sub(r":\n\1", t),
        _apply_inline_numbered,
        lambda t: NUMBERED_END_RE.sub(r"\1\n\n", t),
        lambda t: NUMBERED_CONTINUATION_RE.sub(r"\1[[LIST_BREAK]]", t),
    )


def _preserve_list_newlines(text: str) -> str:
    """Keep newlines that precede bullets or enumerated items."""

    if "\n" not in text:
        return text

    preserved: frozenset[int] = frozenset(match.start() for match in LIST_BREAK_RE.finditer(text))
    return "".join(
        "\n" if index in preserved else (" " if char == "\n" else char)
        for index, char in enumerate(text)
    )


ListBreakSentinel = Tuple[str, str, str]


def _capture_list_break_sentinels(text: str) -> Tuple[str, Tuple[ListBreakSentinel, ...]]:
    """Protect list breaks by swapping them for unique sentinel tuples."""

    sentinels: List[ListBreakSentinel] = []

    def _store(match: Match[str]) -> str:
        token = f"[[LIST_BREAK_{len(sentinels)}]]"
        sentinels.append((token, match.group("break"), match.group("context")))
        return token

    protected = LIST_BREAK_SPAN_RE.sub(_store, text)
    return protected, tuple(sentinels)


def _restore_list_breaks(text: str, sentinels: Tuple[ListBreakSentinel, ...]) -> str:
    """Rebuild previously protected list breaks using a generator pipeline."""

    if not sentinels:
        return text

    sentinel_map = {token: (break_text, context) for token, break_text, context in sentinels}

    def _segments() -> Iterator[str]:
        last = 0
        for match in LIST_BREAK_SENTINEL_RE.finditer(text):
            segment = text[last:match.start()]
            token = match.group(0)
            break_text, context = sentinel_map[token]
            following = text[match.end():]
            prefix = segment.rstrip() if following.startswith(context) else segment
            yield prefix
            yield "\n"
            if len(break_text) == 2:
                yield "\n"
            last = match.end()
        yield text[last:]

    return "".join(_segments())


def _should_lowercase_bullet_stopword(prefix: str) -> bool:
    tokens = tuple(_TOKEN_RE.findall(prefix))
    if not tokens:
        last_char = prefix[-1]
        return last_char.islower() or last_char in _STOPWORD_LOWERCASE_CHARS

    colon_seen = False
    for token in reversed(tokens):
        if token in _CLOSING_QUOTE_CHARS:
            return False
        if token in _EM_DASH_CHARS:
            return False
        if token == ":":
            colon_seen = True
            continue
        if any(char in END_PUNCT for char in token):
            return False
        if colon_seen:
            return False
        last_char = token[-1]
        return last_char.islower() or last_char in _STOPWORD_LOWERCASE_CHARS

    return False


def _normalize_bullet_stopword_case(text: str) -> str:
    """Lowercase stopwords restored mid-sentence inside bullet items."""

    def _normalize_line(line: str) -> str:
        stripped = line.lstrip()
        if not stripped or not re.match(
            rf"(?:[{BULLET_CHARS_ESC}]|-\s|\d+[.)])", stripped
        ):
            return line

        def _replace(match: Match[str]) -> str:
            prefix = line[: match.start()]
            trimmed = prefix.rstrip()
            if not trimmed:
                return match.group(0)

            return (
                match.group(0).lower()
                if _should_lowercase_bullet_stopword(trimmed)
                else match.group(0)
            )

        return _BULLET_STOPWORD_PATTERN.sub(_replace, line)

    return "\n".join(_normalize_line(line) for line in text.splitlines())


def normalize_bullet_stopwords(text: str) -> str:
    """Public helper wrapping bullet stopword normalization."""

    return _normalize_bullet_stopword_case(text)


def _join_bullet_wrapped_lines(text: str) -> str:
    """Collapse intra-item bullet newlines into spaces."""

    def _replace(match: Match[str]) -> str:
        follower = match.string[match.end()] if match.end() < len(match.string) else ""
        return match.group(0) if follower == "\n" else f"{match.group(1)} "

    if not BULLET_CONTINUATION_RE.search(text):
        return _normalize_bullet_stopword_case(text)

    return pipe(
        text,
        lambda value: BULLET_CONTINUATION_RE.sub(_replace, value),
        _normalize_bullet_stopword_case,
    )

  
def collapse_single_newlines(text: str) -> str:
    logger.debug(f"collapse_single_newlines called with {len(text)} chars")
    logger.debug(f"Input text preview: {_preview(text)}")

    para_break = "[[PARAGRAPH_BREAK]]"

    normalized = pipe(
        text,
        lambda value: _guarded_sub(COLON_BULLET_START_RE, ":\n", value),
        merge_number_suffix_lines,
    )
    protected, sentinels = _capture_list_break_sentinels(normalized)

    bullet_joined = pipe(
        protected,
        _join_bullet_wrapped_lines,
    )

    flattened = pipe(
        bullet_joined,
        lambda t: PARAGRAPH_BREAK.sub(para_break, t),
        lambda t: t.replace("\n", " "),
        lambda t: t.replace(para_break, "\n\n"),
        lambda t: t.replace("[[LIST_BREAK]]", "\n\n"),
    )

    result = pipe(
        flattened,
        lambda value: _restore_list_breaks(value, sentinels),
        _join_bullet_wrapped_lines,
        _fix_quote_spacing,
    )

    logger.debug(f"Output text preview: {_preview(result)}")
    return result


# ---------------------------------------------------------------------------
# Heading/list detection & footnotes
# ---------------------------------------------------------------------------


def _starts_list_item(line: str) -> bool:
    """Return True when ``line`` begins with a bullet or numbered marker."""

    return bool(_LIST_MARKER_RE.match(line))


def _starts_new_list_item(text: str) -> bool:
    return _starts_list_item(text.lstrip())


def _split_heading_words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", text) if w]


def _heading_length_guard(stripped: str) -> bool:
    return bool(stripped) and len(stripped) <= 80 and not stripped.endswith("|")


def _has_balanced_edge_quotes(stripped: str) -> bool:
    opens = stripped.startswith(('"', "'"))
    closes = stripped.endswith(('"', "'"))
    return opens == closes


def _is_wrapped_in_quotes(stripped: str) -> bool:
    return stripped.startswith(('"', "'")) and stripped.endswith(('"', "'"))


def _heading_case_signal(words: Sequence[str], stripped: str) -> bool:
    if _is_wrapped_in_quotes(stripped):
        return False
    alpha_words = [w for w in words if w and w[0].isalpha()]
    return bool(alpha_words) and sum(w[0].isupper() for w in alpha_words) / len(alpha_words) >= 0.5


def _heading_colon_signal(stripped: str, words: Sequence[str], has_terminal_punct: bool) -> bool:
    if ":" not in stripped or has_terminal_punct:
        return False
    trailing = stripped.rstrip("\"'")
    if trailing.endswith(":"):
        return True
    if stripped.count(":") != 1 or _is_wrapped_in_quotes(stripped):
        return False
    prefix, suffix = (segment.strip() for segment in stripped.split(":", 1))
    if not prefix or not suffix:
        return False
    segment_word_counts = tuple(len(_split_heading_words(segment)) for segment in (prefix, suffix))
    if not segment_word_counts:
        return False
    total_words = len(words)
    return total_words <= 8 and max(segment_word_counts) <= 4


def _is_short_title_phrase(words: Sequence[str], stripped: str, has_terminal_punct: bool) -> bool:
    return 1 < len(words) <= 3 and stripped[0].isupper() and not has_terminal_punct


def _is_probable_heading(text: str) -> bool:
    """Heuristically determine whether a line looks like a heading."""
    stripped = text.strip()
    if not _heading_length_guard(stripped):
        return False

    words = _split_heading_words(stripped)
    if not words:
        return False

    if not _has_balanced_edge_quotes(stripped):
        return False

    has_terminal_punct = bool(re.search(r"[.!?]$", stripped))
    ends_with_excited = bool(re.search(r"[!?]$", stripped))

    if _is_short_title_phrase(words, stripped, has_terminal_punct):
        return True

    if re.match(r"^[\-\u2022*]\s", stripped):
        return True
    if stripped.isupper():
        return True

    if _heading_colon_signal(stripped, words, has_terminal_punct):
        return True

    if ends_with_excited and 1 < len(words) <= 6 and len(stripped) <= 60:
        return True

    if _heading_case_signal(words, stripped) and not has_terminal_punct:
        return True

    return False


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


def _first_alpha_index(text: str) -> Optional[int]:
    for idx, char in enumerate(text):
        if char.isalpha():
            return idx
    return None


def _leading_word(segment: str) -> Optional[str]:
    match = _LEADING_WORD_RE.match(segment)
    return match.group(0) if match else None


def _should_skip_capitalization(word: str) -> bool:
    return not word.islower() or any(char.isdigit() for char in word)


def restore_leading_capitalization(text: str) -> str:
    """Uppercase the first alphabetic character when the leading word is lowercase."""

    if not text:
        return text

    offset = len(text) - len(text.lstrip())
    remainder = text[offset:]

    marker_match = _LIST_MARKER_RE.match(remainder)
    if marker_match:
        offset += marker_match.end()
        remainder = remainder[marker_match.end():]

    alpha_offset = _first_alpha_index(remainder)
    if alpha_offset is None:
        return text

    absolute = offset + alpha_offset
    current = text[absolute]
    if not current.islower():
        return text

    word = _leading_word(remainder[alpha_offset:])
    if not word or _should_skip_capitalization(word):
        return text

    return f"{text[:absolute]}{current.upper()}{text[absolute + 1:]}"


def remove_underscore_emphasis(text: str) -> str:
    """Remove single/double underscore emphasis markers."""

    def _unwrap(match: Match[str]) -> str:
        content = match.group("content")
        should_strip = (
            bool(content)
            and content[0].isalnum()
            and content[-1].isalnum()
            and all(ch.isalnum() or ch in "-'’" for ch in content)
        )
        return content if should_strip else match.group(0)

    def _strip_leading(match: Match[str]) -> str:
        word = match.group("word")
        if word and (word[0].islower() or word[0].isdigit()):
            return word
        return match.group(0)

    def _strip_trailing(match: Match[str]) -> str:
        word = match.group("word")
        if word and (word[0].islower() or word[0].isdigit()):
            return word
        return match.group(0)

    stripped = UNDERSCORE_WRAP_RE.sub(_unwrap, text)
    stripped = _SIMPLE_LEADING_UNDERSCORE_RE.sub(_strip_leading, stripped)
    return _SIMPLE_TRAILING_UNDERSCORE_RE.sub(_strip_trailing, stripped)


def strip_underscore_wrapping(text: str) -> str:
    """Public helper that removes underscore emphasis wrappers."""
    return remove_underscore_emphasis(text)


def _remove_dangling_underscores_once(text: str) -> str:
    """Return ``text`` with a single pass of dangling underscore removal applied."""

    preserved_ranges = tuple(
        (match.start(), match.end())
        for match in _PRESERVE_MULTIWORD_UNDERSCORE_RE.finditer(text)
    )

    def _is_preserved(index: int) -> bool:
        return any(start <= index < end for start, end in preserved_ranges)

    def _should_keep(start: int, end: int) -> bool:
        if _is_preserved(start):
            return True
        following = text[end] if end < len(text) else ""
        return bool(following and following.isupper())

    def _segments() -> Iterator[str]:
        last = 0
        for match in DANGLING_UNDERSCORE_RE.finditer(text):
            start, end = match.span()
            yield text[last:start]
            if _should_keep(start, end):
                yield match.group(0)
            last = end
        yield text[last:]

    return "".join(_segments())


def remove_dangling_underscores(text: str) -> str:
    """Remove underscores that don't join word characters."""

    return _stabilize(text, _remove_dangling_underscores_once)


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
            if _ends_with_footnote(prev):
                normalized = _normalize_trailing_footnote(prev)
                merged[-1] = normalized
                prev = normalized
                if not _starts_new_list_item(part):
                    merged[-1] = f"{normalized} {part.lstrip()}"
                    continue
            last_line = prev.strip().splitlines()[-1]
            if _starts_list_item(last_line):
                merged.append(part)
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
    def _strip_control(text: str) -> str:
        return CONTROL_CHARS.sub("", text) if CONTROL_CHARS.search(text) else text

    def _strip_problematic_prefix(value: str) -> str:
        return pipe(
            value,
            lambda current: current[3:] if current.startswith('", ') else current,
            lambda current: (
                current[1:]
                if current.startswith('"') and len(current) > 1 and current[1].islower()
                else current
            ),
        )

    def _ensure_utf8(value: str) -> str:
        try:
            value.encode("utf-8")
            return value
        except UnicodeError:
            try:
                return value.encode("utf-8", errors="replace").decode("utf-8")
            except UnicodeError:
                return "".join(ch for ch in value if ord(ch) < 128)

    def _rebalance_quotes(value: str) -> str:
        if value.count('"') % 2 == 0:
            return value
        if value.endswith('"'):
            return value[:-1]
        if value.startswith('"'):
            return value[1:]
        return value

    def _trim_if_needed(value: str) -> str:
        trimmed = value.strip()
        return trimmed if trimmed != value else value

    return pipe(text, _strip_control, _strip_problematic_prefix, _ensure_utf8, _rebalance_quotes, _trim_if_needed)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def _normalize_underscore_artifacts(text: str) -> str:
    """Remove emphasis and dangling underscores in a single functional pass."""

    return pipe(text, remove_underscore_emphasis, remove_dangling_underscores)


def clean_paragraph(paragraph: str) -> str:
    """
    Cleans a single paragraph: removes mid-line hyphens, artifacts,
    collapses all newlines (if present) to spaces, and normalizes.
    """
    return pipe(
        paragraph,
        _normalize_underscore_artifacts,
        rejoin_hyphenated_words,
        strip_headers_and_footers,
        replace_pipes,
        collapse_artifact_breaks,
        cleanup_bullet_fragments,
        _preserve_list_newlines,
        remove_control_characters,
        normalize_ligatures,
        _fix_quote_spacing,
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

    logger.debug("Calling normalize_windows_1252_quotes")
    text = normalize_windows_1252_quotes(text)
    logger.debug(f"After normalize_windows_1252_quotes: {_preview(text)}")

    logger.debug("Calling _normalize_circumflex_whitespace_artifacts")
    text = _normalize_circumflex_whitespace_artifacts(text)
    logger.debug(
        f"After _normalize_circumflex_whitespace_artifacts: {_preview(text)}"
    )

    logger.debug("Calling normalize_non_breaking_spaces")
    text = normalize_non_breaking_spaces(text)
    logger.debug(f"After normalize_non_breaking_spaces: {_preview(text)}")

    # Normalize newlines and fix broken words before other cleanup
    logger.debug("Calling normalize_newlines")
    text = normalize_newlines(text)
    logger.debug(f"After normalize_newlines: {_preview(text)}")

    logger.debug("Calling _normalize_underscore_artifacts")
    text = _normalize_underscore_artifacts(text)
    logger.debug(f"After _normalize_underscore_artifacts: {_preview(text)}")

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
    result = _remove_trailing_bullet_footers(result)

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
    "normalize_windows_1252_quotes",
    "normalize_non_breaking_spaces",
    "remove_underscore_emphasis",
    "strip_underscore_wrapping",
    "remove_dangling_underscores",
    "normalize_bullet_stopwords",
    "normalize_newlines",
    "remove_control_characters",
    "consolidate_whitespace",
    "merge_spurious_paragraph_breaks",
    "validate_json_safety",
    "apply_json_safety_fixes",
    "strip_headers_and_footers",
    "clean_paragraph",
    "clean_text",
    "restore_leading_capitalization",
]
