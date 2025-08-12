import logging
import re
from functools import reduce
from itertools import takewhile
from typing import Optional

from .text_cleaning import clean_text, pipe

ROMAN_RE = r"[ivxlcdm]+"
_ROMAN_MAP = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
DOMAIN_RE = re.compile(r"\b[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)

SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_MAP = str.maketrans(SUPERSCRIPT_DIGITS, "0123456789")
_SUP_DIGITS_ESC = re.escape(SUPERSCRIPT_DIGITS)


def _roman_to_int(value: str) -> int:
    """Convert a Roman numeral to an integer."""

    def _step(acc: tuple[int, int], ch: str) -> tuple[int, int]:
        total, prev = acc
        val = _ROMAN_MAP.get(ch, 0)
        return (total - val, prev) if val < prev else (total + val, val)

    return reduce(_step, reversed(value.lower()), (0, 0))[0]


def _looks_like_footnote(text: str) -> bool:
    """Return ``True`` if ``text`` resembles a footnote line."""

    stripped = text.strip()
    if len(stripped.split()) < 3:
        return False

    if stripped.lower().startswith("footnote"):
        return True

    pattern = rf"^(?:[0-9{_SUP_DIGITS_ESC}]{{1,3}}|[\*\u2020])\s+[A-Z]"
    return bool(re.match(pattern, stripped))


def _starts_with_multiple_numbers(text: str) -> bool:
    """Return ``True`` if ``text`` begins with two or more numbers."""

    parts = text.strip().split()
    return sum(1 for p in takewhile(str.isdigit, parts)) >= 2


def _contains_domain(text: str) -> bool:
    """Return ``True`` if ``text`` contains a domain-like pattern."""

    return bool(DOMAIN_RE.search(text))


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _match_common_patterns(text_lower: str) -> bool:
    """Return True if text matches common header/footer patterns."""
    patterns = [
        r"^\d+$",
        r"^page\s+\d+",
        r"^\d+\s*$",
        r"^chapter\s+\d+$",
        r"^\d+\s+chapter",
        r"^\w+\s*\|\s*\d+$",
        rf"^\w+\s*\|\s*{ROMAN_RE}$",
        r"^\d+\s*\|\s*[\w\s:]+$",
        rf"^{ROMAN_RE}$",
        r"^[0-9]{1,3}[.)]?\s+[A-Z]",
        r"^table\s+of\s+contents",
        r"^bibliography",
        r"^index$",
        r"^appendix\s+[a-z]$",
        r"^[a-z][^|]{0,60}\|$",
    ]
    return any(re.match(p, text_lower) for p in patterns)


def _normalize_page_num(page_num: Optional[int]) -> int:
    """Return a safe page number for comparisons."""

    return page_num if isinstance(page_num, int) else 0


def _match_page_number_suffix(text: str, page_num: Optional[int]) -> bool:
    """Detect page-number fragments at line ends or near the end.

    If ``page_num`` is ``0`` or negative, the numeric check is skipped.
    """

    # Exact trailing page number
    page_num = _normalize_page_num(page_num)

    m = re.search(r"(\d{1,3})\s*$", text)
    if m:
        trailing = int(m.group(1))
        if page_num <= 0 or abs(trailing - page_num) <= 1:
            words = text.split()
            if "|" in text or len(words) <= 8:
                logger.debug(
                    "_match_page_number_suffix trailing digits detected: %s",
                    text[:30],
                )
                return True

    # Page number followed by stray characters from the next line
    m = re.search(r"\|\s*(\d{1,3})(?!\d)", text)
    if m:
        trailing = int(m.group(1))
        if (page_num <= 0 or abs(trailing - page_num) <= 1) and len(
            text
        ) - m.end() <= 20:
            logger.debug(
                "_match_page_number_suffix pipe pattern detected: %s", text[:30]
            )
            return True

    m = re.search(rf"(?:^|\s|\|)({ROMAN_RE})\s*$", text, re.IGNORECASE)
    if m:
        trailing = _roman_to_int(m.group(1))
        if page_num <= 0 or abs(trailing - page_num) <= 1:
            words = text.split()
            if "|" in text or len(words) <= 8:
                logger.debug(
                    "_match_page_number_suffix trailing roman detected: %s",
                    text[:30],
                )
                return True

    m = re.search(rf"\|\s*({ROMAN_RE})(?![0-9ivxlcdm])", text, re.IGNORECASE)
    if m:
        trailing = _roman_to_int(m.group(1))
        if (page_num <= 0 or abs(trailing - page_num) <= 1) and len(
            text
        ) - m.end() <= 20:
            logger.debug(
                "_match_page_number_suffix roman pipe pattern detected: %s",
                text[:30],
            )
            return True

    return False


def is_page_artifact_text(text: str, page_num: Optional[int]) -> bool:
    """Return True if the text looks like a header or footer artifact."""
    text_lower = text.lower().strip()
    if not text_lower:
        return True

    if _match_common_patterns(text_lower):
        logger.info(f"is_page_artifact_text() pattern match: {text[:30]}…")
        return True

    if _match_page_number_suffix(text, page_num):
        logger.info(
            "is_page_artifact_text() page number suffix: %s (page %s)",
            text[:30],
            page_num,
        )
        return True

    if _starts_with_multiple_numbers(text_lower):
        logger.info(
            "is_page_artifact_text() multiple leading numbers: %s",
            text[:30],
        )
        return True

    if _contains_domain(text) and len(text.split()) <= 8:
        logger.info(
            "is_page_artifact_text() domain footer detected: %s",
            text[:30],
        )
        return True

    if _looks_like_footnote(text):
        logger.info("is_page_artifact_text() footnote detected: %s", text[:30])
        return True

    if (
        len(text.split()) <= 3
        and len(text) <= 30
        and any(char.isdigit() for char in text)
    ):
        return True

    return False


def strip_page_artifact_suffix(text: str, page_num: Optional[int]) -> str:
    """Return the line with any trailing ``"| N"`` footer fragment removed."""

    pattern = re.compile(rf"\|\s*(\d{{1,3}}|{ROMAN_RE})(?![0-9ivxlcdm])", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return text

    page_str = match.group(1)
    trailing = int(page_str) if page_str.isdigit() else _roman_to_int(page_str)
    page_num = _normalize_page_num(page_num)
    if (page_num <= 0 or abs(trailing - page_num) <= 1) and len(
        text
    ) - match.end() <= 20:
        logger.info("strip_page_artifact_suffix removed footer fragment: %s", text[:30])
        return text[: match.start()].rstrip()

    return text


def _remove_inline_footer(text: str, page_num: Optional[int]) -> str:
    """Remove footer or header fragments embedded in text."""

    patterns = [
        re.compile(
            rf"(?:^|\n)(?P<footer>[A-Z][^|\n]{{0,60}}?\|\s*(?P<page>\d{{1,3}}|{ROMAN_RE})(?![0-9ivxlcdm]))\n?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?:^|\n)(?P<footer>(?P<page>\d{{1,3}}|{ROMAN_RE})\s*\|\s*[A-Z][^|\n]{{0,60}})\n?",
            re.IGNORECASE,
        ),
        re.compile(r"(?:^|\n)(?P<footer>\|\s*[A-Z][^|\n]{0,60})\n?"),
        re.compile(r"(?:^|\n)(?P<footer>[A-Z][^|\n]{0,60}\|)\n?"),
    ]

    def repl(match: re.Match[str]) -> str:
        page_str = match.groupdict().get("page")
        if page_str:
            trailing = int(page_str) if page_str.isdigit() else _roman_to_int(page_str)
            page_num_checked = _normalize_page_num(page_num)
            if page_num_checked <= 0 or abs(trailing - page_num_checked) <= 1:
                logger.info(
                    "_remove_inline_footer removed footer: %s",
                    match.group("footer")[:30],
                )
                return "" if match.start() == 0 else "\n"
            return match.group(0)

        logger.info(
            "_remove_inline_footer removed footer: %s",
            match.group("footer")[:30],
        )
        return "" if match.start() == 0 else "\n"

    for pattern in patterns:
        text = pattern.sub(repl, text)

    return text


FOOTNOTE_LINE_RE = re.compile(
    rf"(?m)^\s*(?:[0-9{_SUP_DIGITS_ESC}]{{1,3}}|[\*\u2020])\s+[A-Z][^.]{{0,120}}\.(?:\s*|$)\n?"
)


def _remove_embedded_footnote(text: str) -> str:
    """Remove footnote lines merged into surrounding text."""

    return FOOTNOTE_LINE_RE.sub("", text)


FOOTNOTE_MARKER_RE = re.compile(
    rf"([^\s0-9{_SUP_DIGITS_ESC}])([0-9{_SUP_DIGITS_ESC}]+)[\r\n]+"
)

FOOTNOTE_LONE_LINE_RE = re.compile(
    rf"([^\s0-9{_SUP_DIGITS_ESC}])\r?\n([0-9{_SUP_DIGITS_ESC}]+)[\r\n]+"
)


def _normalize_footnote_markers(text: str) -> str:
    """Replace trailing footnote numbers with bracketed form.

    Patterns like ``sentence.3`` or ``sentence.\n3`` followed by one or more
    line breaks are transformed into ``sentence.[3]`` with a single trailing
    space. This keeps the footnote reference while preventing double newlines
    from breaking the paragraph flow.
    """

    def _inline(match: re.Match[str]) -> str:
        char, digits = match.groups()
        normalized = digits.translate(_SUPERSCRIPT_MAP)
        return f"[{normalized}]{char} " if char in ".!?" else f"{char}[{normalized}] "

    def _lone(match: re.Match[str]) -> str:
        char, digits = match.groups()
        normalized = digits.translate(_SUPERSCRIPT_MAP)
        return f"[{normalized}]{char} " if char in ".!?" else f"{char}[{normalized}] "

    return pipe(
        text,
        lambda t: FOOTNOTE_LONE_LINE_RE.sub(_lone, t),
        lambda t: FOOTNOTE_MARKER_RE.sub(_inline, t),
    )


def _flatten_markdown_table(text: str) -> str:
    """Flatten leading markdown table rows into plain lines.

    The function targets artifacts where a header table encodes chapter
    information. It strips alignment markers (``---``), generic column names
    like ``Col2`` and collapses duplicate cells that merely repeat the chapter
    title. ``<br>`` tags are expanded to newlines before deduplication to cover
    cases where author name and location share a single cell.
    """

    stripped = text.lstrip()
    if not stripped.startswith("|"):
        return text

    lines = list(
        takewhile(lambda ln: ln.lstrip().startswith("|"), stripped.splitlines())
    )
    if not any("---" in ln for ln in lines):
        return text

    col_re = re.compile(r"^col\d+$", re.IGNORECASE)
    rule_re = re.compile(r"^[-:]+$")

    cells = (
        cell.strip() for line in lines for cell in line.strip().strip("|").split("|")
    )

    filtered = (
        c for c in cells if c and not col_re.fullmatch(c) and not rule_re.fullmatch(c)
    )

    expanded = (part.strip() for cell in filtered for part in cell.split("<br>"))

    def _dedupe(acc: list[str], t: str) -> list[str]:
        return acc if any(t in prev for prev in acc) else acc + [t]

    deduped: list[str] = reduce(_dedupe, expanded, [])

    remaining = stripped.splitlines()[len(lines) :]  # noqa: E203
    flattened = "\n".join(deduped)
    return "\n".join(filter(None, [flattened, *remaining]))


def remove_page_artifact_lines(text: str, page_num: Optional[int]) -> str:
    """Remove header or footer artifact lines from a block."""

    pipeline = (
        _flatten_markdown_table,
        lambda t: _remove_inline_footer(t, page_num),
        _remove_embedded_footnote,
        _normalize_footnote_markers,
    )
    text = reduce(lambda acc, fn: fn(acc), pipeline, text)

    lines = text.splitlines()

    def _clean_line(ln: str) -> Optional[str]:
        if is_page_artifact_text(clean_text(ln), page_num):
            logger.debug("remove_page_artifact_lines dropped: %s", ln[:30])
            return None
        stripped = strip_page_artifact_suffix(ln, page_num)
        if stripped != ln:
            logger.debug("remove_page_artifact_lines stripped suffix: %s", ln[:30])
        return stripped

    cleaned = filter(None, (_clean_line(ln) for ln in lines))
    return "\n".join(cleaned)
