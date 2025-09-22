import logging
import re
from dataclasses import replace
from functools import reduce
from itertools import takewhile
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .pdf_blocks import Block


ROMAN_RE = r"[ivxlcdm]+"
_ROMAN_MAP = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
DOMAIN_RE = re.compile(r"\b[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)

SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_MAP = str.maketrans(SUPERSCRIPT_DIGITS, "0123456789")
_SUP_DIGITS_ESC = re.escape(SUPERSCRIPT_DIGITS)


TOC_DOT_RE = re.compile(r"(?:\.\s*){2,}")


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


_BULLET_CHARS = ("\u2022", "*", "-")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE_RE = re.compile(r"\+?\d[\d()\s.-]{5,}\d")
_CONTACT_KEYWORDS = ("contact", "tel", "phone", "fax", "email")


def _starts_with_bullet(line: str) -> bool:
    """Return ``True`` if ``line`` begins with a bullet marker."""

    return line.lstrip().startswith(_BULLET_CHARS)


def _looks_like_bullet_footer(text: str) -> bool:
    """Heuristic for bullet footer lines embedded in text."""

    stripped = text.strip()
    words = stripped.split()
    return _starts_with_bullet(stripped) and "?" in stripped and len(words) <= 2


def _looks_like_footer_context(text: str) -> bool:
    """Return ``True`` when ``text`` resembles footer boilerplate."""

    if not text:
        return False
    lowered = text.lower()
    return (
        _contains_domain(text)
        or "http" in lowered
        or "www." in lowered
        or lowered.strip(" -\u2022*").isdigit()
        or lowered.rstrip().endswith("page")
    )


def _bullet_body(text: str) -> str:
    """Return ``text`` without its leading bullet marker and adornments."""

    stripped = text.lstrip()
    if not _starts_with_bullet(stripped):
        return ""
    without_marker = stripped.lstrip("".join(_BULLET_CHARS))
    return without_marker.strip(" -\u2022*.")


def _first_non_empty_line(lines: Iterable[str]) -> str:
    """Return the first non-empty line from ``lines`` with surrounding whitespace trimmed."""

    return next((candidate.strip() for candidate in lines if candidate.strip()), "")


def _looks_like_contact_detail(text: str) -> bool:
    """Return ``True`` when ``text`` resembles an inline contact detail."""

    lowered = text.lower()
    return bool(
        _EMAIL_RE.search(text)
        or _PHONE_RE.search(text)
        or any(keyword in lowered for keyword in _CONTACT_KEYWORDS)
    )


def _trailing_bullet_candidates(lines: list[str]) -> list[tuple[int, str]]:
    """Return trailing bullet candidates paired with their indices."""

    enumerated = (
        (idx, line)
        for idx, line in enumerate(reversed(lines))
    )
    trailing = list(
        takewhile(lambda pair: _starts_with_bullet(pair[1].lstrip()), enumerated)
    )
    return [(len(lines) - 1 - idx, line) for idx, line in trailing]


def _footer_bullet_signals(*candidates: str) -> bool:
    """Return ``True`` if any ``candidates`` resemble footer content."""

    return any(
        predicate(candidate)
        for candidate in candidates
        if candidate
        for predicate in (_looks_like_footer_context, _looks_like_contact_detail)
    )


def _drop_trailing_bullet_footers(lines: list[str]) -> list[str]:
    """Remove isolated trailing bullet lines while preserving real lists."""

    trailing = _trailing_bullet_candidates(lines)
    if len(trailing) != 1:
        return lines

    idx, candidate = trailing[0]
    body = _bullet_body(candidate)
    previous = _first_non_empty_line(lines[pos] for pos in range(idx - 1, -1, -1))
    if body and not _footer_bullet_signals(body, previous):
        return lines

    logger.debug(
        "remove_page_artifact_lines dropped trailing bullet footer: %s",
        candidate.strip()[:30],
    )
    return [line for pos, line in enumerate(lines) if pos != idx]


def _strip_spurious_number_prefix(text: str) -> str:
    """Remove leading number markers preceding lowercase continuation."""

    return re.sub(r"^\s*\d+\.\s*(?=[a-z])", "", text)


_HEADER_CONNECTORS = {"of", "the", "and", "or", "to", "a", "an", "in", "for"}


def _strip_page_header_prefix(text: str) -> str:
    """Remove leading header fragments without chopping real sentences.

    The function walks tokens from the start, skipping over digits, connectors
    (``of``, ``the`` …), and title-cased or uppercase words that typically form
    headers. It tracks whether a comma-separated segment or a run of three or
    more title-cased tokens was seen—signals that the prefix is indeed a header.
    Once a lowercase token follows, the remaining tokens are preserved as body
    text. Lines lacking these header cues are returned untouched.
    """

    tokens = text.split()
    idx = 0
    comma_seen = False
    title_count = 0
    while idx < len(tokens):
        raw = tokens[idx]
        token = raw.strip(",")
        nxt_raw = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        nxt = nxt_raw.strip(",")
        if raw.endswith(","):
            comma_seen = True
        if token.isdigit():
            idx += 1
            continue
        if token.lower() in _HEADER_CONNECTORS:
            idx += 1
            continue
        if token.istitle() or token.isupper():
            title_count += 1
            if nxt.islower() and nxt not in _HEADER_CONNECTORS:
                break
            idx += 1
            continue
        break
    if idx == 0 or idx >= len(tokens) or (not comma_seen and title_count < 3):
        return text
    remainder = " ".join(t.strip(",") for t in tokens[idx:])
    if remainder and remainder[0].islower():
        remainder = remainder[0].upper() + remainder[1:]
    return remainder


def _starts_with_multiple_numbers(text: str) -> bool:
    """Return ``True`` if ``text`` begins with two or more numbers."""

    parts = text.strip().split()
    return sum(1 for p in takewhile(str.isdigit, parts)) >= 2


def _contains_domain(text: str) -> bool:
    """Return ``True`` if ``text`` contains a domain-like pattern."""

    return bool(DOMAIN_RE.search(text))


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _match_common_patterns(text: str) -> bool:
    """Return ``True`` if text matches common header/footer patterns."""

    text_lower = text.lower().strip()
    if _looks_like_numbered_list_item(text):
        return False
    patterns = (
        r"^\d+$",
        r"^page\s+\d+",
        r"^\d+\s*$",
        r"^chapter\s+\d+$",
        r"^\d+\s+chapter",
        r"^\w+\s*\|\s*\d+$",
        rf"^\w+\s*\|\s*{ROMAN_RE}$",
        r"^\d+\s*\|\s*[\w\s:]+$",
        rf"^{ROMAN_RE}$",
        r"^table\s+of\s+contents",
        r"^bibliography",
        r"^index$",
        r"^appendix\s+[a-z]$",
        r"^[a-z][^|]{0,60}\|$",
    )
    if any(re.match(p, text_lower) for p in patterns):
        return True
    return bool(re.match(r"^[0-9]{1,3}[.)]?\s+[A-Z]", text) and len(text.split()) <= 8)


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
        if (page_num <= 0 or abs(trailing - page_num) <= 1) and len(text) - m.end() <= 20:
            logger.debug("_match_page_number_suffix pipe pattern detected: %s", text[:30])
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
        if (page_num <= 0 or abs(trailing - page_num) <= 1) and len(text) - m.end() <= 20:
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

    if _match_common_patterns(text):
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

    if len(text.split()) <= 3 and len(text) <= 30 and any(char.isdigit() for char in text):
        if re.match(r"^\d+\.$", text.strip()):
            return False
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
    if (page_num <= 0 or abs(trailing - page_num) <= 1) and len(text) - match.end() <= 20:
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


FOOTNOTE_PREFIX_RE = re.compile(rf"^\s*(?:[0-9{_SUP_DIGITS_ESC}]+[.)]?|[\*\u2020])\s+")


def is_probable_footnote(line: str, idx: int, total: int) -> bool:
    """Return ``True`` for short numbered lines near page edges."""

    edge_band = 2
    return (
        (idx < edge_band or idx >= total - edge_band)
        and len(line.split()) <= 4
        and bool(FOOTNOTE_PREFIX_RE.match(line))
    )


def _is_number_marker(line: str) -> bool:
    """Return ``True`` for standalone numeric markers like ``"1."``."""

    return bool(re.match(r"^\s*\d+[.)]?\s*$", line))


def _looks_like_numbered_list_item(line: str) -> bool:
    """Heuristically detect numbered list entries to avoid stripping them."""

    stripped = line.strip()
    if not stripped:
        return False

    match = FOOTNOTE_PREFIX_RE.match(stripped)
    if not match:
        return False

    remainder = stripped[match.end() :].strip()
    if not remainder:
        return False

    words = remainder.split()
    if len(words) < 3:
        return False

    trailing = remainder.rstrip()
    return not trailing.endswith((".", "!", "?", ";", ":"))


def _strip_footnote_lines(text: str) -> str:
    """Remove short footnote lines and stray numeric markers."""

    lines = text.splitlines()
    total = len(lines)
    def _should_keep(idx: int, ln: str) -> bool:
        if _is_number_marker(ln):
            return False
        if is_probable_footnote(ln, idx, total) and not _looks_like_numbered_list_item(ln):
            return False
        return True

    kept = (ln for idx, ln in enumerate(lines) if _should_keep(idx, ln))
    return "\n".join(kept)


FOOTNOTE_MARKER_RE = re.compile(rf"(?<=[^\s0-9{_SUP_DIGITS_ESC}])([0-9{_SUP_DIGITS_ESC}]+)[\r\n]+")


def _remove_inline_footnote_prefix(line: str) -> tuple[str, bool]:
    """Strip numeric footnote prefixes while keeping continuation sentences."""

    if not _looks_like_footnote(line):
        return line, False

    stripped = line.lstrip()
    match = FOOTNOTE_PREFIX_RE.match(stripped)
    if not match:
        return line, False

    remainder = stripped[match.end() :].lstrip()
    if not remainder:
        return "", True

    segments = _SENTENCE_BOUNDARY_RE.split(remainder, maxsplit=1)
    if len(segments) == 2:
        continuation = segments[1].strip()
        if continuation:
            return continuation, True
    return "", True


def _normalize_footnote_markers(text: str) -> str:
    """Replace trailing footnote numbers with bracketed form.

    Patterns like ``sentence.3`` followed by one or more line breaks are
    transformed into ``sentence.[3]`` with a single trailing space. This keeps
    the footnote reference while preventing double newlines from breaking the
    paragraph flow.
    """

    def repl(match: re.Match[str]) -> str:
        digits = match.group(1).translate(_SUPERSCRIPT_MAP)
        return f"[{digits}] "

    return FOOTNOTE_MARKER_RE.sub(repl, text)


def _strip_toc_dot_leaders(text: str) -> str:
    """Remove dot leaders and trailing page numbers from Table of Contents lines."""

    def strip_line(ln: str) -> str:
        if TOC_DOT_RE.search(ln):
            head = TOC_DOT_RE.split(ln)[0].rstrip()
            return head if head.strip() else ""
        return ln

    lines = [strip_line(ln) for ln in text.splitlines()]
    return "\n".join(filter(None, lines))


TRAILING_FOOTER_RE = re.compile(
    rf"\s*\|\s*(\d{{1,3}}|{ROMAN_RE})(?![0-9ivxlcdm])\s*$", re.IGNORECASE
)


def _strip_trailing_footer(text: str, page_num: Optional[int]) -> str:
    """Remove a terminal ``"| N"`` fragment if it matches the page."""

    match = TRAILING_FOOTER_RE.search(text)
    if not match:
        return text

    page_str = match.group(1)
    trailing = int(page_str) if page_str.isdigit() else _roman_to_int(page_str)
    page_num = _normalize_page_num(page_num)
    if page_num <= 0 or abs(trailing - page_num) <= 1:
        logger.info("_strip_trailing_footer removed: %s", text[:30])
        return text[: match.start()].rstrip()
    return text


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

    lines = list(takewhile(lambda ln: ln.lstrip().startswith("|"), stripped.splitlines()))
    if not any("---" in ln for ln in lines):
        return text

    col_re = re.compile(r"^col\d+$", re.IGNORECASE)
    rule_re = re.compile(r"^[-:]+$")

    cells = (cell.strip() for line in lines for cell in line.strip().strip("|").split("|"))

    filtered = (c for c in cells if c and not col_re.fullmatch(c) and not rule_re.fullmatch(c))

    expanded = (part.strip() for cell in filtered for part in cell.split("<br>"))

    def _dedupe(acc: list[str], t: str) -> list[str]:
        return acc if any(t in prev for prev in acc) else acc + [t]

    deduped: list[str] = reduce(_dedupe, expanded, [])

    remaining = stripped.splitlines()[len(lines) :]  # noqa: E203
    flattened = "\n".join(deduped)
    return "\n".join(filter(None, [flattened, *remaining]))


def _leading_alpha(text: str) -> tuple[Optional[int], Optional[str]]:
    """Return the position and character of the first alphabetical symbol."""

    return next(((idx, ch) for idx, ch in enumerate(text) if ch.isalpha()), (None, None))


def _uppercase_char(text: str, index: int) -> str:
    """Return ``text`` with the character at ``index`` upper-cased."""

    return "".join(ch.upper() if idx == index else ch for idx, ch in enumerate(text))


def _normalize_leading_case(original: str, cleaned: str) -> str:
    """Return ``cleaned`` with casing aligned to ``original``'s leading alpha."""

    if not cleaned:
        return cleaned

    stripped_original = original.lstrip()
    stripped_cleaned = cleaned.lstrip()
    orig_idx, orig_char = _leading_alpha(stripped_original)
    clean_idx, clean_char = _leading_alpha(stripped_cleaned)

    if not orig_char or not clean_char or orig_char.islower() or clean_char.isupper():
        return cleaned

    offset = len(cleaned) - len(stripped_cleaned)
    return _uppercase_char(cleaned, offset + clean_idx)


def _apply_leading_case(pairs: list[tuple[str, str]]) -> list[str]:
    """Return cleaned lines with the first entry's case normalised."""

    if not pairs:
        return []

    first_original, first_cleaned = pairs[0]
    head = _normalize_leading_case(first_original, first_cleaned)
    tail = [cleaned for _, cleaned in pairs[1:]]
    return [head, *tail]


def remove_page_artifact_lines(text: str, page_num: Optional[int]) -> str:
    """Remove header or footer artifact lines from a block."""

    pipeline = (
        _flatten_markdown_table,
        lambda t: _remove_inline_footer(t, page_num),
        _strip_footnote_lines,
        _normalize_footnote_markers,
        _strip_toc_dot_leaders,
        lambda t: _strip_trailing_footer(t, page_num),
    )
    text = reduce(lambda acc, fn: fn(acc), pipeline, text)

    lines = text.splitlines()

    def _clean_line(ln: str) -> Optional[str]:
        normalized = ln if _starts_with_bullet(ln) else _strip_page_header_prefix(ln)
        normalized, removed_inline = _remove_inline_footnote_prefix(normalized)
        if not normalized:
            if removed_inline:
                logger.debug(
                    "remove_page_artifact_lines dropped inline footnote: %s",
                    ln[:30],
                )
            return None
        if removed_inline:
            logger.debug(
                "remove_page_artifact_lines preserved inline continuation: %s",
                normalized[:30],
            )
        if is_page_artifact_text(normalized, page_num) or _looks_like_bullet_footer(normalized):
            logger.debug("remove_page_artifact_lines dropped: %s", ln[:30])
            return None
        stripped = _strip_spurious_number_prefix(strip_page_artifact_suffix(normalized, page_num))
        if stripped != normalized:
            logger.debug("remove_page_artifact_lines stripped suffix: %s", ln[:30])
        return stripped

    cleaned_pairs = [
        (original, cleaned)
        for original, cleaned in ((ln, _clean_line(ln)) for ln in lines)
        if cleaned
    ]
    normalised = _apply_leading_case(cleaned_pairs)
    pruned = _drop_trailing_bullet_footers(normalised)
    return "\n".join(pruned)


def strip_artifacts(blocks: Iterable["Block"], config=None) -> Iterable["Block"]:
    """Yield blocks with header and footer artifacts stripped."""

    for blk in blocks:
        page = blk.source.get("page")
        cleaned = remove_page_artifact_lines(blk.text, page)
        if cleaned and not is_page_artifact_text(cleaned, page):
            yield replace(blk, text=cleaned)
