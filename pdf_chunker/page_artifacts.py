import logging
import re
from dataclasses import replace
from functools import reduce
from itertools import groupby, takewhile
from typing import Iterable, Optional, Sequence, TYPE_CHECKING

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


def _question_footer_token_stats(body: str) -> tuple[int, int]:
    """Return ``(total_tokens, long_tokens)`` for ``body``."""

    tokens = tuple(re.findall(r"[A-Za-z0-9]+", body))
    long_tokens = sum(1 for token in tokens if len(token) > 3)
    return len(tokens), long_tokens


def _is_question_bullet_footer_run(
    span: tuple[int, ...],
    lines: list[str],
) -> bool:
    """Return ``True`` when ``span`` represents a footer-style bullet run."""

    if not span:
        return False

    bodies = tuple(_bullet_body(lines[idx]) for idx in span)
    if not all(bodies):
        return False

    stats = tuple(_question_footer_token_stats(body) for body in bodies)
    token_totals = tuple(total for total, _ in stats)
    long_totals = tuple(long for _, long in stats)
    question_counts = tuple(body.count("?") for body in bodies)

    if not all(question_counts):
        return False

    run_length = len(span)
    question_sum = sum(question_counts)
    shortish = max(token_totals) <= 28
    limited_long = max(long_totals) <= 14

    if run_length == 1:
        total_tokens, long_token_count = stats[0]
        tokens = re.findall(r"[A-Za-z0-9]+", bodies[0])
        first_token = tokens[0].lower() if tokens else ""
        question_leads = {
            "who",
            "what",
            "when",
            "where",
            "why",
            "how",
            "is",
            "are",
            "am",
            "do",
            "does",
            "did",
            "can",
            "could",
            "should",
            "would",
            "will",
            "have",
            "has",
            "had",
            "may",
            "might",
            "shall",
            "which",
        }
        if first_token in question_leads:
            return False

        if "(" in bodies[0] or ")" in bodies[0]:
            return False

        single_line_short = total_tokens <= 10 and long_token_count <= 6
        return question_counts[0] >= 1 and single_line_short

    threshold = run_length + max(1, run_length // 2)
    question_dense = question_sum >= threshold
    return shortish and limited_long and question_dense


def _question_footer_indices(lines: list[str]) -> frozenset[int]:
    """Return indices for bullet lines that resemble footer question clusters."""

    total = len(lines)
    spans = (
        tuple(idx for idx, _ in group)
        for is_bullet, group in groupby(
            enumerate(lines), key=lambda pair: _starts_with_bullet(pair[1].lstrip())
        )
        if is_bullet
    )

    def _near_footer(span: tuple[int, ...]) -> bool:
        return bool(span) and (total - span[-1] - 1) <= 4

    qualifying = (
        span
        for span in spans
        if _near_footer(span) and _is_question_bullet_footer_run(span, lines)
    )

    bullet_indices = frozenset(idx for span in qualifying for idx in span)

    def _looks_like_question_continuation(idx: int, line: str) -> bool:
        stripped = line.strip()
        if idx in bullet_indices or "?" not in stripped:
            return False
        tokens = re.findall(r"[A-Za-z0-9]+", stripped)
        if not tokens or len(tokens) > 6:
            return False
        from_end = total - idx - 1
        initial = stripped[0]
        question_tail = stripped.endswith("?") or stripped.endswith("?)") or stripped.endswith("?\"")
        return from_end <= 4 and initial.islower() and question_tail

    continuation = frozenset(
        idx for idx, line in enumerate(lines) if _looks_like_question_continuation(idx, line)
    )

    return bullet_indices.union(continuation)


def _first_text_line(text: str) -> str:
    return _first_non_empty_line(text.splitlines())


def _classify_footer_block(text: str) -> str:
    first_line = _first_text_line(text)
    if not first_line:
        return ""
    if _starts_with_bullet(first_line):
        return "bullet"
    if first_line[0].islower():
        return "continuation"
    if len(first_line.split()) <= 2 and "?" in first_line:
        return "continuation"
    return ""


_INLINE_KEEP_LEADS = frozenset({
    "for",
    "when",
    "where",
    "how",
    "why",
    "what",
})


_INLINE_FOOTER_BOTTOM_MARGIN = 96.0


def _inline_bullet_lines(block: "Block") -> tuple[str, ...]:
    text = getattr(block, "text", "") or ""
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not lines or not all(_starts_with_bullet(line) for line in lines):
        return ()
    return lines if len(lines) <= 3 else ()


def _block_first_line(block: "Block") -> str:
    text = getattr(block, "text", "") or ""
    return _first_text_line(text)


def _is_trailing_inline_footer(
    blocks: Sequence["Block"], idx: int, previous_text: str
) -> bool:
    tail_lines = (
        _block_first_line(block)
        for block in blocks[idx + 1 :]
    )
    return all(
        not line
        or _footer_bullet_signals(line, previous_text)
        or _looks_like_bullet_footer(line)
        for line in tail_lines
    )


def _should_drop_inline_footer(
    blocks: Sequence["Block"], idx: int, block: "Block", lines: tuple[str, ...], bottom: Optional[float]
) -> bool:
    word_totals = tuple(len(line.split()) for line in lines)
    total_words = sum(word_totals)
    if total_words >= 40 or any(count >= 12 for count in word_totals):
        return False

    bodies = tuple(_bullet_body(line) for line in lines)
    if not all(bodies):
        return False

    previous_text = getattr(blocks[idx - 1], "text", "") if idx else ""
    inviting_previous = previous_text.strip().endswith(":")
    if inviting_previous:
        return False

    first_tokens = tuple((body.split() or ("",))[0].lower() for body in bodies)
    if any(token in _INLINE_KEEP_LEADS for token in first_tokens):
        return False

    if any("?" in body or _footer_bullet_signals(body, previous_text) for body in bodies):
        return True

    bbox = getattr(block, "bbox", None)
    near_bottom = (
        bottom is not None
        and bbox is not None
        and (bottom - bbox[3]) <= _INLINE_FOOTER_BOTTOM_MARGIN
    )
    trailing_short = (
        len(lines) <= 2
        and total_words <= 24
        and (
            _is_trailing_inline_footer(blocks, idx, previous_text)
            or near_bottom
        )
    )
    return trailing_short


def _inline_footer_drop_indices(blocks: Sequence["Block"]) -> frozenset[int]:
    """Return indices for inline bullet footers worth pruning."""

    bottoms = [blk.bbox[3] for blk in blocks if getattr(blk, "bbox", None)]
    page_bottom = max(bottoms) if bottoms else None

    return frozenset(
        idx
        for idx, block in enumerate(blocks)
        for lines in (_inline_bullet_lines(block),)
        if lines and _should_drop_inline_footer(blocks, idx, block, lines, page_bottom)
    )


def _prune_footer_blocks(blocks: list["Block"]) -> list["Block"]:
    if not blocks:
        return []

    inline_drops = _inline_footer_drop_indices(blocks)
    filtered = [blk for idx, blk in enumerate(blocks) if idx not in inline_drops]
    if not filtered:
        return []

    kinds = [(_classify_footer_block(blk.text), idx) for idx, blk in enumerate(filtered)]
    cluster: list[int] = []
    bullet_count = 0
    idx = len(kinds) - 1
    while idx >= 0:
        kind, _ = kinds[idx]
        if not kind:
            break
        cluster.append(idx)
        if kind == "bullet":
            bullet_count += 1
        idx -= 1

    if bullet_count < 3:
        return filtered

    cluster.reverse()
    start = cluster[0]
    previous_text = filtered[start - 1].text if start > 0 else ""
    if previous_text and not previous_text.strip().endswith(":"):
        return filtered

    cluster_texts = tuple(filtered[idx].text for idx in cluster)
    bodies = tuple(_first_non_empty_line(text.splitlines()) for text in cluster_texts)
    if not all(bodies):
        return filtered

    word_totals = tuple(len(body.split()) for body in bodies)
    dense_items = sum(count >= 12 for count in word_totals)
    has_sentence = any("." in body for body in bodies if len(body.split()) >= 8)
    total_words = sum(word_totals)
    footerish_bodies = tuple(
        _footer_bullet_signals(body, previous_text) or "?" in body
        for body in bodies
    )

    if dense_items >= 3 or has_sentence or total_words >= 60:
        return filtered

    if not all(footerish_bodies):
        return filtered

    drop = set(cluster)
    return [blk for idx, blk in enumerate(filtered) if idx not in drop]


def _first_non_empty_line(lines: Iterable[str]) -> str:
    """Return the first non-empty line from ``lines`` with surrounding whitespace trimmed."""

    return next((candidate.strip() for candidate in lines if candidate.strip()), "")


def _next_non_empty_line(lines: list[str], start: int) -> str:
    """Return the next non-empty line after ``start`` with whitespace stripped."""

    return _first_non_empty_line(lines[pos] for pos in range(start + 1, len(lines)))


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
    if not trailing:
        return lines

    bodies = [(_bullet_body(line), pos) for pos, line in trailing]
    trailing_indices = tuple(pos for pos, _ in trailing)
    previous = _first_non_empty_line(
        lines[pos] for pos in range(trailing[-1][0] - 1, -1, -1)
    )

    def _should_prune(body: str) -> bool:
        return not body or _footer_bullet_signals(body, previous)

    after_idx = trailing[-1][0] + 1
    after_line = lines[after_idx] if after_idx < len(lines) else ""
    trailing_count = len(trailing)
    context_allows = any(
        (
            _looks_like_shipping_footer(after_line),
            _footer_bullet_signals(after_line, previous),
            _header_invites_footer(previous, trailing_count),
        )
    )

    stats = tuple(
        (_question_footer_token_stats(body), pos)
        for body, pos in bodies
        if body
    )
    dense_footer_run = (
        len(trailing_indices) >= 4
        and previous.rstrip().endswith(":")
        and stats
        and all(total >= 20 and long >= 10 for (total, long), _ in stats)
    )

    if _is_question_bullet_footer_run(trailing_indices, lines) or dense_footer_run:
        totals = tuple(total for (total, _), _ in stats)
        if totals and (sum(totals) >= 40 or any(total >= 12 for total in totals)):
            return lines
        removals = list(trailing_indices)
        keep_indices = set(removals)
        logger.debug(
            "remove_page_artifact_lines dropped trailing question footer bullets: %s",
            "; ".join(lines[pos].strip()[:30] for pos in removals),
        )
        return [line for idx, line in enumerate(lines) if idx not in keep_indices]

    removals = [pos for body, pos in bodies if _should_prune(body)]
    if len(removals) != len(bodies) or not context_allows:
        return lines

    keep_indices = set(removals)
    logger.debug(
        "remove_page_artifact_lines dropped trailing bullet footers: %s",
        "; ".join(lines[pos].strip()[:30] for pos in removals),
    )
    return [line for idx, line in enumerate(lines) if idx not in keep_indices]


def _header_invites_footer(previous_line: str, trailing_count: int) -> bool:
    """Return ``True`` when ``previous_line`` resembles a footer heading."""

    stripped = previous_line.strip()
    if not stripped:
        return False
    colon_header = stripped.endswith(":") and trailing_count <= 3
    uppercase_header = stripped.isupper() and len(stripped.split()) <= 5
    return colon_header or uppercase_header


_TITLE_CONNECTORS = {
    "and",
    "or",
    "the",
    "of",
    "for",
    "in",
    "on",
    "from",
    "to",
    "by",
    "with",
    "at",
}

_SHIPPING_FOOTER_PREFIXES = (
    "directed to",
    "some trader",
    "he expects",
)


def _is_titlecase_token(token: str) -> bool:
    return token.isupper() or token.istitle()


def _looks_like_named_entity_line(text: str) -> bool:
    """Return ``True`` for lines dominated by title-cased tokens."""

    tokens = [re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", part) for part in text.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < 2:
        return False

    title_tokens = sum(1 for token in tokens if token[0].isupper())
    if text.count(",") >= 1:
        return title_tokens >= 2 and title_tokens >= len(tokens) // 2
    return title_tokens >= max(3, (2 * len(tokens)) // 3)


def _looks_like_isolated_title(line: str) -> bool:
    """Return ``True`` for short title-case phrases without punctuation."""

    stripped = line.strip()
    if not stripped or any(ch.isdigit() for ch in stripped):
        return False
    if any(ch in stripped for ch in ",;:|?/•"):
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", stripped)
    if not 3 <= len(tokens) <= 8:
        return False

    return all(
        _is_titlecase_token(token) or token.lower() in _TITLE_CONNECTORS
        for token in tokens
    )


def _looks_like_running_text(line: str) -> bool:
    """Return ``True`` when ``line`` resembles flowing body text."""

    stripped = line.strip()
    if not stripped:
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", stripped)
    if not tokens:
        return False

    lowercase_tokens = sum(token.islower() for token in tokens)
    punctuation_present = any(ch in stripped for ch in ",.;:?!")

    return punctuation_present or lowercase_tokens >= 2 or len(tokens) >= 10


def _should_remove_isolated_title(
    line: str,
    idx: int,
    lines: list[str],
    page_num: Optional[int],
) -> bool:
    """Return ``True`` when an isolated title most likely belongs to a header/footer."""

    if not _looks_like_isolated_title(line):
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", line)
    if len(tokens) > 5:
        return False

    prev_line = _first_non_empty_line(lines[pos] for pos in range(idx - 1, -1, -1))
    next_line = _next_non_empty_line(lines, idx)
    neighbours = [candidate for candidate in (prev_line, next_line) if candidate]

    neighbour_signals = any(
        is_page_artifact_text(candidate, page_num)
        or _looks_like_footer_context(candidate)
        or _looks_like_contact_detail(candidate)
        for candidate in neighbours
    )

    if neighbour_signals:
        return True

    if idx == 0 and (page_num or 0) > 1:
        next_line = _next_non_empty_line(lines, idx)
        if next_line and _looks_like_running_text(next_line):
            return True

    return False


def _looks_like_shipping_footer(line: str) -> bool:
    stripped = line.strip().lower()
    return any(stripped.startswith(prefix) for prefix in _SHIPPING_FOOTER_PREFIXES)


def _strip_spurious_number_prefix(text: str) -> str:
    """Remove leading number markers preceding lowercase continuation."""

    return re.sub(r"^\s*\d+\.\s*(?=[a-z])", "", text)


_HEADER_CONNECTORS = {"of", "the", "and", "or", "to", "a", "an", "in", "for", "on"}


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
BR_SPLIT_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
CELL_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
CONTACT_KEYWORD_RE = re.compile(
    r"\b(?:phone|email|contact|tel|mobile|fax)\b",
    re.IGNORECASE,
)
CREDENTIAL_RE = re.compile(
    r"\b(?:pmp|phd|mba|cfa|cpa|esq|md|jd)\b",
    re.IGNORECASE,
)
DIGIT_SEQUENCE_RE = re.compile(r"\d{3,}")
WORD_RE = re.compile(r"[A-Za-z]+")


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


def _normalize_table_cell(text: str) -> str:
    """Return a lowercase, punctuation-stripped form of ``text``."""

    return CELL_NORMALIZE_RE.sub("", text.lower())


def _looks_like_contact_cell(segment: str, source: str) -> bool:
    """Heuristically determine whether ``segment`` holds contact metadata."""

    lower_source = source.lower()
    if "@" in source or DOMAIN_RE.search(source) or CONTACT_KEYWORD_RE.search(lower_source):
        return True
    if CREDENTIAL_RE.search(segment) or DIGIT_SEQUENCE_RE.search(segment):
        return True

    words = WORD_RE.findall(segment)
    if not words or any(word.islower() for word in words):
        return False

    has_multiple_words = len(words) >= 2
    has_contact_punctuation = "," in source or "<br" in lower_source
    has_short_upper_suffix = any(len(word) <= 3 and word.isupper() for word in words)

    return has_multiple_words and (has_contact_punctuation or has_short_upper_suffix)


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

    expanded = (
        (segment.strip(), cell)
        for cell in filtered
        for segment in BR_SPLIT_RE.split(cell)
    )

    def _dedupe(
        acc: tuple[list[str], tuple[tuple[str, str], ...]],
        item: tuple[str, str],
    ) -> tuple[list[str], tuple[tuple[str, str], ...]]:
        values, contexts = acc
        segment, source = item
        if not segment:
            return acc
        if any(segment in prev for prev in values):
            return acc
        return (values + [segment], contexts + ((segment, source),))

    deduped, contexts = reduce(_dedupe, expanded, ([], tuple()))

    def _filter_segments() -> Iterable[str]:
        seen_norm = set()
        kept: list[str] = []
        for segment, source in contexts:
            normalized = _normalize_table_cell(segment)
            if not normalized:
                continue
            if _looks_like_contact_cell(segment, source):
                continue
            if normalized in seen_norm:
                continue
            if any(segment.lower() in existing.lower() for existing in kept):
                continue
            seen_norm.add(normalized)
            kept.append(segment)
            yield segment

    deduped = list(_filter_segments()) or deduped

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


def _first_alpha_after(text: str, start: int) -> Optional[int]:
    """Return the index of the first alphabetic character in ``text`` after ``start``."""

    return next((idx for idx, ch in enumerate(text[start:], start=start) if ch.isalpha()), None)


def _restore_colon_suffix_case(original: str, cleaned: str) -> str:
    """Uppercase colon suffix initials when the original text used title casing."""

    if ":" not in original or ":" not in cleaned:
        return cleaned

    orig_colons = (idx for idx, ch in enumerate(original) if ch == ":")
    cleaned_colons = (idx for idx, ch in enumerate(cleaned) if ch == ":")
    chars = list(cleaned)

    for orig_idx, cleaned_idx in zip(orig_colons, cleaned_colons):
        orig_alpha = _first_alpha_after(original, orig_idx + 1)
        cleaned_alpha = _first_alpha_after(chars, cleaned_idx + 1)
        if (
            orig_alpha is None
            or cleaned_alpha is None
            or not original[orig_alpha].isupper()
            or not chars[cleaned_alpha].islower()
        ):
            continue
        chars[cleaned_alpha] = chars[cleaned_alpha].upper()

    return "".join(chars)


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
    total = len(lines)

    question_footer_indices = _question_footer_indices(lines)

    def _clean_line(idx: int, ln: str) -> Optional[str]:
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
        stripped_norm = normalized.strip()
        if stripped_norm in _BULLET_CHARS:
            next_line = _next_non_empty_line(lines, idx)
            previous_line = _first_non_empty_line(lines[pos] for pos in range(idx - 1, -1, -1))
            prev_raw = lines[idx - 1] if idx > 0 else ""
            prev_has_body = prev_raw.lstrip().startswith(_BULLET_CHARS) and bool(_bullet_body(prev_raw))
            next_is_bullet = next_line.lstrip().startswith(_BULLET_CHARS) if next_line else False
            if (
                not next_line
                or _footer_bullet_signals(next_line, previous_line)
                or any(
                    _looks_like_named_entity_line(candidate)
                    for candidate in (next_line, previous_line)
                    if candidate
                )
                or (prev_has_body and not next_is_bullet)
                or (next_line and _looks_like_shipping_footer(next_line))
                or next_is_bullet
            ):
                logger.debug(
                    "remove_page_artifact_lines dropped empty bullet marker: %s",
                    ln[:30],
                )
                return None
        if idx in question_footer_indices:
            logger.debug(
                "remove_page_artifact_lines dropped question bullet footer: %s",
                ln[:30],
            )
            return None

        edge_band = 2
        if idx < edge_band or idx >= total - edge_band:
            if _should_remove_isolated_title(normalized, idx, lines, page_num):
                logger.debug(
                    "remove_page_artifact_lines dropped isolated title: %s",
                    normalized[:30],
                )
                return None

        if is_page_artifact_text(normalized, page_num) or _looks_like_bullet_footer(normalized):
            logger.debug("remove_page_artifact_lines dropped: %s", ln[:30])
            return None
        stripped = _strip_spurious_number_prefix(strip_page_artifact_suffix(normalized, page_num))
        if stripped != normalized:
            logger.debug("remove_page_artifact_lines stripped suffix: %s", ln[:30])
        return stripped

    cleaned_pairs = [
        (original, cleaned)
        for original, cleaned in (
            (ln, _clean_line(idx, ln))
            for idx, ln in enumerate(lines)
        )
        if cleaned
    ]
    normalised = _apply_leading_case(cleaned_pairs)
    colon_restored = [
        _restore_colon_suffix_case(original, cleaned)
        for (original, _), cleaned in zip(cleaned_pairs, normalised)
    ]
    pruned = _drop_trailing_bullet_footers(colon_restored)
    return "\n".join(pruned)


def strip_artifacts(blocks: Iterable["Block"], config=None) -> Iterable["Block"]:
    """Yield blocks with header and footer artifacts stripped."""

    pending: list["Block"] = []
    current_page: Optional[int] = None

    def _flush() -> Iterable["Block"]:
        nonlocal pending
        pruned = _prune_footer_blocks(pending)
        pending = []
        return pruned

    for blk in blocks:
        page = blk.source.get("page")
        cleaned = remove_page_artifact_lines(blk.text, page)
        if not cleaned or is_page_artifact_text(cleaned, page):
            continue
        updated = replace(blk, text=cleaned)
        if current_page is None or page == current_page:
            pending.append(updated)
            current_page = page
            continue
        for pruned_block in _flush():
            yield pruned_block
        pending.append(updated)
        current_page = page

    if pending:
        for pruned_block in _flush():
            yield pruned_block
