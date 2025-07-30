import logging
import re

from .text_cleaning import clean_text

logger = logging.getLogger(__name__)


def _match_common_patterns(text_lower: str) -> bool:
    """Return True if text matches common header/footer patterns."""
    patterns = [
        r"^\d+$",
        r"^page\s+\d+",
        r"^\d+\s*$",
        r"^chapter\s+\d+$",
        r"^\d+\s+chapter",
        r"^\w+\s*\|\s*\d+$",
        r"^\d+\s*\|\s*[\w\s:]+$",
        r"^[0-9]{1,3}[.)]?\s+[A-Z]",
        r"^table\s+of\s+contents",
        r"^bibliography",
        r"^index$",
        r"^appendix\s+[a-z]$",
    ]
    return any(re.match(p, text_lower) for p in patterns)


def _match_page_number_suffix(text: str, page_num: int) -> bool:
    """Detect page-number fragments at line ends or near the end.

    If ``page_num`` is ``0`` or negative, the numeric check is skipped.
    """

    # Exact trailing page number
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

    return False


def is_page_artifact_text(text: str, page_num: int) -> bool:
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

    if (
        len(text.split()) <= 3
        and len(text) <= 30
        and any(char.isdigit() for char in text)
    ):
        return True

    return False


def strip_page_artifact_suffix(text: str, page_num: int) -> str:
    """Return the line with any trailing ``"| N"`` footer fragment removed."""

    pattern = re.compile(r"\|\s*(\d{1,3})(?!\d)")
    match = pattern.search(text)
    if not match:
        return text

    trailing = int(match.group(1))
    if (page_num <= 0 or abs(trailing - page_num) <= 1) and len(
        text
    ) - match.end() <= 20:
        logger.info("strip_page_artifact_suffix removed footer fragment: %s", text[:30])
        return text[: match.start()].rstrip()

    return text


def _remove_inline_footer(text: str, page_num: int) -> str:
    """Remove footer fragments embedded inside a paragraph."""

    pattern = re.compile(r"\n\n([A-Z][^|\n]{0,60}?\|\s*(\d{1,3})(?!\d))")

    def repl(match: re.Match[str]) -> str:
        trailing = int(match.group(2))
        if page_num <= 0 or abs(trailing - page_num) <= 1:
            logger.info("_remove_inline_footer removed footer: %s", match.group(1)[:30])
            return "\n\n"
        return match.group(0)

    return pattern.sub(repl, text)


def remove_page_artifact_lines(text: str, page_num: int) -> str:
    """Remove header or footer artifact lines from a block."""

    text = _remove_inline_footer(text, page_num)

    lines = text.splitlines()

    def _is_artifact(idx: int) -> bool:
        ln = clean_text(lines[idx])
        return is_page_artifact_text(ln, page_num)

    cleaned_lines = []
    for idx, ln in enumerate(lines):
        if _is_artifact(idx):
            logger.debug("remove_page_artifact_lines dropped: %s", ln[:30])
            continue
        stripped = strip_page_artifact_suffix(ln, page_num)
        if stripped != ln:
            logger.debug("remove_page_artifact_lines stripped suffix: %s", ln[:30])
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)
