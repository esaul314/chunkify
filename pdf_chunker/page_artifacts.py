import re
import logging

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
    """Detect trailing page numbers that align with the page number."""
    m = re.search(r"(\d{1,3})\s*$", text)
    if m:
        trailing = int(m.group(1))
        if abs(trailing - page_num) <= 1:
            words = text.split()
            if "|" in text or len(words) <= 8:
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
            f"is_page_artifact_text() page number suffix: {text[:30]}… (page {page_num})"
        )
        return True

    if (
        len(text.split()) <= 3
        and len(text) <= 30
        and any(char.isdigit() for char in text)
    ):
        return True

    return False
