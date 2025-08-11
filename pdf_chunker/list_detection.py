import re
from typing import Tuple

BULLET_CHARS = "*•◦▪‣·●◉○‧"
BULLET_CHARS_ESC = re.escape(BULLET_CHARS)
HYPHEN_BULLET_PREFIX = "- "
NUMBERED_RE = re.compile(r"\s*\d+[.)]")


def strip_bullet_prefix(text: str) -> str:
    """Remove leading bullet markers and surrounding whitespace."""
    return re.sub(rf"^[\s\-]*[{BULLET_CHARS_ESC}]\s*", "", text.lstrip())


def starts_with_bullet(text: str) -> bool:
    """Return True if ``text`` begins with a bullet marker or hyphen bullet."""
    stripped = text.lstrip()
    return stripped.startswith(tuple(BULLET_CHARS)) or stripped.startswith(
        HYPHEN_BULLET_PREFIX
    )


def _last_non_empty_line(text: str) -> str:
    return next(
        (line.strip() for line in reversed(text.splitlines()) if line.strip()),
        "",
    )


def is_bullet_continuation(curr: str, nxt: str) -> bool:
    """Return True when ``nxt`` continues a bullet item from ``curr``."""
    last_line = _last_non_empty_line(curr)
    return last_line.endswith(tuple(BULLET_CHARS)) and nxt[:1].islower()


def is_bullet_fragment(curr: str, nxt: str) -> bool:
    """Return True when ``nxt`` continues the last bullet item from ``curr``."""
    last_line = _last_non_empty_line(curr)
    if not (
        starts_with_bullet(last_line)
        and not last_line.rstrip().endswith((".", "!", "?"))
    ):
        return False
    stripped = strip_bullet_prefix(nxt)
    return bool(stripped) and stripped[:1].islower()


def split_bullet_fragment(text: str) -> Tuple[str, str]:
    """Split a bullet-prefixed continuation from any following list items."""
    stripped = strip_bullet_prefix(text)
    bullet_match = re.search(rf"[{BULLET_CHARS_ESC}]", stripped)
    if bullet_match:
        idx = bullet_match.start()
        return stripped[:idx].strip(), stripped[idx:].lstrip()
    match = re.match(r"([a-z0-9,;:'\"()\-\s]+)([A-Z].*)", stripped, re.DOTALL)
    return (
        (match.group(1).strip(), match.group(2).lstrip())
        if match
        else (stripped.strip(), "")
    )


def is_bullet_list_pair(curr: str, nxt: str) -> bool:
    """Return True when ``curr`` and ``nxt`` belong to the same bullet list."""
    colon_bullet = curr.rstrip().endswith(":") or re.search(
        rf":\s*(?:[{BULLET_CHARS_ESC}]|-)", curr
    )
    has_bullet = starts_with_bullet(curr) or any(
        starts_with_bullet(line) for line in curr.splitlines()
    )
    return starts_with_bullet(nxt) and (has_bullet or colon_bullet)


def starts_with_number(text: str) -> bool:
    """Return True if ``text`` begins with a numbered list marker."""
    return bool(NUMBERED_RE.match(text))


def is_numbered_list_pair(curr: str, nxt: str) -> bool:
    """Return True when ``curr`` and ``nxt`` belong to the same numbered list."""
    has_number = starts_with_number(curr) or any(
        starts_with_number(line) for line in curr.splitlines()
    )
    return starts_with_number(nxt) and has_number


def is_numbered_continuation(curr: str, nxt: str) -> bool:
    """Return True when ``nxt`` continues a numbered item from ``curr``."""
    return (
        starts_with_number(curr)
        and not starts_with_number(nxt)
        and not curr.rstrip().endswith((".", "!", "?"))
    )
