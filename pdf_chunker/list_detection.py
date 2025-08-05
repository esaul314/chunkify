import re

BULLET_CHARS = "*•◦▪‣·●◉○‧"
BULLET_CHARS_ESC = re.escape(BULLET_CHARS)
NUMBERED_RE = re.compile(r"\s*\d+[.)]")


def starts_with_bullet(text: str) -> bool:
    """Return True if ``text`` begins with a bullet character."""
    return text.lstrip().startswith(tuple(BULLET_CHARS))


def is_bullet_continuation(curr: str, nxt: str) -> bool:
    """Return True when ``nxt`` continues a bullet item from ``curr``."""
    return curr.rstrip().endswith(tuple(BULLET_CHARS)) and nxt[:1].islower()


def is_bullet_list_pair(curr: str, nxt: str) -> bool:
    """Return True when ``curr`` and ``nxt`` belong to the same bullet list."""
    colon_bullet = re.search(rf":\s*[{BULLET_CHARS_ESC}]", curr)
    has_bullet = starts_with_bullet(curr) or any(
        starts_with_bullet(line) for line in curr.splitlines()
    )
    return starts_with_bullet(nxt) and (has_bullet or colon_bullet is not None)


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
