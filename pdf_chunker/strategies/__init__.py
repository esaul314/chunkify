"""Strategy objects encapsulating reusable heuristics."""

from .bullets import (  # noqa: F401
    BULLET_CHARS,
    BULLET_CHARS_ESC,
    DEFAULT_STRATEGY,
    HYPHEN_BULLET_PREFIX,
    NUMBERED_RE,
    BulletHeuristicStrategy,
    default_bullet_strategy,
    is_bullet_continuation,
    is_bullet_fragment,
    is_bullet_list_pair,
    is_numbered_continuation,
    is_numbered_list_pair,
    last_non_empty_line,
    split_bullet_fragment,
    starts_with_bullet,
    starts_with_number,
)

__all__ = [
    "BulletHeuristicStrategy",
    "BULLET_CHARS",
    "BULLET_CHARS_ESC",
    "HYPHEN_BULLET_PREFIX",
    "NUMBERED_RE",
    "DEFAULT_STRATEGY",
    "default_bullet_strategy",
    "starts_with_bullet",
    "is_bullet_continuation",
    "is_bullet_fragment",
    "split_bullet_fragment",
    "is_bullet_list_pair",
    "starts_with_number",
    "is_numbered_list_pair",
    "is_numbered_continuation",
    "last_non_empty_line",
]
