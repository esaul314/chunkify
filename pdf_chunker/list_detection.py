"""Functional shims for list detection helpers.

The pure strategy lives in :mod:`pdf_chunker.strategies.bullets`. This module
re-exports its callables to preserve the legacy functional API relied upon by
downstream consumers.
"""

from pdf_chunker.strategies.bullets import (
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

_last_non_empty_line = last_non_empty_line

__all__ = [
    "BULLET_CHARS",
    "BULLET_CHARS_ESC",
    "HYPHEN_BULLET_PREFIX",
    "NUMBERED_RE",
    "DEFAULT_STRATEGY",
    "BulletHeuristicStrategy",
    "default_bullet_strategy",
    "starts_with_bullet",
    "is_bullet_continuation",
    "is_bullet_fragment",
    "split_bullet_fragment",
    "is_bullet_list_pair",
    "starts_with_number",
    "is_numbered_list_pair",
    "is_numbered_continuation",
    "_last_non_empty_line",
]
