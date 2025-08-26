"""Backward-compatible shims for list detection helpers.

The implementations now live in :mod:`pdf_chunker.passes.list_detect`.
This module re-exports those functions to preserve the public API.
"""

from pdf_chunker.passes.list_detect import (
    BULLET_CHARS,
    BULLET_CHARS_ESC,
    HYPHEN_BULLET_PREFIX,
    NUMBERED_RE,
    is_bullet_continuation,
    is_bullet_fragment,
    is_bullet_list_pair,
    _last_non_empty_line,
    is_numbered_continuation,
    is_numbered_list_pair,
    split_bullet_fragment,
    starts_with_bullet,
    starts_with_number,
)

__all__ = [
    "BULLET_CHARS",
    "BULLET_CHARS_ESC",
    "HYPHEN_BULLET_PREFIX",
    "NUMBERED_RE",
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
