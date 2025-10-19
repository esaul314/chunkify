"""Bullet list heuristics packaged as a pure strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import takewhile
from typing import Pattern, Tuple


@dataclass(frozen=True)
class BulletHeuristicStrategy:
    """Encapsulate bullet and numbered list heuristics as pure callables."""

    bullet_chars: str = "*•◦▪‣·●◉○‧"
    hyphen_bullet_prefix: str = "- "
    numbered_pattern: str = r"(?P<number>\d+)[.)]\s+(?P<body>.+)"
    bullet_chars_esc: str = field(init=False)
    leading_bullet_re: Pattern[str] = field(init=False)
    leading_hyphen_re: Pattern[str] = field(init=False)
    inline_colon_bullet_re: Pattern[str] = field(init=False)
    numbered_re: Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        bullet_chars_esc = re.escape(self.bullet_chars)
        object.__setattr__(self, "bullet_chars_esc", bullet_chars_esc)
        object.__setattr__(
            self,
            "leading_bullet_re",
            re.compile(rf"^\s*(?:[{bullet_chars_esc}]\s+)"),
        )
        object.__setattr__(
            self,
            "leading_hyphen_re",
            re.compile(r"^\s*-\s+"),
        )
        object.__setattr__(
            self,
            "inline_colon_bullet_re",
            re.compile(rf":\s*(?:[{bullet_chars_esc}]\s+|-\s+)"),
        )
        object.__setattr__(
            self,
            "numbered_re",
            re.compile(self.numbered_pattern),
        )

    # ------------------------------------------------------------------
    # Bullet helpers
    # ------------------------------------------------------------------
    def starts_with_bullet(self, text: str) -> bool:
        """Return ``True`` when ``text`` begins with a bullet marker."""

        return bool(
            self.leading_bullet_re.match(text)
            or self.leading_hyphen_re.match(text)
        )

    def last_non_empty_line(self, text: str) -> str:
        """Return the trailing non-empty line from ``text``."""

        return next(
            (line.strip() for line in reversed(text.splitlines()) if line.strip()),
            "",
        )

    def is_bullet_continuation(self, curr: str, nxt: str) -> bool:
        """Return ``True`` if ``nxt`` continues a bullet marker from ``curr``."""

        last_line = self.last_non_empty_line(curr)
        return bool(last_line.endswith(tuple(self.bullet_chars)) and nxt[:1].islower())

    def is_bullet_fragment(self, curr: str, nxt: str) -> bool:
        """Return ``True`` when ``nxt`` extends the last bullet line in ``curr``."""

        last_line = self.last_non_empty_line(curr)
        return (
            self.starts_with_bullet(last_line)
            and not last_line.rstrip().endswith((".", "!", "?"))
            and nxt[:1].islower()
        )

    def split_bullet_fragment(self, text: str) -> Tuple[str, str]:
        """Split ``text`` into its head line and remainder."""

        if "\n" not in text:
            return text.strip(), ""
        first, rest = text.split("\n", 1)
        return first.strip(), rest.lstrip()

    def block_contains_bullet_marker(self, text: str) -> bool:
        """Return ``True`` if any line in ``text`` begins with a bullet marker."""

        return self.starts_with_bullet(text) or any(
            self.starts_with_bullet(line) for line in text.splitlines()
        )

    def colon_leads_bullet_list(self, text: str) -> bool:
        """Return ``True`` when a trailing colon signals inline bullet leaders."""

        stripped = text.rstrip()
        inline_marker = bool(self.inline_colon_bullet_re.search(text))
        has_bullet = self.block_contains_bullet_marker(text)
        return inline_marker or (stripped.endswith(":") and has_bullet)

    def is_bullet_list_pair(self, curr: str, nxt: str) -> bool:
        """Return ``True`` when ``curr`` and ``nxt`` belong to the same bullet list."""

        if not self.starts_with_bullet(nxt):
            return False
        has_bullet = self.block_contains_bullet_marker(curr)
        return has_bullet or self.colon_leads_bullet_list(curr)

    # ------------------------------------------------------------------
    # Numbered helpers
    # ------------------------------------------------------------------
    def numbered_body(self, text: str) -> str | None:
        """Return the numbered list body if present."""

        if not text or not text[0].isdigit():
            return None
        digits = "".join(takewhile(str.isdigit, text))
        remainder = text[len(digits) :]
        if not remainder or remainder[0] not in ".)":
            return None
        tail = remainder[1:].lstrip(" \t")
        return tail

    def starts_with_number(self, text: str) -> bool:
        """Return ``True`` when ``text`` begins with a numbered marker."""

        return self.numbered_body(text) is not None

    def is_numbered_list_pair(self, curr: str, nxt: str) -> bool:
        """Return ``True`` when ``curr`` and ``nxt`` form a numbered list pair."""

        has_number = self.starts_with_number(curr) or any(
            self.starts_with_number(line) for line in curr.splitlines()
        )
        return self.starts_with_number(nxt) and has_number

    def is_numbered_continuation(self, curr: str, nxt: str) -> bool:
        """Return ``True`` when ``nxt`` continues a numbered list item from ``curr``."""

        return (
            self.starts_with_number(curr)
            and not self.starts_with_number(nxt)
            and not curr.rstrip().endswith((".", "!", "?"))
        )


def _resolve(strategy: BulletHeuristicStrategy | None) -> BulletHeuristicStrategy:
    return strategy or DEFAULT_STRATEGY


def starts_with_bullet(text: str, strategy: BulletHeuristicStrategy | None = None) -> bool:
    return _resolve(strategy).starts_with_bullet(text)


def last_non_empty_line(text: str, strategy: BulletHeuristicStrategy | None = None) -> str:
    return _resolve(strategy).last_non_empty_line(text)


def is_bullet_continuation(
    curr: str,
    nxt: str,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    return _resolve(strategy).is_bullet_continuation(curr, nxt)


def is_bullet_fragment(
    curr: str,
    nxt: str,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    return _resolve(strategy).is_bullet_fragment(curr, nxt)


def split_bullet_fragment(
    text: str,
    strategy: BulletHeuristicStrategy | None = None,
) -> Tuple[str, str]:
    return _resolve(strategy).split_bullet_fragment(text)


def is_bullet_list_pair(
    curr: str,
    nxt: str,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    return _resolve(strategy).is_bullet_list_pair(curr, nxt)


def starts_with_number(text: str, strategy: BulletHeuristicStrategy | None = None) -> bool:
    return _resolve(strategy).starts_with_number(text)


def is_numbered_list_pair(
    curr: str,
    nxt: str,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    return _resolve(strategy).is_numbered_list_pair(curr, nxt)


def is_numbered_continuation(
    curr: str,
    nxt: str,
    strategy: BulletHeuristicStrategy | None = None,
) -> bool:
    return _resolve(strategy).is_numbered_continuation(curr, nxt)


DEFAULT_STRATEGY = BulletHeuristicStrategy()

BULLET_CHARS = DEFAULT_STRATEGY.bullet_chars
BULLET_CHARS_ESC = DEFAULT_STRATEGY.bullet_chars_esc
HYPHEN_BULLET_PREFIX = DEFAULT_STRATEGY.hyphen_bullet_prefix
NUMBERED_RE = DEFAULT_STRATEGY.numbered_re


def default_bullet_strategy() -> BulletHeuristicStrategy:
    """Return the immutable default bullet heuristic strategy."""

    return DEFAULT_STRATEGY


__all__ = [
    "BulletHeuristicStrategy",
    "BULLET_CHARS",
    "BULLET_CHARS_ESC",
    "HYPHEN_BULLET_PREFIX",
    "NUMBERED_RE",
    "DEFAULT_STRATEGY",
    "default_bullet_strategy",
    "starts_with_bullet",
    "last_non_empty_line",
    "is_bullet_continuation",
    "is_bullet_fragment",
    "split_bullet_fragment",
    "is_bullet_list_pair",
    "starts_with_number",
    "is_numbered_list_pair",
    "is_numbered_continuation",
]
