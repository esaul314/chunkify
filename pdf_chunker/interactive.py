"""Interactive user callbacks for ambiguous pipeline decisions.

This module provides a protocol-based callback mechanism that lets CLI or GUI
consumers confirm uncertain classifications (e.g., footer vs. body text).

Design philosophy:
- Pure functions by default; IO happens only through explicit callbacks
- Adapters at the boundary: this module defines the **interface**, not the
  implementation of stdin/stdout prompts—those live in the CLI layer
- Configuration precedence: explicit patterns > interactive prompts > heuristics

Usage:
    # Define your callback (e.g., in CLI)
    def my_footer_prompt(text: str, page: int, context: dict) -> bool:
        return input(f"Is this a footer? '{text[:60]}...' [y/N] ").lower() == 'y'

    # Pass it through pipeline options
    spec = load_spec("pipeline.yaml", overrides={
        "detect_page_artifacts": {
            "footer_callback": my_footer_prompt,
            "known_footer_patterns": ["Collective Wisdom.*\\d+"],
        }
    })
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass


class FooterCallback(Protocol):
    """Protocol for interactive footer confirmation.

    Implementations receive candidate footer text, page number, and context
    and return True if the text should be treated as a footer (stripped).
    """

    def __call__(
        self,
        text: str,
        page: int,
        context: Mapping[str, Any],
    ) -> bool:
        """Return True if ``text`` should be classified as footer."""
        ...


@dataclass(frozen=True)
class FooterConfig:
    """Configuration for footer detection behaviour.

    Attributes:
        known_patterns: Regex patterns that always match footers
        never_patterns: Regex patterns that never match footers (body text)
        callback: Optional callback for ambiguous cases
        cache_decisions: Whether to remember user decisions for similar text
        confidence_threshold: Heuristic confidence below which to prompt
    """

    known_patterns: tuple[re.Pattern[str], ...] = ()
    never_patterns: tuple[re.Pattern[str], ...] = ()
    callback: FooterCallback | None = None
    cache_decisions: bool = True
    confidence_threshold: float = 0.7

    @classmethod
    def from_dict(cls, opts: Mapping[str, Any]) -> FooterConfig:
        """Build config from dictionary options (e.g., from YAML spec)."""
        known = opts.get("known_footer_patterns") or []
        never = opts.get("never_footer_patterns") or []
        return cls(
            known_patterns=tuple(re.compile(p, re.IGNORECASE) for p in known if isinstance(p, str)),
            never_patterns=tuple(re.compile(p, re.IGNORECASE) for p in never if isinstance(p, str)),
            callback=opts.get("footer_callback"),
            cache_decisions=opts.get("cache_decisions", True),
            confidence_threshold=opts.get("confidence_threshold", 0.7),
        )


@dataclass
class FooterDecisionCache:
    """Remembers user decisions for similar footer candidates.

    Normalizes text before comparison so slight variations match.
    """

    _is_footer: dict[str, bool] = field(default_factory=dict)

    def _normalize(self, text: str) -> str:
        """Strip page numbers and normalize whitespace."""
        stripped = re.sub(r"\s+", " ", text.strip())
        # Remove trailing digits (page numbers)
        return re.sub(r"\s*\d+\s*$", "", stripped)

    def get(self, text: str) -> bool | None:
        """Return cached decision or None if not cached."""
        key = self._normalize(text)
        return self._is_footer.get(key)

    def set(self, text: str, is_footer: bool) -> None:
        """Cache a decision for ``text``."""
        key = self._normalize(text)
        self._is_footer[key] = is_footer


def classify_footer(
    text: str,
    page: int,
    *,
    config: FooterConfig,
    cache: FooterDecisionCache | None = None,
    heuristic_confidence: float = 0.5,
    context: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Classify ``text`` as footer or body with explanation.

    Returns:
        (is_footer, reason) tuple where reason explains the classification.

    Classification order:
        1. Explicit ``known_patterns`` → footer
        2. Explicit ``never_patterns`` → body
        3. Cached decision → use cache
        4. High heuristic confidence → use heuristic
        5. Interactive callback (if provided) → prompt user
        6. Fall back to heuristic
    """
    ctx = dict(context or {})

    # 1. Explicit known patterns
    for pat in config.known_patterns:
        if pat.search(text):
            return True, f"matched known footer pattern: {pat.pattern}"

    # 2. Explicit never patterns
    for pat in config.never_patterns:
        if pat.search(text):
            return False, f"matched never-footer pattern: {pat.pattern}"

    # 3. Check cache
    if cache is not None and config.cache_decisions:
        cached = cache.get(text)
        if cached is not None:
            return cached, "cached user decision"

    # 4. High confidence heuristic
    if heuristic_confidence >= config.confidence_threshold:
        is_footer = heuristic_confidence >= 0.5
        return is_footer, f"heuristic confidence {heuristic_confidence:.2f}"

    # 5. Interactive callback
    if config.callback is not None:
        ctx["heuristic_confidence"] = heuristic_confidence
        ctx["page"] = page
        decision = config.callback(text, page, ctx)
        if cache is not None and config.cache_decisions:
            cache.set(text, decision)
        return decision, "user confirmation"

    # 6. Fallback to heuristic
    is_footer = heuristic_confidence >= 0.5
    return is_footer, f"heuristic fallback (confidence {heuristic_confidence:.2f})"


# ---------------------------------------------------------------------------
# CLI prompt helpers (for use by CLI layer)
# ---------------------------------------------------------------------------


def make_cli_footer_prompt(
    *, max_preview: int = 80, default_is_footer: bool = False
) -> FooterCallback:
    """Create a CLI-friendly footer prompt callback.

    This function returns a callback suitable for interactive CLI usage.
    It prints the candidate text and asks for y/n confirmation.
    """

    def _prompt(text: str, page: int, context: Mapping[str, Any]) -> bool:
        preview = text[:max_preview].replace("\n", "\\n")
        ellipsis = "..." if len(text) > max_preview else ""
        confidence = context.get("heuristic_confidence", 0.5)

        print(f"\n--- Footer candidate (page {page}, confidence {confidence:.0%}) ---")
        print(f"  {preview}{ellipsis}")
        default_hint = "[Y/n]" if default_is_footer else "[y/N]"
        response = input(f"Treat as footer? {default_hint} ").strip().lower()

        if not response:
            return default_is_footer
        return response in ("y", "yes", "1", "true")

    return _prompt


def make_batch_footer_prompt(decisions: Mapping[str, bool]) -> FooterCallback:
    """Create a callback that looks up decisions from a pre-defined mapping.

    Useful for non-interactive batch runs or testing. The mapping keys should
    be normalized text (or substring matches).
    """

    def _lookup(text: str, page: int, context: Mapping[str, Any]) -> bool:
        # Try exact match first
        if text in decisions:
            return decisions[text]
        # Try substring match
        for key, val in decisions.items():
            if key in text or text in key:
                return val
        # Default to heuristic
        return context.get("heuristic_confidence", 0.5) >= 0.5

    return _lookup
