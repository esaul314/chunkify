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
        "text_clean": {
            "footer_patterns": ["Collective Wisdom.*\\d+", "Chapter \\d+.*\\d+$"],
        }
    })

Inline Footer Pattern:
    Footers often appear merged mid-text with a pattern like:
        "...previous text\\n\\nChapter Title 202 next text continues..."

    Use patterns that match this structure:
        - "\\n\\n[A-Z][^\\n]{0,50}\\s+\\d{1,3}(?=\\s)"  # Generic chapter footer
        - "\\n\\nScale Communication.*?\\d{1,3}(?=\\s)"  # Specific book title
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


# ---------------------------------------------------------------------------
# Heuristic inline footer detection (pattern-free)
# ---------------------------------------------------------------------------

# Generic pattern for inline footers: \n\n + TitleCase words + page number
# Matches things like:
#   "\n\nScale Communication Through Writing 202\n\naside we can"
#   "\n\nScale Communication Through Writing 202 Aside from"
#   "\n\nChapter 5\n\n"
#   "\n\nThe Art of Leadership 123 "
_INLINE_FOOTER_HEURISTIC = re.compile(
    r"\n\n"  # paragraph break
    r"([A-Z][A-Za-z]*(?:\s+[A-Za-z]+){1,6})"  # Title: Cap word + 1-6 more words
    r"\s+"  # whitespace before page number
    r"(\d{1,3})"  # 1-3 digit page number
    r"(?=\s)",  # followed by any whitespace (space, newline, etc.)
    re.UNICODE,
)

# Extended heuristic for mid-text footers that may not have \n\n prefix
# Detects patterns like:
#   "...having an Collective Wisdom from the Experts 153 explicit..."
#   "...as your team grows. Collective Wisdom from the Experts 169"
#   "Collective Wisdom from the Experts 149 Operating expenditures..."
#   "On Accountability 160" (2-word titles)
#   "Not Technology 164" (2-word titles)
# Uses sentence boundary detection (., !, ?) or word boundary before the title
_MIDTEXT_FOOTER_HEURISTIC = re.compile(
    r"(?:(?<=\. )|(?<=! )|(?<= )|(?<=\n)|^)"  # After sentence end, space, newline, or start
    r"([A-Z][A-Za-z]*(?:\s+(?:from\s+the\s+)?[A-Za-z]+){1,8})"  # Title with 1-8 additional words
    r"\s+"  # whitespace before page number
    r"(\d{1,3})"  # 1-3 digit page number
    r"(?=\s|$|[.!?,])",  # followed by whitespace, end, or punctuation
    re.UNICODE,
)

# Pattern for standalone footer blocks (entire block is a footer)
# Matches things like:
#   "Scale Communication Through Writing 1"
#   "Chapter 5"
#   "On Accountability 160"
_STANDALONE_FOOTER_HEURISTIC = re.compile(
    r"^([A-Z][A-Za-z]*(?:\s+[A-Za-z]+){1,6})\s+(\d{1,3})$",
    re.UNICODE,
)


def is_standalone_footer_candidate(text: str) -> tuple[str, str] | None:
    """Check if text is a standalone footer block (entire block is footer-like).

    Returns:
        Tuple of (title, page_number) if it matches, None otherwise.
    """
    text = text.strip()
    match = _STANDALONE_FOOTER_HEURISTIC.match(text)
    if match:
        return match.group(1), match.group(2)
    return None


def detect_inline_footer_candidates(text: str) -> list[tuple[str, int, int]]:
    """Detect potential inline footers heuristically (without user-provided patterns).

    Looks for the pattern: \\n\\n{TitleCase Text} {PageNumber} {Continuation}
    Also detects mid-text footers without \\n\\n prefix.

    Returns:
        List of (footer_text, start_pos, end_pos) tuples for each candidate.
    """
    candidates = []
    seen_ranges: set[tuple[int, int]] = set()
    
    # First check for \n\n prefixed footers (high confidence)
    for match in _INLINE_FOOTER_HEURISTIC.finditer(text):
        footer_text = match.group(0).strip()
        pos = (match.start(), match.end())
        if pos not in seen_ranges:
            candidates.append((footer_text, match.start(), match.end()))
            seen_ranges.add(pos)
    
    # Then check for mid-text footers (lower confidence, may need user confirmation)
    for match in _MIDTEXT_FOOTER_HEURISTIC.finditer(text):
        footer_text = match.group(0).strip()
        start, end = match.start(), match.end()
        # Avoid duplicates with \n\n matches
        overlaps = any(
            not (end <= s or start >= e) for (s, e) in seen_ranges
        )
        if not overlaps:
            candidates.append((footer_text, start, end))
            seen_ranges.add((start, end))
    
    return candidates


def strip_inline_footers_interactive(
    text: str,
    *,
    callback: FooterCallback,
    cache: FooterDecisionCache | None = None,
    page: int = 0,
) -> tuple[str, list[str]]:
    """Remove inline footers detected heuristically, confirming each with callback.

    This is the true interactive mode: it detects footer candidates without
    user-provided patterns and asks for confirmation on each.

    Args:
        text: The text to clean
        callback: Callback for interactive confirmation (required)
        cache: Optional cache for remembering decisions
        page: Page number for context

    Returns:
        Tuple of (cleaned text, list of stripped footer strings)
    """
    stripped: list[str] = []
    
    # Collect all candidates with their positions
    candidates = detect_inline_footer_candidates(text)
    if not candidates:
        return text, stripped
    
    # Sort by position (reverse order so we can replace from end to start)
    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    result = text
    for footer_text, start, end in candidates_sorted:
        # Extract title (first part before the page number)
        title_match = re.match(
            r"([A-Z][A-Za-z]*(?:\s+(?:from\s+the\s+)?[A-Za-z]+)+)\s+\d+",
            footer_text.strip()
        )
        title = title_match.group(1) if title_match else footer_text.strip()
        page_num_match = re.search(r"\d+$", footer_text.strip())
        page_num = page_num_match.group(0) if page_num_match else "?"
        
        # Check cache first
        if cache is not None:
            cached = cache.get(title)
            if cached is not None:
                if cached:
                    stripped.append(footer_text.strip())
                    # Preserve paragraph break if there was \n\n, else just space
                    replacement = "\n\n" if result[start:end].startswith("\n\n") else " "
                    result = result[:start] + replacement + result[end:]
                continue  # Skip callback if cached
        
        # Determine confidence based on pattern type
        # \n\n footers are higher confidence
        has_newline_prefix = result[start:end].startswith("\n\n")
        confidence = 0.8 if has_newline_prefix else 0.6
        
        # Ask user
        ctx = {
            "inline": True,
            "page": page,
            "detected_page_num": page_num,
            "heuristic_confidence": confidence,
            "heuristic": True,
            "midtext": not has_newline_prefix,
        }
        is_footer = callback(footer_text.strip(), page, ctx)
        
        # Cache by title (not full text) so similar footers are auto-handled
        if cache is not None:
            cache.set(title, is_footer)
        
        if is_footer:
            stripped.append(footer_text.strip())
            # Preserve paragraph break if there was \n\n, else just space
            replacement = "\n\n" if has_newline_prefix else " "
            result = result[:start] + replacement + result[end:]
    
    return result, stripped


# ---------------------------------------------------------------------------
# Inline footer stripping (footers merged into text)
# ---------------------------------------------------------------------------


def build_inline_footer_pattern(title_pattern: str) -> re.Pattern[str]:
    """Build a regex that matches inline footers with the given title pattern.

    Inline footers typically appear as:
        "...previous text\\n\\nChapter Title 123 next text..."

    Args:
        title_pattern: Regex pattern for the footer title (e.g., "Scale Communication")

    Returns:
        Compiled regex that matches the full inline footer including newlines.
    """
    # Match: \n\n + title + optional whitespace + 1-3 digit page number + word boundary
    return re.compile(
        rf"\n\n({title_pattern})\s+(\d{{1,3}})(?=\s|$)",
        re.IGNORECASE,
    )


def strip_inline_footers(
    text: str,
    patterns: tuple[re.Pattern[str], ...],
    *,
    callback: FooterCallback | None = None,
    cache: FooterDecisionCache | None = None,
    page: int = 0,
) -> tuple[str, list[str]]:
    """Remove inline footers from text and return (cleaned_text, stripped_footers).

    Inline footers are detected by patterns that match text preceded by \\n\\n
    and ending with a page number, or mid-text footers after sentence boundaries.

    Args:
        text: The text to clean
        patterns: Compiled regex patterns to match inline footers
        callback: Optional callback for interactive confirmation
        cache: Optional cache for remembering decisions
        page: Page number for context

    Returns:
        Tuple of (cleaned text, list of stripped footer strings)
    """
    stripped: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        footer_text = match.group(0)
        has_newline_prefix = footer_text.startswith("\n\n")
        
        # If we have a callback, ask for confirmation
        if callback is not None:
            confidence = 0.9 if has_newline_prefix else 0.8
            ctx = {"inline": True, "page": page, "heuristic_confidence": confidence}
            if cache is not None:
                cached = cache.get(footer_text)
                if cached is not None:
                    if cached:
                        stripped.append(footer_text.strip())
                        return "\n\n" if has_newline_prefix else " "
                    return match.group(0)  # Keep footer as-is

            is_footer = callback(footer_text.strip(), page, ctx)
            if cache is not None:
                cache.set(footer_text, is_footer)
            if not is_footer:
                return match.group(0)

        stripped.append(footer_text.strip())
        # Preserve paragraph break if there was \n\n, else replace with space
        return "\n\n" if has_newline_prefix else " "

    result = text
    for pattern in patterns:
        result = pattern.sub(_replace, result)

    return result, stripped


def compile_footer_patterns(
    patterns: tuple[str, ...],
    *,
    inline: bool = True,
    midtext: bool = True,
) -> tuple[re.Pattern[str], ...]:
    """Compile footer pattern strings to regex objects.

    Args:
        patterns: Tuple of regex pattern strings
        inline: If True, wrap patterns to match inline footers (\\n\\n prefix + page number)
        midtext: If True, also create patterns for mid-text footers (no \\n\\n required)

    Returns:
        Tuple of compiled regex patterns
    """
    if not inline:
        return tuple(re.compile(p, re.IGNORECASE) for p in patterns if isinstance(p, str))

    compiled = []
    for p in patterns:
        if not isinstance(p, str):
            continue
        # Check if pattern already handles inline structure
        if r"\n\n" in p or p.startswith("^"):
            compiled.append(re.compile(p, re.IGNORECASE))
        else:
            # Wrap pattern to match inline footer with \n\n prefix
            # Pattern: \n\n + user_pattern + whitespace + page_number + word_boundary
            inline_pat = rf"\n\n({p})\s+(\d{{1,3}})(?=\s|$)"
            compiled.append(re.compile(inline_pat, re.IGNORECASE))
            
            # Also create mid-text pattern (without \n\n requirement)
            if midtext:
                # Match after sentence boundary, space, or start of text
                # This catches footers that appear mid-paragraph
                midtext_pat = rf"(?:(?<=\. )|(?<=\.\n)|(?<= )|(?<=\n)|^)({p})\s+(\d{{1,3}})(?=\s|$|[.!?,])"
                compiled.append(re.compile(midtext_pat, re.IGNORECASE))
    return tuple(compiled)
