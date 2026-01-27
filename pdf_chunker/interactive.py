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
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Unified Decision Protocol (Phase 4 Refactoring)
# ---------------------------------------------------------------------------


class DecisionKind(Enum):
    """Types of interactive decisions the pipeline can request."""

    FOOTER = "footer"  # Is this text a footer?
    LIST_CONTINUATION = "list_continuation"  # Does this continue a list item?
    PATTERN_MERGE = "pattern_merge"  # Should these texts be merged (pattern-based)?
    HEADING_BOUNDARY = "heading_boundary"  # Is this a heading boundary?


@dataclass(frozen=True)
class DecisionContext:
    """Context for an interactive decision request.

    This unified context provides all information needed for any decision type,
    allowing a single callback protocol to handle all interactive decisions.

    Attributes:
        kind: Type of decision being requested
        curr_text: The text being evaluated
        prev_text: Previous text (for merge decisions)
        page: Page number for context
        confidence: Heuristic confidence (0.0-1.0)
        pattern_name: Name of pattern that triggered the decision (if any)
        extra: Additional context-specific data
    """

    kind: DecisionKind
    curr_text: str
    prev_text: str | None = None
    page: int = 0
    confidence: float = 0.5
    pattern_name: str | None = None
    extra: dict[str, Any] | None = None


@dataclass
class Decision:
    """Result of an interactive decision.

    Attributes:
        action: The decided action ("merge", "split", or "skip")
        remember: Whether to remember this decision
            - "once": Just this instance
            - "always": Remember and apply to similar cases
            - "never": Remember and do the opposite for similar cases
        reason: Optional explanation of the decision
    """

    action: Literal["merge", "split", "skip"]
    remember: Literal["once", "always", "never"] = "once"
    reason: str | None = None


class InteractiveDecisionCallback(Protocol):
    """Unified protocol for all interactive pipeline decisions.

    This is the primary callback interface for Phase 4 refactoring.
    It handles all types of interactive decisions through a single
    entry point with typed context.

    Usage:
        def my_callback(ctx: DecisionContext) -> Decision:
            if ctx.kind == DecisionKind.FOOTER:
                is_footer = ask_user(f"Is this a footer? {ctx.curr_text[:60]}...")
                return Decision(action="split" if is_footer else "merge")
            return Decision(action="skip")  # Let heuristics decide
    """

    def __call__(self, context: DecisionContext) -> Decision:
        """Process a decision request and return the user's choice."""
        ...


# ---------------------------------------------------------------------------
# Adapter functions for legacy callbacks
# ---------------------------------------------------------------------------


def adapt_footer_callback(
    callback: FooterCallback,
) -> InteractiveDecisionCallback:
    """Wrap a legacy FooterCallback in the unified protocol.

    The footer callback returns True (is footer) -> action="split" (remove it)
    Returns False (not footer) -> action="merge" (keep it as content)
    """

    def unified(ctx: DecisionContext) -> Decision:
        if ctx.kind != DecisionKind.FOOTER:
            return Decision(action="skip")
        extra = ctx.extra or {}
        result = callback(ctx.curr_text, ctx.page, extra)
        # Footer=True means remove it (split/separate from content)
        return Decision(action="split" if result else "merge")

    return unified


def adapt_list_continuation_callback(
    callback: ListContinuationCallback,
) -> InteractiveDecisionCallback:
    """Wrap a legacy ListContinuationCallback in the unified protocol.

    The list callback returns True (continue) -> action="merge"
    Returns False (not continuation) -> action="split"
    """

    def unified(ctx: DecisionContext) -> Decision:
        if ctx.kind != DecisionKind.LIST_CONTINUATION:
            return Decision(action="skip")
        extra = ctx.extra or {}
        result = callback(ctx.prev_text or "", ctx.curr_text, ctx.page, extra)
        return Decision(action="merge" if result else "split")

    return unified


def make_unified_cli_prompt(
    *,
    max_preview: int = 80,
) -> InteractiveDecisionCallback:
    """Create a CLI-friendly unified prompt callback.

    This callback handles all decision types with appropriate prompts.
    """

    def _prompt(ctx: DecisionContext) -> Decision:
        kind = ctx.kind.value.replace("_", " ").title()
        confidence_str = f"{ctx.confidence:.0%}"

        print(f"\n--- {kind} Decision (page {ctx.page}, confidence {confidence_str}) ---")

        if ctx.kind == DecisionKind.FOOTER:
            preview = ctx.curr_text[:max_preview].replace("\n", "\\n")
            ellipsis = "..." if len(ctx.curr_text) > max_preview else ""
            print(f"  Text: {preview}{ellipsis}")
            response = input("Treat as footer (remove)? [y/N/always/never] ").strip().lower()
            is_footer = response in ("y", "yes")
            remember = (
                "always" if response == "always" else "never" if response == "never" else "once"
            )
            return Decision(action="split" if is_footer else "merge", remember=remember)

        elif ctx.kind == DecisionKind.LIST_CONTINUATION:
            item_preview = (ctx.prev_text or "")[:max_preview].replace("\n", "\\n")
            cand_preview = ctx.curr_text[:max_preview].replace("\n", "\\n")
            item_ellipsis = "..." if len(ctx.prev_text or "") > max_preview else ""
            cand_ellipsis = "..." if len(ctx.curr_text) > max_preview else ""
            print(f"  List item: {item_preview}{item_ellipsis}")
            print(f"  Candidate: {cand_preview}{cand_ellipsis}")
            response = input("Merge into list item? [Y/n/always/never] ").strip().lower()
            should_merge = response in ("", "y", "yes", "always")
            remember = (
                "always" if response == "always" else "never" if response == "never" else "once"
            )
            return Decision(action="merge" if should_merge else "split", remember=remember)

        elif ctx.kind == DecisionKind.PATTERN_MERGE:
            prev_preview = (ctx.prev_text or "")[-max_preview:].replace("\n", "\\n")
            curr_preview = ctx.curr_text[:max_preview].replace("\n", "\\n")
            pattern = ctx.pattern_name or "unknown"
            print(f"  Pattern: {pattern}")
            print(f"  Previous ends: ...{prev_preview}")
            print(f"  Current starts: {curr_preview}...")
            response = input("Keep together (merge)? [Y/n/always/never] ").strip().lower()
            should_merge = response in ("", "y", "yes", "always")
            remember = (
                "always" if response == "always" else "never" if response == "never" else "once"
            )
            return Decision(action="merge" if should_merge else "split", remember=remember)

        elif ctx.kind == DecisionKind.HEADING_BOUNDARY:
            preview = ctx.curr_text[:max_preview].replace("\n", "\\n")
            print(f"  Text: {preview}{'...' if len(ctx.curr_text) > max_preview else ''}")
            response = (
                input("Treat as heading boundary (split here)? [y/N/always/never] ").strip().lower()
            )
            is_boundary = response in ("y", "yes")
            remember = (
                "always" if response == "always" else "never" if response == "never" else "once"
            )
            return Decision(action="split" if is_boundary else "merge", remember=remember)

        # Default: skip (let heuristics decide)
        return Decision(action="skip")

    return _prompt


# ---------------------------------------------------------------------------
# Legacy Protocols (kept for backward compatibility)
# ---------------------------------------------------------------------------


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


class ListContinuationCallback(Protocol):
    """Protocol for interactive list continuation confirmation.

    Implementations receive the previous list item text, candidate continuation,
    page number, and context. Return True if the candidate should be merged
    into the previous list item.
    """

    def __call__(
        self,
        list_item: str,
        candidate: str,
        page: int,
        context: Mapping[str, Any],
    ) -> bool:
        """Return True if ``candidate`` continues ``list_item``."""
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
        overlaps = any(not (end <= s or start >= e) for (s, e) in seen_ranges)
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
            r"([A-Z][A-Za-z]*(?:\s+(?:from\s+the\s+)?[A-Za-z]+)+)\s+\d+", footer_text.strip()
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


def _make_pattern_non_greedy(pattern: str) -> str:
    """Convert greedy quantifiers to non-greedy to prevent over-matching.

    User-provided patterns like 'Collective Wisdom.*' with greedy '.*' can
    match far more text than intended when wrapped with page number suffix.
    For example, 'Collective Wisdom.*\\s+(\\d{1,3})' would match from
    'Collective Wisdom' at the start of a block all the way to '106' at the
    end, consuming the entire content as a 'footer'.

    This function converts:
      - .* -> .*?  (non-greedy)
      - .+ -> .+?  (non-greedy)
    """
    # Only convert if not already non-greedy
    result = re.sub(r"(\.\*)(?!\?)", r"\1?", pattern)
    result = re.sub(r"(\.\+)(?!\?)", r"\1?", result)
    return result


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

    Note:
        Greedy quantifiers (.* and .+) in user patterns are converted to
        non-greedy (.*? and .+?) to prevent over-matching. Without this,
        a pattern like 'Collective Wisdom.*' would consume all text between
        the first occurrence and the last page number in the block.
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
            # Make pattern non-greedy to prevent over-matching
            safe_p = _make_pattern_non_greedy(p)

            # Wrap pattern to match inline footer with \n\n prefix
            # Pattern: \n\n + user_pattern + whitespace + page_number + word_boundary
            inline_pat = rf"\n\n({safe_p})\s+(\d{{1,3}})(?=\s|$)"
            compiled.append(re.compile(inline_pat, re.IGNORECASE))

            # Also create mid-text pattern (without \n\n requirement)
            if midtext:
                # Match after sentence boundary, space, or start of text
                # This catches footers that appear mid-paragraph
                midtext_pat = rf"(?:(?<=\. )|(?<=\.\n)|(?<= )|(?<=\n)|^)({safe_p})\s+(\d{{1,3}})(?=\s|$|[.!?,])"
                compiled.append(re.compile(midtext_pat, re.IGNORECASE))
    return tuple(compiled)


# ---------------------------------------------------------------------------
# List continuation detection and confirmation
# ---------------------------------------------------------------------------

# Bullet characters for list detection
_LIST_BULLET_CHARS = frozenset("•●○◦▪▫‣⁃-–—")
_NUMBERED_PREFIX_RE = re.compile(r"^\s*(\d+)[.)]\s+")


def _looks_like_list_item(text: str) -> bool:
    """Return True if text contains a bullet or numbered marker.

    This checks both leading markers AND inline markers (e.g., after a colon).
    """
    stripped = text.lstrip()
    if not stripped:
        return False
    # Check for leading bullet
    if stripped[0] in _LIST_BULLET_CHARS:
        return True
    # Check for leading number
    if _NUMBERED_PREFIX_RE.match(stripped):
        return True
    # Check for inline bullet (after colon or at line start)
    for line in text.splitlines():
        line_stripped = line.lstrip()
        if line_stripped and line_stripped[0] in _LIST_BULLET_CHARS:
            return True
        if _NUMBERED_PREFIX_RE.match(line_stripped):
            return True
    # Check for bullet after colon (common pattern: "Here is a guide: • First item")
    if ":" in text:
        after_colon = text.split(":", 1)[-1].lstrip()
        if after_colon and after_colon[0] in _LIST_BULLET_CHARS:
            return True
    return False


def _extract_list_body(text: str) -> str | None:
    """Extract the list item body from text, or None if not a list item."""
    stripped = text.lstrip()
    if not stripped:
        return None

    # Leading bullet
    if stripped[0] in _LIST_BULLET_CHARS:
        return stripped[1:].lstrip()

    # Leading number
    match = _NUMBERED_PREFIX_RE.match(stripped)
    if match:
        return stripped[match.end() :]

    # Inline bullet after colon
    if ":" in text:
        after_colon = text.split(":", 1)[-1].lstrip()
        if after_colon and after_colon[0] in _LIST_BULLET_CHARS:
            return after_colon[1:].lstrip()

    return None


def _list_item_looks_incomplete(text: str) -> bool:
    """Return True if a list item text looks incomplete.

    A list item looks incomplete if:
    - It ends with punctuation that suggests continuation (comma, semicolon, colon)
    - It's very short (<=5 words after the marker)
    - It ends mid-sentence (no sentence-ending punctuation and short)
    - It's a short sentence that could reasonably continue (<=6 words with period)
    - It has unbalanced parentheses, brackets, or quotes
    """
    stripped = text.rstrip()
    if not stripped:
        return True

    # Extract the list item body
    body = _extract_list_body(stripped)
    if body is None:
        # Not a list item, use full text
        body = stripped

    # Continuation punctuation at end
    if body and body[-1] in ",;:":
        return True

    # Very short items are suspicious (<=3 words is definitely incomplete)
    words = body.split()
    if len(words) <= 3:
        return True

    # Short items without sentence-ending punctuation
    if len(words) <= 7 and body[-1] not in ".!?":
        return True

    # Short items even with period may be incomplete (<=5 words)
    # This catches cases like "• Reduce wordiness." which is really just
    # the first sentence of a longer list item
    if len(words) <= 5:
        return True

    # Check for unbalanced delimiters (parens, brackets, quotes)
    open_parens = body.count("(") - body.count(")")
    open_brackets = body.count("[") - body.count("]")
    open_braces = body.count("{") - body.count("}")
    if open_parens > 0 or open_brackets > 0 or open_braces > 0:
        return True

    # Check for unbalanced quotes (odd count of each type)
    # Apostrophe is tricky (used in contractions), skip for now
    return body.count('"') % 2 != 0


def _candidate_continues_list_item(
    list_item: str,
    candidate: str,
    *,
    confidence_threshold: float = 0.7,
) -> tuple[bool, float, str]:
    """Heuristically determine if candidate continues a list item.

    Returns:
        Tuple of (should_merge, confidence, reason)
    """
    if not list_item or not candidate:
        return False, 0.0, "empty_input"

    if not _looks_like_list_item(list_item):
        return False, 0.0, "not_a_list_item"

    # If candidate starts with a bullet, it's a new item not a continuation
    if _looks_like_list_item(candidate):
        return False, 0.0, "candidate_is_new_item"

    # Check if the list item looks incomplete
    incomplete = _list_item_looks_incomplete(list_item)
    if not incomplete:
        # List item looks complete - lower confidence for merging
        confidence = 0.3
        reason = "item_looks_complete"
    else:
        confidence = 0.8
        reason = "item_looks_incomplete"

    # Adjust confidence based on candidate characteristics
    candidate_stripped = candidate.strip()
    first_char = candidate_stripped[0] if candidate_stripped else ""

    # Lower case start suggests continuation
    if first_char.islower():
        confidence = min(1.0, confidence + 0.15)
        reason = f"{reason}+lowercase_start"
    # Conjunction/continuation words
    elif first_char.isupper():
        first_word = candidate_stripped.split()[0].lower() if candidate_stripped.split() else ""
        continuation_words = {
            "and",
            "or",
            "but",
            "which",
            "that",
            "because",
            "since",
            "although",
            "however",
            "therefore",
            "thus",
            "also",
            "as",
            "if",
            "when",
            "where",
            "for",
            "to",
            "with",
            "without",
        }
        if first_word in continuation_words:
            confidence = min(1.0, confidence + 0.1)
            reason = f"{reason}+continuation_word"

    return confidence >= confidence_threshold, confidence, reason


@dataclass
class ListContinuationConfig:
    """Configuration for list continuation detection.

    Attributes:
        callback: Optional callback for interactive confirmation
        cache_decisions: Whether to remember user decisions
        confidence_threshold: Heuristic confidence below which to prompt
        auto_merge_threshold: Confidence above which to auto-merge without prompt
    """

    callback: ListContinuationCallback | None = None
    cache_decisions: bool = True
    confidence_threshold: float = 0.6
    auto_merge_threshold: float = 0.9


@dataclass
class ListContinuationCache:
    """Remembers user decisions for list continuation patterns."""

    _decisions: dict[str, bool] = field(default_factory=dict)

    def _normalize(self, list_item: str, candidate: str) -> str:
        """Create a cache key from the item and candidate."""
        # Use first 50 chars of each for the key
        item_key = list_item.strip()[:50]
        cand_key = candidate.strip()[:50]
        return f"{item_key}|{cand_key}"

    def get(self, list_item: str, candidate: str) -> bool | None:
        """Return cached decision or None if not cached."""
        key = self._normalize(list_item, candidate)
        return self._decisions.get(key)

    def set(self, list_item: str, candidate: str, should_merge: bool) -> None:
        """Cache a decision."""
        key = self._normalize(list_item, candidate)
        self._decisions[key] = should_merge


def classify_list_continuation(
    list_item: str,
    candidate: str,
    page: int,
    *,
    config: ListContinuationConfig,
    cache: ListContinuationCache | None = None,
) -> tuple[bool, str]:
    """Classify whether candidate should merge with list_item.

    Returns:
        (should_merge, reason) tuple
    """
    should_merge, confidence, heuristic_reason = _candidate_continues_list_item(
        list_item,
        candidate,
        confidence_threshold=config.confidence_threshold,
    )

    # Check cache first
    if cache is not None and config.cache_decisions:
        cached = cache.get(list_item, candidate)
        if cached is not None:
            return cached, "cached_decision"

    # High confidence - auto-decide
    if confidence >= config.auto_merge_threshold:
        return True, f"auto_merge ({confidence:.0%})"

    if confidence <= (1.0 - config.auto_merge_threshold):
        return False, f"auto_reject ({confidence:.0%})"

    # Interactive callback for uncertain cases
    if config.callback is not None:
        ctx = {
            "heuristic_confidence": confidence,
            "heuristic_reason": heuristic_reason,
            "page": page,
        }
        decision = config.callback(list_item, candidate, page, ctx)
        if cache is not None and config.cache_decisions:
            cache.set(list_item, candidate, decision)
        return decision, "user_confirmation"

    # Fallback to heuristic
    return should_merge, f"heuristic ({confidence:.0%})"


def make_cli_list_continuation_prompt(
    *,
    max_preview: int = 60,
    default_merge: bool = True,
) -> ListContinuationCallback:
    """Create a CLI-friendly list continuation prompt callback."""

    def _prompt(
        list_item: str,
        candidate: str,
        page: int,
        context: Mapping[str, Any],
    ) -> bool:
        item_preview = list_item[:max_preview].replace("\n", "\\n")
        cand_preview = candidate[:max_preview].replace("\n", "\\n")
        item_ellipsis = "..." if len(list_item) > max_preview else ""
        cand_ellipsis = "..." if len(candidate) > max_preview else ""
        confidence = context.get("heuristic_confidence", 0.5)
        reason = context.get("heuristic_reason", "unknown")

        print(f"\n--- List continuation candidate (page {page}, confidence {confidence:.0%}) ---")
        print(f"  List item: {item_preview}{item_ellipsis}")
        print(f"  Candidate: {cand_preview}{cand_ellipsis}")
        print(f"  Heuristic: {reason}")
        default_hint = "[Y/n]" if default_merge else "[y/N]"
        response = input(f"Merge into list item? {default_hint} ").strip().lower()

        if not response:
            return default_merge
        return response in ("y", "yes", "1", "true")

    return _prompt


def make_batch_list_continuation_prompt(
    decisions: Mapping[str, bool],
) -> ListContinuationCallback:
    """Create a callback that looks up decisions from a pre-defined mapping."""

    def _lookup(
        list_item: str,
        candidate: str,
        page: int,
        context: Mapping[str, Any],
    ) -> bool:
        # Try with item text as key
        for key, val in decisions.items():
            if key in list_item or key in candidate:
                return val
        # Default to heuristic
        return context.get("heuristic_confidence", 0.5) >= 0.5

    return _lookup
