"""Centralized pattern registry for text merge/split decisions.

This module consolidates all pattern detection logic that was previously
scattered across split_semantic.py, sentence_fusion.py, page_artifacts.py,
and other modules. Each pattern has explicit precedence and behavior.

Design philosophy:
- Declarative over imperative: patterns are data, not nested conditionals
- Precedence is explicit: CRITICAL > HIGH > MEDIUM > LOW > BOUNDARY
- Interactive learning: ambiguous patterns can prompt users
- Testable in isolation: each pattern can be unit-tested independently

Usage:
    registry = PatternRegistry()  # Uses DEFAULT_PATTERNS

    # Check if texts should be merged
    decision = registry.should_merge(prev_text, curr_text)
    if decision.should_merge:
        merged = prev_text + " " + curr_text

    # With interactive callback for ambiguous cases
    decision = registry.should_merge(
        prev_text, curr_text,
        interactive_callback=my_prompt_function
    )
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Protocol


class Precedence(IntEnum):
    """Determines which pattern wins when multiple match.

    Lower numbers = higher priority. Patterns are evaluated in precedence
    order, and the first matching pattern determines behavior.
    """

    CRITICAL = 10  # Q&A sequences, numbered lists — always merge
    HIGH = 20  # Bullet continuations with clear signals
    MEDIUM = 30  # Heading attachment, footnotes
    LOW = 40  # Heuristic-based merges (continuation words)
    BOUNDARY = 100  # Chapter headings — always split


class MergeBehavior(Enum):
    """What to do when a pattern matches."""

    MERGE = "merge"  # Always combine matching text
    ASK = "ask"  # Prompt user if interactive mode enabled
    SPLIT = "split"  # Always keep separate
    BOUNDARY = "boundary"  # Marks chunk boundary (never merge across)


@dataclass(frozen=True)
class Pattern:
    """A text pattern with merge behavior and precedence.

    Attributes:
        name: Unique identifier for this pattern (used in logging/learning)
        match: Regex that must match for pattern to apply
        precedence: Priority when multiple patterns match
        behavior: What to do when pattern matches
        description: Human-readable explanation (shown in interactive mode)
        continuation: Optional regex for detecting sequence continuation
                     (e.g., Q1: followed by Q2:)
    """

    name: str
    match: re.Pattern[str]
    precedence: Precedence
    behavior: MergeBehavior
    description: str
    continuation: re.Pattern[str] | None = None

    def matches(self, text: str) -> bool:
        """Return True if this pattern matches the text."""
        return bool(self.match.search(text))

    def is_sequence_continuation(self, prev: str, curr: str) -> bool:
        """Return True if curr continues a sequence started in prev.

        This is used for patterns like Q&A (Q1: → Q2:) or numbered lists
        (1. → 2.) where the continuation is a specific follow-on pattern.
        """
        if self.continuation is None:
            return False
        return (
            self.match.search(prev) is not None
            and self.continuation.match(curr.lstrip()) is not None
        )


@dataclass
class MergeDecision:
    """Result of evaluating whether to merge blocks.

    This is the return type of all merge evaluation functions, making
    decisions explicit and auditable.
    """

    should_merge: bool
    reason: str
    pattern: Pattern | None = None
    confidence: float = 1.0

    def __str__(self) -> str:
        pattern_name = self.pattern.name if self.pattern else "none"
        return f"MergeDecision({self.should_merge}, {self.reason}, pattern={pattern_name})"


class InteractiveCallback(Protocol):
    """Protocol for interactive merge confirmation.

    Implementations receive context about the merge decision and return
    the user's choice along with whether to remember it.
    """

    def __call__(
        self,
        prev_text: str,
        curr_text: str,
        pattern: Pattern,
        context: Mapping[str, Any],
    ) -> tuple[bool, str]:  # (should_merge, remember: "once"|"always"|"never")
        ...


# ---------------------------------------------------------------------------
# Default patterns (consolidated from across the codebase)
# ---------------------------------------------------------------------------

DEFAULT_PATTERNS: list[Pattern] = [
    # CRITICAL precedence — always merge these sequences
    Pattern(
        name="qa_sequence",
        match=re.compile(r"[QA]\d+:", re.IGNORECASE),
        precedence=Precedence.CRITICAL,
        behavior=MergeBehavior.MERGE,
        description="Q&A interview format (Q1:, A1:, Q2:, etc.)",
        continuation=re.compile(r"^[QA]\d+:", re.IGNORECASE),
    ),
    Pattern(
        name="numbered_list",
        match=re.compile(r"^\s*\d+[\.\)]\s"),
        precedence=Precedence.CRITICAL,
        behavior=MergeBehavior.MERGE,
        description="Numbered list items (1. Item, 2. Item)",
        continuation=re.compile(r"^\s*\d+[\.\)]\s"),
    ),
    Pattern(
        name="step_sequence",
        match=re.compile(r"^Step\s+\d+[:\.]", re.IGNORECASE),
        precedence=Precedence.CRITICAL,
        behavior=MergeBehavior.MERGE,
        description="Step-by-step instructions (Step 1:, Step 2:)",
        continuation=re.compile(r"^Step\s+\d+[:\.]", re.IGNORECASE),
    ),
    Pattern(
        name="lettered_list",
        match=re.compile(r"^\s*[a-zA-Z][\.\)]\s"),
        precedence=Precedence.CRITICAL,
        behavior=MergeBehavior.MERGE,
        description="Lettered list items (a. Item, b. Item)",
        continuation=re.compile(r"^\s*[a-zA-Z][\.\)]\s"),
    ),
    # HIGH precedence — merge with clear signals
    Pattern(
        name="bullet_list",
        match=re.compile(r"^\s*[•\-\*]\s"),
        precedence=Precedence.HIGH,
        behavior=MergeBehavior.MERGE,
        description="Bullet list items (• Item, - Item, * Item)",
        continuation=re.compile(r"^\s*[•\-\*]\s"),
    ),
    Pattern(
        name="dialogue_tag",
        match=re.compile(r"^[A-Z][a-z]+:\s"),
        precedence=Precedence.HIGH,
        behavior=MergeBehavior.ASK,  # Could be heading, ask user
        description="Dialogue or speaker tag (Alice:, Bob:)",
    ),
    # MEDIUM precedence — context-dependent
    Pattern(
        name="figure_reference",
        match=re.compile(r"^(?:Figure|Fig\.?|Table|Exhibit)\s+\d+", re.IGNORECASE),
        precedence=Precedence.MEDIUM,
        behavior=MergeBehavior.SPLIT,
        description="Figure or table reference (Figure 1:, Table 2)",
    ),
    Pattern(
        name="footnote_marker",
        match=re.compile(r"^[\[\(]?\d+[\]\)]?\s"),
        precedence=Precedence.MEDIUM,
        behavior=MergeBehavior.ASK,  # Could be list item or footnote
        description="Possible footnote marker ([1], (2), 3 )",
    ),
    # LOW precedence — heuristic-based
    Pattern(
        name="continuation_word",
        match=re.compile(
            r"^(?:And|But|So|However|Therefore|Yet|Still|Also|Meanwhile|"
            r"Additionally|Then|Thus|Instead|Nevertheless|Nonetheless|"
            r"Consequently|Moreover)\b",
            re.IGNORECASE,
        ),
        precedence=Precedence.LOW,
        behavior=MergeBehavior.MERGE,
        description="Sentence continuation word (And, But, However...)",
    ),
    # BOUNDARY precedence — always split here
    Pattern(
        name="chapter_heading",
        match=re.compile(r"^Chapter\s+\d+", re.IGNORECASE),
        precedence=Precedence.BOUNDARY,
        behavior=MergeBehavior.BOUNDARY,
        description="Chapter heading (Chapter 1, Chapter 2)",
    ),
    Pattern(
        name="part_marker",
        match=re.compile(r"^Part\s+(?:\d+|[IVX]+|One|Two|Three)", re.IGNORECASE),
        precedence=Precedence.BOUNDARY,
        behavior=MergeBehavior.BOUNDARY,
        description="Part marker (Part I, Part One)",
    ),
    Pattern(
        name="section_marker",
        match=re.compile(r"^(?:Section|§)\s*\d+", re.IGNORECASE),
        precedence=Precedence.BOUNDARY,
        behavior=MergeBehavior.SPLIT,
        description="Section marker (Section 1, §1)",
    ),
]


@dataclass
class PatternRegistry:
    """Centralized pattern detection with learning capability.

    Patterns are evaluated in precedence order. When interactive mode is
    enabled and a pattern with ASK behavior matches, the user is prompted.
    User decisions can be remembered for the session ("always"/"never").
    """

    patterns: list[Pattern] = field(default_factory=lambda: list(DEFAULT_PATTERNS))
    _learned: dict[str, MergeBehavior] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure patterns are sorted by precedence
        self.patterns = sorted(self.patterns, key=lambda p: p.precedence.value)

    def detect(self, text: str) -> list[Pattern]:
        """Return all patterns matching ``text``, sorted by precedence."""
        return [p for p in self.patterns if p.matches(text)]

    def should_merge(
        self,
        prev_text: str,
        curr_text: str,
        *,
        interactive_callback: InteractiveCallback | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> MergeDecision:
        """Determine if texts should merge based on pattern matches.

        Evaluation order:
        1. Check for BOUNDARY patterns in curr_text → SPLIT
        2. Check for sequence continuations (Q1→Q2, 1.→2.) → MERGE
        3. Check for ASK patterns with learned decisions → use learned
        4. Check for ASK patterns with callback → prompt user
        5. Check for MERGE/SPLIT patterns → use behavior
        6. Default → SPLIT (preserve boundaries)

        Returns:
            MergeDecision with should_merge, reason, and matched pattern
        """
        ctx = dict(context or {})
        curr_stripped = curr_text.lstrip()

        # 1. Check for boundary patterns first (always split)
        for pattern in self.patterns:
            if pattern.precedence == Precedence.BOUNDARY and pattern.matches(curr_stripped):
                return MergeDecision(
                    should_merge=False,
                    reason=f"boundary:{pattern.name}",
                    pattern=pattern,
                )

        # 2. Check for sequence continuations (highest priority for merging)
        for pattern in self.patterns:
            if pattern.is_sequence_continuation(prev_text, curr_text):
                # Check if user has learned a different behavior
                learned = self._learned.get(pattern.name)
                if learned == MergeBehavior.SPLIT:
                    return MergeDecision(
                        should_merge=False,
                        reason=f"learned_split:{pattern.name}",
                        pattern=pattern,
                    )
                return MergeDecision(
                    should_merge=True,
                    reason=f"sequence:{pattern.name}",
                    pattern=pattern,
                )

        # 3. Check patterns by precedence
        for pattern in self.patterns:
            if not pattern.matches(curr_stripped):
                continue

            # Check learned decisions
            learned = self._learned.get(pattern.name)
            if learned is not None:
                should = learned in (MergeBehavior.MERGE,)
                return MergeDecision(
                    should_merge=should,
                    reason=f"learned:{pattern.name}",
                    pattern=pattern,
                )

            if pattern.behavior == MergeBehavior.MERGE:
                return MergeDecision(
                    should_merge=True,
                    reason=f"pattern:{pattern.name}",
                    pattern=pattern,
                )

            if pattern.behavior == MergeBehavior.SPLIT:
                return MergeDecision(
                    should_merge=False,
                    reason=f"pattern:{pattern.name}",
                    pattern=pattern,
                )

            if pattern.behavior == MergeBehavior.ASK:
                if interactive_callback is not None:
                    decision, remember = interactive_callback(prev_text, curr_text, pattern, ctx)
                    if remember == "always":
                        self._learned[pattern.name] = (
                            MergeBehavior.MERGE if decision else MergeBehavior.SPLIT
                        )
                    elif remember == "never":
                        self._learned[pattern.name] = (
                            MergeBehavior.SPLIT if decision else MergeBehavior.MERGE
                        )
                    return MergeDecision(
                        should_merge=decision,
                        reason=f"interactive:{pattern.name}",
                        pattern=pattern,
                    )
                # No callback, use heuristic (don't merge ambiguous)
                return MergeDecision(
                    should_merge=False,
                    reason=f"ambiguous:{pattern.name}",
                    pattern=pattern,
                    confidence=0.5,
                )

        # 4. Default: don't merge (preserve block boundaries)
        return MergeDecision(
            should_merge=False,
            reason="no_pattern_match",
        )

    def learn(self, pattern_name: str, behavior: MergeBehavior) -> None:
        """Record a learned behavior for a pattern."""
        self._learned[pattern_name] = behavior

    def forget(self, pattern_name: str) -> None:
        """Remove a learned behavior."""
        self._learned.pop(pattern_name, None)

    def add_pattern(self, pattern: Pattern) -> None:
        """Add a custom pattern and re-sort by precedence."""
        self.patterns.append(pattern)
        self.patterns = sorted(self.patterns, key=lambda p: p.precedence.value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize learned behaviors for persistence."""
        return {"learned": {name: behavior.value for name, behavior in self._learned.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatternRegistry:
        """Load registry with previously learned behaviors."""
        registry = cls()
        for name, behavior_str in data.get("learned", {}).items():
            registry._learned[name] = MergeBehavior(behavior_str)
        return registry


# ---------------------------------------------------------------------------
# Convenience functions for backward compatibility
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY: PatternRegistry | None = None


def get_registry() -> PatternRegistry:
    """Return the default pattern registry (singleton)."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PatternRegistry()
    return _DEFAULT_REGISTRY


def is_qa_sequence_continuation(prev_text: str, curr_text: str) -> bool:
    """Check if curr_text continues a Q&A sequence from prev_text.

    This is a backward-compatible wrapper around the pattern registry.
    Prefer using PatternRegistry.should_merge() for new code.
    """
    registry = get_registry()
    qa_pattern = next((p for p in registry.patterns if p.name == "qa_sequence"), None)
    if qa_pattern is None:
        return False
    return qa_pattern.is_sequence_continuation(prev_text, curr_text)


def is_continuation_lead(text: str) -> bool:
    """Check if text starts with a continuation word.

    This is a backward-compatible wrapper around the pattern registry.
    Prefer using PatternRegistry.should_merge() for new code.
    """
    registry = get_registry()
    cont_pattern = next((p for p in registry.patterns if p.name == "continuation_word"), None)
    if cont_pattern is None:
        return False
    return cont_pattern.matches(text.lstrip())


# ---------------------------------------------------------------------------
# Confidence-based pattern evaluation (Phase 4)
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceDecision:
    """Result of confidence-based pattern evaluation.

    This extends MergeDecision with confidence scoring for interactive mode.
    When confidence is low enough, the system can prompt for user confirmation.
    """

    should_merge: bool
    confidence: float  # 0.0 to 1.0
    reason: str
    pattern: Pattern | None = None
    needs_confirmation: bool = False  # True when confidence is in "ask" range

    @property
    def is_certain(self) -> bool:
        """Return True if confidence is high enough to not need confirmation."""
        return self.confidence >= 0.85 or self.confidence <= 0.15


def qa_sequence_confidence(prev_text: str, curr_text: str) -> ConfidenceDecision:
    """Evaluate Q&A sequence continuation with confidence scoring.

    Returns a ConfidenceDecision that can be used for interactive prompts
    when confidence is in the uncertain range (0.3-0.7).

    High confidence (>=0.85): Definite Q&A sequence (Q1: followed by Q2:)
    Medium confidence (0.5-0.84): Possible Q&A sequence (partial pattern match)
    Low confidence (<0.3): Unlikely to be Q&A sequence
    """
    curr_stripped = curr_text.lstrip()

    # Check for explicit Q&A pattern match
    qa_pattern = re.compile(r"^[QA]\d+:", re.IGNORECASE)
    prev_has_qa = bool(re.search(r"[QA]\d+:", prev_text, re.IGNORECASE))
    curr_starts_qa = bool(qa_pattern.match(curr_stripped))

    if prev_has_qa and curr_starts_qa:
        # Both have Q&A patterns - high confidence
        return ConfidenceDecision(
            should_merge=True,
            confidence=0.95,
            reason="qa_sequence:both_have_qa_pattern",
            needs_confirmation=False,
        )

    if curr_starts_qa and not prev_has_qa:
        # Current starts Q&A but previous doesn't - might be new sequence
        return ConfidenceDecision(
            should_merge=False,
            confidence=0.7,
            reason="qa_sequence:new_qa_sequence_start",
            needs_confirmation=True,
        )

    if prev_has_qa and not curr_starts_qa:
        # Previous has Q&A but current doesn't - might be answer continuation
        # Check if current looks like answer text
        if curr_stripped and curr_stripped[0].isupper() and len(curr_stripped) > 20:
            return ConfidenceDecision(
                should_merge=True,
                confidence=0.6,
                reason="qa_sequence:possible_answer_continuation",
                needs_confirmation=True,
            )
        return ConfidenceDecision(
            should_merge=False,
            confidence=0.4,
            reason="qa_sequence:no_continuation_signal",
            needs_confirmation=True,
        )

    # No Q&A patterns
    return ConfidenceDecision(
        should_merge=False,
        confidence=0.1,
        reason="qa_sequence:no_qa_patterns",
        needs_confirmation=False,
    )


def colon_list_boundary_confidence(
    prev_text: str,
    curr_text: str,
) -> ConfidenceDecision:
    """Evaluate colon-list boundary with confidence scoring.

    Detects patterns like "Here are the items:" followed by "• First item"

    High confidence: Clear colon followed by clear bullet
    Medium confidence: Colon but ambiguous list start
    Low confidence: No colon-list pattern
    """
    prev_stripped = prev_text.rstrip()
    curr_stripped = curr_text.lstrip()

    # Check for colon at end of previous text
    if not prev_stripped.endswith(":"):
        return ConfidenceDecision(
            should_merge=False,
            confidence=0.1,
            reason="colon_list:no_colon_ending",
            needs_confirmation=False,
        )

    # Check for bullet at start of current text
    bullet_chars = frozenset("•●○◦▪▫‣⁃-–—")
    numbered_pattern = re.compile(r"^\d+[.)]\s")

    if curr_stripped and curr_stripped[0] in bullet_chars:
        return ConfidenceDecision(
            should_merge=True,
            confidence=0.95,
            reason="colon_list:clear_bullet_after_colon",
            needs_confirmation=False,
        )

    if numbered_pattern.match(curr_stripped):
        return ConfidenceDecision(
            should_merge=True,
            confidence=0.9,
            reason="colon_list:numbered_after_colon",
            needs_confirmation=False,
        )

    # Colon but no clear list start - might be subtitle or other structure
    if curr_stripped and curr_stripped[0].isupper():
        return ConfidenceDecision(
            should_merge=False,
            confidence=0.5,
            reason="colon_list:colon_with_capitalized_text",
            needs_confirmation=True,
        )

    return ConfidenceDecision(
        should_merge=True,
        confidence=0.6,
        reason="colon_list:colon_with_lowercase_continuation",
        needs_confirmation=True,
    )


def evaluate_merge_with_confidence(
    prev_text: str,
    curr_text: str,
    *,
    interactive_callback: InteractiveCallback | None = None,
    context: Mapping[str, Any] | None = None,
) -> MergeDecision:
    """Evaluate merge decision using confidence-based approach.

    This function combines pattern matching with confidence scoring,
    falling back to interactive prompts when confidence is uncertain.

    Args:
        prev_text: Text from previous block
        curr_text: Text from current block
        interactive_callback: Optional callback for uncertain cases
        context: Additional context for the callback

    Returns:
        MergeDecision with should_merge, reason, and confidence
    """
    # First check Q&A sequences - only return early if Q&A pattern was found
    qa_decision = qa_sequence_confidence(prev_text, curr_text)
    if qa_decision.confidence >= 0.5 and qa_decision.is_certain:
        return MergeDecision(
            should_merge=qa_decision.should_merge,
            reason=qa_decision.reason,
            confidence=qa_decision.confidence,
        )
    # If Q&A pattern detected but uncertain, continue to check other patterns

    # Check colon-list boundaries
    colon_decision = colon_list_boundary_confidence(prev_text, curr_text)
    if colon_decision.is_certain:
        return MergeDecision(
            should_merge=colon_decision.should_merge,
            reason=colon_decision.reason,
            confidence=colon_decision.confidence,
        )

    # Use standard pattern registry for other patterns
    registry = get_registry()
    pattern_decision = registry.should_merge(
        prev_text,
        curr_text,
        interactive_callback=interactive_callback,
        context=context,
    )

    return pattern_decision
