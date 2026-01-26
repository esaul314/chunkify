"""Persistence layer for learned interactive decisions.

This module handles saving and loading user decisions from interactive mode
so they can be automatically applied in future runs. Patterns are stored in
YAML format at ~/.config/pdf_chunker/learned_patterns.yaml by default.

Usage:
    # Load existing patterns
    learned = LearnedPatterns.load()

    # Add a new pattern
    learned.add(LearnedPattern(
        kind="footer",
        pattern="Collective Wisdom.*",
        decision="split",
        confidence=0.95,
    ))

    # Save patterns
    learned.save()

    # Check if a text matches a learned pattern
    match = learned.find_match("footer", "Collective Wisdom from the Experts 159")
    if match:
        print(f"Learned decision: {match.decision}")
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Optional YAML support - fall back to JSON if not available
try:
    import yaml

    _HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    _HAS_YAML = False

import json


def _default_config_path() -> Path:
    """Return the default path for learned patterns config."""
    return Path.home() / ".config" / "pdf_chunker" / "learned_patterns.yaml"


@dataclass
class LearnedPattern:
    """A single learned pattern from an interactive decision.

    Attributes:
        kind: Type of decision (footer, list_continuation, pattern_merge, heading_boundary)
        pattern: Regex pattern or text hash identifying the content
        decision: The user's decision ("merge", "split", "skip")
        confidence: Confidence level (0.0-1.0) - higher means stronger learning
        description: Optional human-readable description
        source_file: Optional source file where this was learned
        learned_at: ISO timestamp when pattern was learned
    """

    kind: str
    pattern: str
    decision: str
    confidence: float = 1.0
    description: str | None = None
    source_file: str | None = None
    learned_at: str | None = None

    def matches(self, text: str) -> bool:
        """Return True if this pattern matches the given text.

        The pattern is treated as a regex. If it fails to compile,
        it falls back to substring matching.
        """
        try:
            return bool(re.search(self.pattern, text, re.IGNORECASE))
        except re.error:
            # Fall back to substring match
            return self.pattern.lower() in text.lower()


@dataclass
class LearnedPatterns:
    """Collection of learned patterns with persistence.

    This class manages loading, saving, and querying learned patterns.
    Patterns are stored by kind (footer, list_continuation, etc.) for
    efficient lookup.
    """

    patterns: list[LearnedPattern] = field(default_factory=list)
    _by_kind: dict[str, list[LearnedPattern]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Build the kind index after initialization."""
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the by-kind index from patterns list."""
        self._by_kind = {}
        for pattern in self.patterns:
            self._by_kind.setdefault(pattern.kind, []).append(pattern)

    @classmethod
    def load(cls, path: Path | None = None) -> LearnedPatterns:
        """Load learned patterns from a config file.

        Args:
            path: Path to config file. Defaults to ~/.config/pdf_chunker/learned_patterns.yaml

        Returns:
            LearnedPatterns instance with loaded patterns, or empty if file doesn't exist.
        """
        path = path or _default_config_path()

        if not path.exists():
            return cls()

        content = path.read_text(encoding="utf-8")

        # Try YAML first, fall back to JSON
        if _HAS_YAML:
            data = yaml.safe_load(content)
        else:
            # Try JSON if YAML not available
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Invalid file, return empty
                return cls()

        if not isinstance(data, dict):
            return cls()

        patterns_data = data.get("patterns", [])
        patterns = [
            LearnedPattern(**p) for p in patterns_data if isinstance(p, dict) and "kind" in p
        ]

        return cls(patterns=patterns)

    def save(self, path: Path | None = None) -> None:
        """Save learned patterns to a config file.

        Args:
            path: Path to config file. Defaults to ~/.config/pdf_chunker/learned_patterns.yaml
        """
        path = path or _default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "patterns": [asdict(p) for p in self.patterns],
        }

        if _HAS_YAML:
            content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        else:
            content = json.dumps(data, indent=2)

        path.write_text(content, encoding="utf-8")

    def add(self, pattern: LearnedPattern) -> None:
        """Add a learned pattern.

        If a pattern with the same kind and pattern text already exists,
        it will be replaced.
        """
        # Remove any existing pattern with same kind and pattern
        self.patterns = [
            p
            for p in self.patterns
            if not (p.kind == pattern.kind and p.pattern == pattern.pattern)
        ]
        self.patterns.append(pattern)
        self._rebuild_index()

    def remove(self, kind: str, pattern: str) -> bool:
        """Remove a learned pattern.

        Args:
            kind: Pattern kind (footer, list_continuation, etc.)
            pattern: Pattern text to remove

        Returns:
            True if pattern was removed, False if not found.
        """
        original_len = len(self.patterns)
        self.patterns = [p for p in self.patterns if not (p.kind == kind and p.pattern == pattern)]
        removed = len(self.patterns) < original_len
        if removed:
            self._rebuild_index()
        return removed

    def find_match(self, kind: str, text: str) -> LearnedPattern | None:
        """Find a learned pattern that matches the given text.

        Args:
            kind: Pattern kind to search
            text: Text to match against patterns

        Returns:
            The first matching LearnedPattern, or None if no match.
        """
        for pattern in self._by_kind.get(kind, []):
            if pattern.matches(text):
                return pattern
        return None

    def find_all_matches(self, kind: str, text: str) -> list[LearnedPattern]:
        """Find all learned patterns that match the given text.

        Args:
            kind: Pattern kind to search
            text: Text to match against patterns

        Returns:
            List of matching LearnedPattern instances.
        """
        return [p for p in self._by_kind.get(kind, []) if p.matches(text)]

    def clear(self, kind: str | None = None) -> int:
        """Clear learned patterns.

        Args:
            kind: If provided, only clear patterns of this kind.
                 If None, clear all patterns.

        Returns:
            Number of patterns removed.
        """
        if kind is None:
            count = len(self.patterns)
            self.patterns = []
        else:
            original = len(self.patterns)
            self.patterns = [p for p in self.patterns if p.kind != kind]
            count = original - len(self.patterns)

        self._rebuild_index()
        return count

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (for embedding in other configs)."""
        return {
            "version": 1,
            "patterns": [asdict(p) for p in self.patterns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedPatterns:
        """Load from dictionary (for reading from other configs)."""
        patterns_data = data.get("patterns", [])
        patterns = [
            LearnedPattern(**p) for p in patterns_data if isinstance(p, dict) and "kind" in p
        ]
        return cls(patterns=patterns)


def text_to_pattern(text: str, *, use_hash: bool = False) -> str:
    """Convert text to a pattern string for learning.

    Args:
        text: Text to convert to pattern
        use_hash: If True, use MD5 hash. If False, escape regex chars.

    Returns:
        Pattern string suitable for storage.
    """
    if use_hash:
        return f"hash:{hashlib.md5(text.encode()).hexdigest()}"
    # Escape special regex characters and truncate
    escaped = re.escape(text.strip()[:100])
    return escaped


def extract_pattern_from_text(text: str) -> str:
    """Extract a reusable pattern from footer/header text.

    This tries to create a general pattern by:
    1. Removing page numbers at the end
    2. Keeping the title/header portion
    3. Making it case-insensitive

    Args:
        text: Footer or header text

    Returns:
        Generalized regex pattern.
    """
    stripped = text.strip()
    # Remove trailing page numbers
    pattern = re.sub(r"\s+\d{1,3}$", "", stripped)
    # Escape regex special chars but keep spaces flexible
    escaped = re.escape(pattern)
    # Make spaces match any whitespace
    flexible = escaped.replace(r"\ ", r"\s+")
    return flexible
