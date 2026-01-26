"""Tests for the learned patterns persistence module.

These tests validate:
1. LearnedPattern dataclass creation and matching
2. LearnedPatterns collection operations (add, remove, find)
3. Persistence (save/load) with both YAML and JSON
4. Pattern extraction and matching utilities
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pdf_chunker.learned_patterns import (
    LearnedPattern,
    LearnedPatterns,
    extract_pattern_from_text,
    text_to_pattern,
)


class TestLearnedPattern:
    """Tests for LearnedPattern dataclass."""

    def test_minimal_creation(self):
        """Can create with required fields only."""
        pattern = LearnedPattern(
            kind="footer",
            pattern="Book Title.*",
            decision="split",
        )
        assert pattern.kind == "footer"
        assert pattern.pattern == "Book Title.*"
        assert pattern.decision == "split"
        assert pattern.confidence == 1.0

    def test_full_creation(self):
        """Can create with all fields."""
        pattern = LearnedPattern(
            kind="list_continuation",
            pattern="• First item.*",
            decision="merge",
            confidence=0.95,
            description="List items should merge",
            source_file="test.pdf",
            learned_at="2026-01-26T10:00:00",
        )
        assert pattern.kind == "list_continuation"
        assert pattern.confidence == 0.95
        assert pattern.description == "List items should merge"
        assert pattern.source_file == "test.pdf"

    def test_matches_regex_pattern(self):
        """matches() works with regex patterns."""
        pattern = LearnedPattern(
            kind="footer",
            pattern=r"Chapter \d+",
            decision="split",
        )
        assert pattern.matches("Chapter 5")
        assert pattern.matches("See Chapter 42 for details")
        assert not pattern.matches("No chapter here")

    def test_matches_literal_pattern(self):
        """matches() works with literal text."""
        pattern = LearnedPattern(
            kind="footer",
            pattern="Collective Wisdom",
            decision="split",
        )
        assert pattern.matches("Collective Wisdom from the Experts 159")
        assert pattern.matches("See Collective Wisdom")
        assert not pattern.matches("Other book title")

    def test_matches_invalid_regex_falls_back(self):
        """Invalid regex falls back to substring match."""
        pattern = LearnedPattern(
            kind="footer",
            pattern="[invalid regex(",
            decision="split",
        )
        # Should not raise, falls back to substring
        assert pattern.matches("This [invalid regex( text")
        assert not pattern.matches("No match here")

    def test_matches_case_insensitive(self):
        """matches() is case-insensitive."""
        pattern = LearnedPattern(
            kind="footer",
            pattern="CHAPTER",
            decision="split",
        )
        assert pattern.matches("chapter 1")
        assert pattern.matches("Chapter 1")
        assert pattern.matches("CHAPTER 1")


class TestLearnedPatterns:
    """Tests for LearnedPatterns collection."""

    def test_empty_creation(self):
        """Can create empty collection."""
        learned = LearnedPatterns()
        assert learned.patterns == []

    def test_creation_with_patterns(self):
        """Can create with initial patterns."""
        patterns = [
            LearnedPattern(kind="footer", pattern="A", decision="split"),
            LearnedPattern(kind="footer", pattern="B", decision="split"),
        ]
        learned = LearnedPatterns(patterns=patterns)
        assert len(learned.patterns) == 2

    def test_add_pattern(self):
        """add() adds pattern to collection."""
        learned = LearnedPatterns()
        pattern = LearnedPattern(kind="footer", pattern="Test", decision="split")
        learned.add(pattern)
        assert len(learned.patterns) == 1
        assert learned.patterns[0].pattern == "Test"

    def test_add_replaces_duplicate(self):
        """add() replaces pattern with same kind and pattern text."""
        learned = LearnedPatterns()
        p1 = LearnedPattern(kind="footer", pattern="Test", decision="split", confidence=0.8)
        p2 = LearnedPattern(kind="footer", pattern="Test", decision="merge", confidence=0.9)

        learned.add(p1)
        learned.add(p2)

        assert len(learned.patterns) == 1
        assert learned.patterns[0].decision == "merge"
        assert learned.patterns[0].confidence == 0.9

    def test_remove_pattern(self):
        """remove() removes pattern by kind and pattern text."""
        learned = LearnedPatterns()
        pattern = LearnedPattern(kind="footer", pattern="Test", decision="split")
        learned.add(pattern)

        result = learned.remove("footer", "Test")
        assert result is True
        assert len(learned.patterns) == 0

    def test_remove_nonexistent_returns_false(self):
        """remove() returns False if pattern not found."""
        learned = LearnedPatterns()
        result = learned.remove("footer", "Nonexistent")
        assert result is False

    def test_find_match_returns_matching_pattern(self):
        """find_match() returns first matching pattern."""
        learned = LearnedPatterns()
        learned.add(LearnedPattern(kind="footer", pattern="Chapter.*", decision="split"))
        learned.add(LearnedPattern(kind="footer", pattern="Appendix.*", decision="split"))

        match = learned.find_match("footer", "Chapter 5: Introduction")
        assert match is not None
        assert "Chapter" in match.pattern

    def test_find_match_returns_none_for_no_match(self):
        """find_match() returns None if no pattern matches."""
        learned = LearnedPatterns()
        learned.add(LearnedPattern(kind="footer", pattern="Chapter.*", decision="split"))

        match = learned.find_match("footer", "Some other text")
        assert match is None

    def test_find_match_filters_by_kind(self):
        """find_match() only searches patterns of the specified kind."""
        learned = LearnedPatterns()
        learned.add(LearnedPattern(kind="footer", pattern="Test", decision="split"))
        learned.add(LearnedPattern(kind="list_continuation", pattern="Test", decision="merge"))

        footer_match = learned.find_match("footer", "Test text")
        list_match = learned.find_match("list_continuation", "Test text")

        assert footer_match is not None
        assert footer_match.decision == "split"
        assert list_match is not None
        assert list_match.decision == "merge"

    def test_find_all_matches(self):
        """find_all_matches() returns all matching patterns."""
        learned = LearnedPatterns()
        learned.add(LearnedPattern(kind="footer", pattern="Test", decision="split"))
        learned.add(LearnedPattern(kind="footer", pattern="Te", decision="merge"))

        matches = learned.find_all_matches("footer", "Test text")
        assert len(matches) == 2

    def test_clear_all(self):
        """clear() without kind removes all patterns."""
        learned = LearnedPatterns()
        learned.add(LearnedPattern(kind="footer", pattern="A", decision="split"))
        learned.add(LearnedPattern(kind="list", pattern="B", decision="merge"))

        count = learned.clear()
        assert count == 2
        assert len(learned.patterns) == 0

    def test_clear_by_kind(self):
        """clear() with kind removes only that kind."""
        learned = LearnedPatterns()
        learned.add(LearnedPattern(kind="footer", pattern="A", decision="split"))
        learned.add(LearnedPattern(kind="footer", pattern="B", decision="split"))
        learned.add(LearnedPattern(kind="list", pattern="C", decision="merge"))

        count = learned.clear("footer")
        assert count == 2
        assert len(learned.patterns) == 1
        assert learned.patterns[0].kind == "list"


class TestLearnedPatternsPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load_yaml(self):
        """Can save and load patterns in YAML format."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "learned.yaml"

            # Save
            learned = LearnedPatterns()
            learned.add(
                LearnedPattern(
                    kind="footer",
                    pattern="Chapter.*",
                    decision="split",
                    confidence=0.95,
                    description="Chapter headers",
                )
            )
            learned.save(path)

            # Verify file exists
            assert path.exists()

            # Load
            loaded = LearnedPatterns.load(path)
            assert len(loaded.patterns) == 1
            assert loaded.patterns[0].kind == "footer"
            assert loaded.patterns[0].pattern == "Chapter.*"
            assert loaded.patterns[0].confidence == 0.95

    def test_save_and_load_json_fallback(self):
        """Can save and load as JSON when YAML unavailable."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "learned.json"

            # Save using to_dict + json
            learned = LearnedPatterns()
            learned.add(LearnedPattern(kind="footer", pattern="Test", decision="split"))
            data = learned.to_dict()
            path.write_text(json.dumps(data))

            # Load
            loaded = LearnedPatterns.load(path)
            assert len(loaded.patterns) == 1

    def test_load_nonexistent_returns_empty(self):
        """Loading nonexistent file returns empty collection."""
        learned = LearnedPatterns.load(Path("/nonexistent/path/file.yaml"))
        assert len(learned.patterns) == 0

    def test_save_creates_parent_directories(self):
        """save() creates parent directories if needed."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "learned.yaml"

            learned = LearnedPatterns()
            learned.add(LearnedPattern(kind="footer", pattern="Test", decision="split"))
            learned.save(path)

            assert path.exists()
            assert path.parent.exists()

    def test_to_dict_and_from_dict(self):
        """Can round-trip through dict serialization."""
        original = LearnedPatterns()
        original.add(LearnedPattern(kind="footer", pattern="A", decision="split"))
        original.add(LearnedPattern(kind="list", pattern="B", decision="merge"))

        data = original.to_dict()
        restored = LearnedPatterns.from_dict(data)

        assert len(restored.patterns) == 2
        assert restored.patterns[0].kind == original.patterns[0].kind
        assert restored.patterns[1].pattern == original.patterns[1].pattern


class TestTextToPattern:
    """Tests for text_to_pattern utility."""

    def test_escapes_regex_chars(self):
        """Escapes regex special characters."""
        pattern = text_to_pattern("Chapter [1]")
        assert pattern == r"Chapter\ \[1\]"

    def test_truncates_long_text(self):
        """Truncates very long text."""
        long_text = "A" * 200
        pattern = text_to_pattern(long_text)
        assert len(pattern) <= 100

    def test_hash_mode(self):
        """Can use hash mode for text matching."""
        pattern = text_to_pattern("Some text", use_hash=True)
        assert pattern.startswith("hash:")

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace."""
        pattern = text_to_pattern("  spaced text  ")
        assert not pattern.startswith(" ")


class TestExtractPatternFromText:
    """Tests for extract_pattern_from_text utility."""

    def test_removes_trailing_page_number(self):
        """Removes trailing page numbers from text."""
        pattern = extract_pattern_from_text("Collective Wisdom 159")
        assert "159" not in pattern
        assert "Collective" in pattern

    def test_makes_spaces_flexible(self):
        """Converts spaces to flexible whitespace match."""
        pattern = extract_pattern_from_text("Book Title")
        assert r"\s+" in pattern

    def test_handles_no_page_number(self):
        """Works with text that has no page number."""
        pattern = extract_pattern_from_text("Just a title")
        assert "title" in pattern.lower() or "Just" in pattern
