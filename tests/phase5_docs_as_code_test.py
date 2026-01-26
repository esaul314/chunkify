"""Tests for Phase 5: Documentation as Code.

These tests verify that:
1. The doc generation script produces valid output
2. MergeDecision is used consistently across merge points
3. Pattern registry documentation matches actual behavior
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pdf_chunker.patterns import (
    DEFAULT_PATTERNS,
    MergeBehavior,
    MergeDecision,
    PatternRegistry,
    Precedence,
)


class TestDocGeneration:
    """Test the generate_merge_docs.py script."""

    def test_script_runs_successfully(self, tmp_path: Path) -> None:
        """Script should generate valid markdown output."""
        output = tmp_path / "MERGE_DECISIONS.md"
        result = subprocess.run(
            [sys.executable, "scripts/generate_merge_docs.py", "-o", str(output)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert output.exists()
        content = output.read_text()
        assert "# Merge Decision Reference" in content
        assert "Quick Reference" in content

    def test_script_includes_all_patterns(self, tmp_path: Path) -> None:
        """Generated docs should include all DEFAULT_PATTERNS."""
        output = tmp_path / "MERGE_DECISIONS.md"
        subprocess.run(
            [sys.executable, "scripts/generate_merge_docs.py", "-o", str(output)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
        )
        content = output.read_text()
        for pattern in DEFAULT_PATTERNS:
            assert pattern.name in content, f"Missing pattern: {pattern.name}"

    def test_check_mode_detects_changes(self, tmp_path: Path) -> None:
        """--check should fail if docs don't exist."""
        output = tmp_path / "MERGE_DECISIONS.md"
        result = subprocess.run(
            [sys.executable, "scripts/generate_merge_docs.py", "--check", "-o", str(output)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "does not exist" in result.stderr


class TestMergeDecision:
    """Test MergeDecision usage in pattern registry."""

    def test_merge_decision_has_required_fields(self) -> None:
        """MergeDecision should have should_merge, reason, pattern, confidence."""
        decision = MergeDecision(
            should_merge=True,
            reason="test_reason",
            pattern=DEFAULT_PATTERNS[0],
            confidence=0.95,
        )
        assert decision.should_merge is True
        assert decision.reason == "test_reason"
        assert decision.pattern is not None
        assert decision.confidence == 0.95

    def test_registry_returns_merge_decision(self) -> None:
        """PatternRegistry.should_merge should return MergeDecision."""
        registry = PatternRegistry()
        decision = registry.should_merge("Q1: First question", "Q2: Second question")
        assert isinstance(decision, MergeDecision)
        assert decision.should_merge is True
        assert "sequence" in decision.reason

    def test_boundary_pattern_returns_split(self) -> None:
        """BOUNDARY patterns should return should_merge=False."""
        registry = PatternRegistry()
        decision = registry.should_merge("Some text", "Chapter 1: Introduction")
        assert isinstance(decision, MergeDecision)
        assert decision.should_merge is False
        assert "boundary" in decision.reason

    def test_no_match_returns_decision_with_reason(self) -> None:
        """Even no-match cases should return MergeDecision with reason."""
        registry = PatternRegistry()
        decision = registry.should_merge("Hello world.", "Goodbye moon.")
        assert isinstance(decision, MergeDecision)
        assert "no_pattern_match" in decision.reason


class TestPrecedenceDocumentation:
    """Test that precedence is properly documented and ordered."""

    def test_patterns_sorted_by_precedence(self) -> None:
        """Registry should maintain patterns sorted by precedence."""
        registry = PatternRegistry()
        precedences = [p.precedence.value for p in registry.patterns]
        assert precedences == sorted(precedences), "Patterns not sorted by precedence"

    def test_critical_before_boundary(self) -> None:
        """CRITICAL (10) should come before BOUNDARY (100)."""
        assert Precedence.CRITICAL.value < Precedence.BOUNDARY.value

    def test_all_precedence_levels_documented(self) -> None:
        """Each Precedence level should have at least one pattern."""
        used_precedences = {p.precedence for p in DEFAULT_PATTERNS}
        # At minimum, we should have CRITICAL, HIGH, and BOUNDARY
        assert Precedence.CRITICAL in used_precedences
        assert Precedence.HIGH in used_precedences
        assert Precedence.BOUNDARY in used_precedences

    def test_all_behaviors_documented(self) -> None:
        """Each MergeBehavior should be used by at least one pattern."""
        used_behaviors = {p.behavior for p in DEFAULT_PATTERNS}
        assert MergeBehavior.MERGE in used_behaviors
        assert MergeBehavior.ASK in used_behaviors
        assert MergeBehavior.BOUNDARY in used_behaviors


class TestStitchingDocumentation:
    """Test that stitching module has proper documentation."""

    def test_stitch_function_has_precedence_docstring(self) -> None:
        """stitch_block_continuations should document precedence order."""
        from pdf_chunker.passes.split_modules.stitching import stitch_block_continuations

        docstring = stitch_block_continuations.__doc__
        assert docstring is not None
        assert "Precedence Order" in docstring
        assert "Q&A sequence" in docstring
        assert "CRITICAL" in docstring or "highest priority" in docstring.lower()
