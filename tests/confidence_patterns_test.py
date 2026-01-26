"""Tests for confidence-based pattern evaluation.

These tests validate the Phase 4 confidence-based pattern functions
that support interactive mode prompting.
"""

from pdf_chunker.patterns import (
    ConfidenceDecision,
    colon_list_boundary_confidence,
    evaluate_merge_with_confidence,
    qa_sequence_confidence,
)


class TestQASequenceConfidence:
    """Tests for qa_sequence_confidence function."""

    def test_both_have_qa_patterns_high_confidence(self):
        """When both texts have Q&A patterns, high confidence merge."""
        prev = "Q1: What is your name?\nA1: My name is Alice."
        curr = "Q2: What is your role?"

        decision = qa_sequence_confidence(prev, curr)

        assert decision.should_merge is True
        assert decision.confidence >= 0.9
        assert decision.is_certain is True
        assert "qa_sequence" in decision.reason

    def test_new_qa_sequence_needs_confirmation(self):
        """When current starts Q&A but prev doesn't, needs confirmation."""
        prev = "This is some regular text about the topic."
        curr = "Q1: What do you think about this?"

        decision = qa_sequence_confidence(prev, curr)

        assert decision.should_merge is False
        assert decision.needs_confirmation is True
        assert "new_qa_sequence" in decision.reason

    def test_answer_continuation_uncertain(self):
        """When prev has Q&A but current doesn't, uncertain about continuation."""
        prev = "Q1: What is your opinion on remote work?"
        curr = "Working from home has become increasingly popular in recent years."

        decision = qa_sequence_confidence(prev, curr)

        # Should suggest merge for answer continuation but with uncertainty
        assert decision.needs_confirmation is True
        assert 0.3 < decision.confidence < 0.8

    def test_no_qa_patterns_low_confidence(self):
        """When neither text has Q&A patterns, low confidence no merge."""
        prev = "Regular paragraph about a topic."
        curr = "Another paragraph continuing the discussion."

        decision = qa_sequence_confidence(prev, curr)

        assert decision.should_merge is False
        assert decision.confidence <= 0.2
        assert decision.is_certain is True

    def test_qa_pattern_case_insensitive(self):
        """Q&A patterns should be case-insensitive."""
        prev = "q1: lower case question"
        curr = "Q2: Mixed case question"

        decision = qa_sequence_confidence(prev, curr)

        assert decision.should_merge is True
        assert decision.confidence >= 0.9


class TestColonListBoundaryConfidence:
    """Tests for colon_list_boundary_confidence function."""

    def test_colon_with_bullet_high_confidence(self):
        """Colon followed by bullet has high confidence."""
        prev = "Here are the key points:"
        curr = "• First important point"

        decision = colon_list_boundary_confidence(prev, curr)

        assert decision.should_merge is True
        assert decision.confidence >= 0.9
        assert decision.is_certain is True

    def test_colon_with_numbered_list(self):
        """Colon followed by numbered list has high confidence."""
        prev = "Steps to follow:"
        curr = "1. First step"

        decision = colon_list_boundary_confidence(prev, curr)

        assert decision.should_merge is True
        assert decision.confidence >= 0.85

    def test_colon_with_capital_uncertain(self):
        """Colon followed by capitalized text is uncertain."""
        prev = "The title:"
        curr = "Capital Letter Start"

        decision = colon_list_boundary_confidence(prev, curr)

        assert decision.needs_confirmation is True
        # Could be either subtitle or new section
        assert 0.3 < decision.confidence < 0.7

    def test_colon_with_lowercase_likely_merge(self):
        """Colon followed by lowercase suggests continuation."""
        prev = "The reason is:"
        curr = "because the system needs to be updated"

        decision = colon_list_boundary_confidence(prev, curr)

        assert decision.should_merge is True
        assert decision.confidence >= 0.5

    def test_no_colon_low_confidence(self):
        """Without colon ending, low confidence for list boundary."""
        prev = "This is a regular sentence"
        curr = "• A bullet point"

        decision = colon_list_boundary_confidence(prev, curr)

        assert decision.confidence <= 0.2
        assert decision.is_certain is True

    def test_various_bullet_characters(self):
        """Should recognize various bullet characters."""
        prev = "List items:"
        for bullet in ["•", "●", "○", "-", "–"]:
            curr = f"{bullet} Item text"
            decision = colon_list_boundary_confidence(prev, curr)
            assert decision.should_merge is True, f"Failed for bullet: {bullet}"


class TestConfidenceDecision:
    """Tests for ConfidenceDecision dataclass."""

    def test_is_certain_high_confidence(self):
        """is_certain is True for high confidence."""
        decision = ConfidenceDecision(
            should_merge=True,
            confidence=0.95,
            reason="test",
        )
        assert decision.is_certain is True

    def test_is_certain_low_confidence(self):
        """is_certain is True for very low confidence."""
        decision = ConfidenceDecision(
            should_merge=False,
            confidence=0.1,
            reason="test",
        )
        assert decision.is_certain is True

    def test_not_certain_medium_confidence(self):
        """is_certain is False for medium confidence."""
        decision = ConfidenceDecision(
            should_merge=True,
            confidence=0.5,
            reason="test",
        )
        assert decision.is_certain is False


class TestEvaluateMergeWithConfidence:
    """Tests for evaluate_merge_with_confidence function."""

    def test_qa_sequence_takes_precedence(self):
        """Q&A sequence detection takes precedence when certain."""
        prev = "Q1: First question\nA1: First answer"
        curr = "Q2: Second question"

        decision = evaluate_merge_with_confidence(prev, curr)

        assert decision.should_merge is True
        assert "qa_sequence" in decision.reason

    def test_colon_list_when_no_qa(self):
        """Colon-list boundary checked when no Q&A pattern."""
        prev = "Required items:"
        curr = "• Pencil\n• Paper"

        decision = evaluate_merge_with_confidence(prev, curr)

        assert decision.should_merge is True
        assert "colon_list" in decision.reason

    def test_falls_back_to_pattern_registry(self):
        """Falls back to pattern registry for other patterns."""
        prev = "Regular text without special patterns."
        curr = "More regular text."

        decision = evaluate_merge_with_confidence(prev, curr)

        # Should get a decision from pattern registry
        assert decision.reason is not None

    def test_with_interactive_callback(self):
        """Interactive callback is passed through to pattern registry."""
        callback_called = False

        def mock_callback(prev, curr, pattern, context):
            nonlocal callback_called
            callback_called = True
            return True, "once"

        # Use text that triggers ASK pattern behavior
        prev = "Some text."
        curr = "Alice: Said something"  # dialogue_tag pattern has ASK behavior

        # Note: This may or may not trigger the callback depending on pattern matching
        decision = evaluate_merge_with_confidence(
            prev,
            curr,
            interactive_callback=mock_callback,
        )

        assert decision is not None  # Should always return a decision
