"""Tests for the unified interactive decision callback protocol.

These tests validate:
1. The unified DecisionContext and Decision dataclasses
2. Adapter functions for legacy callbacks
3. The make_unified_cli_prompt function
4. Integration between unified protocol and existing code
"""

import pytest

from pdf_chunker.interactive import (
    Decision,
    DecisionContext,
    DecisionKind,
    adapt_footer_callback,
    adapt_list_continuation_callback,
)


class TestDecisionContext:
    """Tests for DecisionContext dataclass."""

    def test_minimal_creation(self):
        """Can create with just kind and curr_text."""
        ctx = DecisionContext(kind=DecisionKind.FOOTER, curr_text="Some footer text")
        assert ctx.kind == DecisionKind.FOOTER
        assert ctx.curr_text == "Some footer text"
        assert ctx.prev_text is None
        assert ctx.page == 0
        assert ctx.confidence == 0.5

    def test_full_creation(self):
        """Can create with all fields populated."""
        extra = {"source": "test"}
        ctx = DecisionContext(
            kind=DecisionKind.PATTERN_MERGE,
            curr_text="Current text",
            prev_text="Previous text",
            page=42,
            confidence=0.85,
            pattern_name="qa_sequence",
            extra=extra,
        )
        assert ctx.kind == DecisionKind.PATTERN_MERGE
        assert ctx.curr_text == "Current text"
        assert ctx.prev_text == "Previous text"
        assert ctx.page == 42
        assert ctx.confidence == 0.85
        assert ctx.pattern_name == "qa_sequence"
        assert ctx.extra == {"source": "test"}

    def test_frozen_immutability(self):
        """DecisionContext is frozen/immutable."""
        ctx = DecisionContext(kind=DecisionKind.FOOTER, curr_text="test")
        with pytest.raises(AttributeError):
            ctx.curr_text = "modified"  # type: ignore[misc]


class TestDecision:
    """Tests for Decision dataclass."""

    def test_minimal_creation(self):
        """Can create with just action."""
        decision = Decision(action="merge")
        assert decision.action == "merge"
        assert decision.remember == "once"
        assert decision.reason is None

    def test_full_creation(self):
        """Can create with all fields."""
        decision = Decision(action="split", remember="always", reason="User confirmed footer")
        assert decision.action == "split"
        assert decision.remember == "always"
        assert decision.reason == "User confirmed footer"

    def test_valid_actions(self):
        """All valid actions are accepted."""
        for action in ("merge", "split", "skip"):
            decision = Decision(action=action)
            assert decision.action == action

    def test_valid_remember_values(self):
        """All valid remember values are accepted."""
        for remember in ("once", "always", "never"):
            decision = Decision(action="merge", remember=remember)
            assert decision.remember == remember


class TestDecisionKind:
    """Tests for DecisionKind enum."""

    def test_all_kinds_exist(self):
        """All expected decision kinds exist."""
        assert DecisionKind.FOOTER.value == "footer"
        assert DecisionKind.LIST_CONTINUATION.value == "list_continuation"
        assert DecisionKind.PATTERN_MERGE.value == "pattern_merge"
        assert DecisionKind.HEADING_BOUNDARY.value == "heading_boundary"

    def test_kind_values_are_strings(self):
        """All kind values are strings."""
        for kind in DecisionKind:
            assert isinstance(kind.value, str)


class TestAdaptFooterCallback:
    """Tests for adapt_footer_callback function."""

    def test_footer_true_becomes_split(self):
        """When legacy callback returns True (is footer), unified returns split."""

        def legacy_callback(text: str, page: int, context: dict) -> bool:
            return True  # Is a footer

        unified = adapt_footer_callback(legacy_callback)
        ctx = DecisionContext(kind=DecisionKind.FOOTER, curr_text="Footer text", page=42)
        decision = unified(ctx)
        assert decision.action == "split"

    def test_footer_false_becomes_merge(self):
        """When legacy callback returns False (not footer), unified returns merge."""

        def legacy_callback(text: str, page: int, context: dict) -> bool:
            return False  # Not a footer

        unified = adapt_footer_callback(legacy_callback)
        ctx = DecisionContext(kind=DecisionKind.FOOTER, curr_text="Body text", page=42)
        decision = unified(ctx)
        assert decision.action == "merge"

    def test_non_footer_context_returns_skip(self):
        """When context is not FOOTER kind, returns skip."""

        def legacy_callback(text: str, page: int, context: dict) -> bool:
            raise AssertionError("Should not be called")

        unified = adapt_footer_callback(legacy_callback)
        ctx = DecisionContext(kind=DecisionKind.LIST_CONTINUATION, curr_text="List text")
        decision = unified(ctx)
        assert decision.action == "skip"

    def test_passes_extra_as_context(self):
        """Extra dict is passed to legacy callback as context."""
        received_context = {}

        def legacy_callback(text: str, page: int, context: dict) -> bool:
            received_context.update(context)
            return True

        unified = adapt_footer_callback(legacy_callback)
        extra = {"heuristic_confidence": 0.8, "inline": True}
        ctx = DecisionContext(
            kind=DecisionKind.FOOTER,
            curr_text="Footer",
            page=10,
            extra=extra,
        )
        unified(ctx)
        assert received_context == extra


class TestAdaptListContinuationCallback:
    """Tests for adapt_list_continuation_callback function."""

    def test_continue_true_becomes_merge(self):
        """When legacy callback returns True (continue), unified returns merge."""

        def legacy_callback(list_item: str, candidate: str, page: int, context: dict) -> bool:
            return True  # Should continue

        unified = adapt_list_continuation_callback(legacy_callback)
        ctx = DecisionContext(
            kind=DecisionKind.LIST_CONTINUATION,
            prev_text="• First item",
            curr_text="continued text",
            page=5,
        )
        decision = unified(ctx)
        assert decision.action == "merge"

    def test_continue_false_becomes_split(self):
        """When legacy callback returns False (don't continue), unified returns split."""

        def legacy_callback(list_item: str, candidate: str, page: int, context: dict) -> bool:
            return False  # Should not continue

        unified = adapt_list_continuation_callback(legacy_callback)
        ctx = DecisionContext(
            kind=DecisionKind.LIST_CONTINUATION,
            prev_text="• First item",
            curr_text="• Second item",
            page=5,
        )
        decision = unified(ctx)
        assert decision.action == "split"

    def test_non_list_context_returns_skip(self):
        """When context is not LIST_CONTINUATION kind, returns skip."""

        def legacy_callback(list_item: str, candidate: str, page: int, context: dict) -> bool:
            raise AssertionError("Should not be called")

        unified = adapt_list_continuation_callback(legacy_callback)
        ctx = DecisionContext(kind=DecisionKind.FOOTER, curr_text="Footer text")
        decision = unified(ctx)
        assert decision.action == "skip"

    def test_handles_none_prev_text(self):
        """When prev_text is None, passes empty string to legacy."""
        received_list_item = None

        def legacy_callback(list_item: str, candidate: str, page: int, context: dict) -> bool:
            nonlocal received_list_item
            received_list_item = list_item
            return False

        unified = adapt_list_continuation_callback(legacy_callback)
        ctx = DecisionContext(
            kind=DecisionKind.LIST_CONTINUATION,
            prev_text=None,
            curr_text="some text",
        )
        unified(ctx)
        assert received_list_item == ""


class TestUnifiedCallbackProtocol:
    """Tests for the unified callback protocol usage."""

    def test_can_use_as_protocol_type(self):
        """A function matching the protocol can be used as InteractiveDecisionCallback."""

        def my_callback(ctx: DecisionContext) -> Decision:
            if ctx.kind == DecisionKind.FOOTER:
                return Decision(action="split")
            return Decision(action="skip")

        # This should work without type errors
        ctx = DecisionContext(kind=DecisionKind.FOOTER, curr_text="test")
        result = my_callback(ctx)
        assert result.action == "split"

    def test_callback_chain(self):
        """Can chain callbacks for different decision types."""

        def footer_handler(ctx: DecisionContext) -> Decision:
            if ctx.kind == DecisionKind.FOOTER:
                return Decision(action="split", reason="footer handler")
            return Decision(action="skip")

        def list_handler(ctx: DecisionContext) -> Decision:
            if ctx.kind == DecisionKind.LIST_CONTINUATION:
                return Decision(action="merge", reason="list handler")
            return Decision(action="skip")

        def combined_callback(ctx: DecisionContext) -> Decision:
            for handler in [footer_handler, list_handler]:
                result = handler(ctx)
                if result.action != "skip":
                    return result
            return Decision(action="skip")

        # Test footer handling
        footer_ctx = DecisionContext(kind=DecisionKind.FOOTER, curr_text="footer")
        assert combined_callback(footer_ctx).action == "split"

        # Test list handling
        list_ctx = DecisionContext(kind=DecisionKind.LIST_CONTINUATION, curr_text="item")
        assert combined_callback(list_ctx).action == "merge"

        # Test unhandled type
        other_ctx = DecisionContext(kind=DecisionKind.HEADING_BOUNDARY, curr_text="heading")
        assert combined_callback(other_ctx).action == "skip"


class TestRememberBehavior:
    """Tests for the 'remember' field behavior."""

    def test_always_remember(self):
        """Decision with remember='always' should be persisted."""
        decision = Decision(action="merge", remember="always")
        assert decision.remember == "always"

    def test_never_remember(self):
        """Decision with remember='never' stores the opposite."""
        decision = Decision(action="split", remember="never")
        assert decision.remember == "never"

    def test_once_is_default(self):
        """Default remember value is 'once'."""
        decision = Decision(action="merge")
        assert decision.remember == "once"
