"""Tests for the transformation logging and pattern registry modules.

These tests verify Phase 0 and Phase 1 of the strategic refactoring plan:
- TransformationLog tracks text changes through the pipeline
- PatternRegistry provides centralized, declarative pattern matching
"""

from pdf_chunker.passes.transform_log import (
    TransformationLog,
    _short_hash,
    maybe_record,
)
from pdf_chunker.patterns import (
    MergeBehavior,
    PatternRegistry,
    Precedence,
    is_continuation_lead,
    is_qa_sequence_continuation,
)


class TestTransformationLog:
    """Tests for TransformationLog dataclass."""

    def test_create_from_text(self):
        log = TransformationLog.create("Hello world")
        assert log.fragment_id == _short_hash("Hello world")
        assert log.original_preview == "Hello world"
        assert log.entries == []

    def test_record_transformation(self):
        log = TransformationLog.create("Original text")
        log.record(
            "cleaned",
            "text_clean",
            "Normalized quotes",
            '"Original"',
            '"Original"',
        )
        assert len(log.entries) == 1
        entry = log.entries[0]
        assert entry.kind == "cleaned"
        assert entry.pass_name == "text_clean"
        assert entry.reason == "Normalized quotes"

    def test_debug_view(self):
        log = TransformationLog.create("Test text")
        log.record("extracted", "pdf_parse", "Initial", "src", "dst")
        log.record("merged", "split_semantic", "Q&A sequence", "a", "ab")

        view = log.debug_view()
        assert "Fragment" in view
        assert "[pdf_parse] extracted:" in view
        assert "[split_semantic] merged:" in view

    def test_serialization_roundtrip(self):
        log = TransformationLog.create("Serialize me")
        log.record("cleaned", "text_clean", "Test", "a", "b", details={"confidence": 0.9})

        data = log.to_dict()
        restored = TransformationLog.from_dict(data)

        assert restored.fragment_id == log.fragment_id
        assert len(restored.entries) == 1
        assert restored.entries[0].details == {"confidence": 0.9}

    def test_maybe_record_with_log(self):
        log = TransformationLog.create("Test")
        maybe_record(log, "cleaned", "test_pass", "reason", "src", "dst")
        assert len(log.entries) == 1

    def test_maybe_record_without_log(self):
        # Should not raise
        maybe_record(None, "cleaned", "test_pass", "reason", "src", "dst")


class TestPatternRegistry:
    """Tests for PatternRegistry and Pattern matching."""

    def test_qa_sequence_continuation(self):
        registry = PatternRegistry()

        # Q1 → Q2 should merge
        decision = registry.should_merge(
            "Q1: What is the first question?",
            "Q2: What about the second?",
        )
        assert decision.should_merge is True
        assert decision.pattern is not None
        assert decision.pattern.name == "qa_sequence"
        assert "sequence" in decision.reason

    def test_chapter_boundary_splits(self):
        registry = PatternRegistry()

        decision = registry.should_merge(
            "The end of the previous section.",
            "Chapter 5: A New Beginning",
        )
        assert decision.should_merge is False
        assert "boundary" in decision.reason

    def test_continuation_word_merges(self):
        registry = PatternRegistry()

        decision = registry.should_merge(
            "The project was successful.",
            "However, there were challenges.",
        )
        assert decision.should_merge is True
        assert decision.pattern is not None
        assert decision.pattern.name == "continuation_word"

    def test_numbered_list_continuation(self):
        registry = PatternRegistry()

        decision = registry.should_merge(
            "1. First item in the list",
            "2. Second item in the list",
        )
        assert decision.should_merge is True
        assert decision.pattern is not None
        assert decision.pattern.name == "numbered_list"

    def test_bullet_list_continuation(self):
        registry = PatternRegistry()

        decision = registry.should_merge(
            "• First bullet point",
            "• Second bullet point",
        )
        assert decision.should_merge is True
        assert decision.pattern is not None
        assert decision.pattern.name == "bullet_list"

    def test_no_pattern_defaults_to_split(self):
        registry = PatternRegistry()

        decision = registry.should_merge(
            "Completely unrelated sentence.",
            "Another unrelated sentence.",
        )
        assert decision.should_merge is False
        assert decision.reason == "no_pattern_match"

    def test_learned_behavior_override(self):
        registry = PatternRegistry()

        # Learn to always split on dialogue tags
        registry.learn("dialogue_tag", MergeBehavior.SPLIT)

        decision = registry.should_merge(
            "Previous dialogue.",
            "Alice: Said something",
        )
        assert decision.should_merge is False
        assert "learned" in decision.reason

    def test_serialization_preserves_learned(self):
        registry = PatternRegistry()
        registry.learn("dialogue_tag", MergeBehavior.MERGE)

        data = registry.to_dict()
        restored = PatternRegistry.from_dict(data)

        assert restored._learned.get("dialogue_tag") == MergeBehavior.MERGE


class TestBackwardCompatibility:
    """Tests for backward-compatible wrapper functions."""

    def test_is_qa_sequence_continuation(self):
        assert (
            is_qa_sequence_continuation(
                "Q1: First question?",
                "Q2: Second question?",
            )
            is True
        )

        assert (
            is_qa_sequence_continuation(
                "Regular text.",
                "More regular text.",
            )
            is False
        )

    def test_is_continuation_lead(self):
        assert is_continuation_lead("However, this continues.") is True
        assert is_continuation_lead("Therefore we conclude.") is True
        assert is_continuation_lead("The quick brown fox.") is False


class TestPatternPrecedence:
    """Tests for pattern precedence ordering."""

    def test_patterns_sorted_by_precedence(self):
        registry = PatternRegistry()
        precedences = [p.precedence.value for p in registry.patterns]
        assert precedences == sorted(precedences)

    def test_critical_patterns_first(self):
        registry = PatternRegistry()
        critical = [p for p in registry.patterns if p.precedence == Precedence.CRITICAL]
        assert len(critical) >= 3  # qa_sequence, numbered_list, step_sequence

    def test_boundary_patterns_last(self):
        registry = PatternRegistry()
        boundary = [p for p in registry.patterns if p.precedence == Precedence.BOUNDARY]
        assert len(boundary) >= 2  # chapter_heading, part_marker


class TestStitchBlockTransformLog:
    """Tests for TransformationLog integration with _stitch_block_continuations."""

    def test_qa_sequence_logged(self):
        """Q&A sequence merges are recorded in the transformation log."""
        from pdf_chunker.passes.split_semantic import _stitch_block_continuations
        from pdf_chunker.passes.transform_log import TransformationLog

        log = TransformationLog.create("Q&A test")
        records = [
            (1, {"type": "paragraph"}, "Q1: What is the capital?"),
            (1, {"type": "paragraph"}, "A1: Paris."),
            (1, {"type": "paragraph"}, "Q2: What is the population?"),
        ]
        result = _stitch_block_continuations(records, limit=None, transform_log=log)

        # Should have merged Q&A sequences
        assert len(result) < len(records)
        # Log should record the merge
        assert any(e.reason == "qa_sequence_merge" for e in log.entries)

    def test_heading_boundary_logged(self):
        """Heading boundaries are recorded when preserved."""
        from pdf_chunker.passes.split_semantic import _stitch_block_continuations
        from pdf_chunker.passes.transform_log import TransformationLog

        log = TransformationLog.create("Heading test")
        records = [
            (1, {"type": "paragraph"}, "Some text here."),
            (2, {"type": "heading"}, "Chapter 2"),
        ]
        result = _stitch_block_continuations(records, limit=None, transform_log=log)

        # Both records should remain separate
        assert len(result) == 2
        # Log should record the boundary decision
        assert any(e.reason == "heading_boundary_kept" for e in log.entries)

    def test_no_log_works(self):
        """Function works correctly when no log is provided."""
        from pdf_chunker.passes.split_semantic import _stitch_block_continuations

        records = [
            (1, {"type": "paragraph"}, "Q1: Question?"),
            (1, {"type": "paragraph"}, "A1: Answer."),
        ]
        # Should not raise
        result = _stitch_block_continuations(records, limit=None, transform_log=None)
        assert len(result) >= 1


class TestOverlapModule:
    """Tests for the overlap module extracted from split_semantic.py."""

    def test_split_words_basic(self):
        from pdf_chunker.passes.split_modules.overlap import split_words

        result = split_words("hello world foo")
        assert result == ("hello", "world", "foo")

    def test_split_words_empty(self):
        from pdf_chunker.passes.split_modules.overlap import split_words

        result = split_words("")
        assert result == ()

    def test_overlap_window_full_match(self):
        from pdf_chunker.passes.split_modules.overlap import overlap_window

        prev = ("a", "b", "c")
        curr = ("b", "c", "d")
        assert overlap_window(prev, curr, 3) == 2

    def test_overlap_window_no_match(self):
        from pdf_chunker.passes.split_modules.overlap import overlap_window

        prev = ("a", "b", "c")
        curr = ("x", "y", "z")
        assert overlap_window(prev, curr, 3) == 0

    def test_overlap_text(self):
        from pdf_chunker.passes.split_modules.overlap import overlap_text

        words = ("hello", "world", "foo")
        assert overlap_text(words, 2) == "hello world"

    def test_is_overlap_token_filters_empty(self):
        from pdf_chunker.passes.split_modules.overlap import is_overlap_token

        assert not is_overlap_token("")
        assert not is_overlap_token("   ")

    def test_is_overlap_token_preserves_bullets(self):
        from pdf_chunker.passes.split_modules.overlap import is_overlap_token

        assert is_overlap_token("•")
        assert is_overlap_token("-")
        assert is_overlap_token("—")

    def test_is_overlap_token_preserves_words(self):
        from pdf_chunker.passes.split_modules.overlap import is_overlap_token

        assert is_overlap_token("hello")
        assert is_overlap_token("123")

    def test_missing_overlap_prefix_calculates_missing(self):
        from pdf_chunker.passes.split_modules.overlap import missing_overlap_prefix

        prev = ("a", "b", "c", "d")
        curr = ("c", "d", "e")
        result = missing_overlap_prefix(prev, curr, overlap=4)
        assert result == ("a", "b")

    def test_missing_overlap_prefix_no_missing(self):
        from pdf_chunker.passes.split_modules.overlap import missing_overlap_prefix

        prev = ("a", "b", "c")
        curr = ("a", "b", "c", "d")
        result = missing_overlap_prefix(prev, curr, overlap=3)
        assert result == ()

    def test_prepend_words_basic(self):
        from pdf_chunker.passes.split_modules.overlap import prepend_words

        result = prepend_words(("hello", "world"), "foo bar")
        assert result == "hello world foo bar"

    def test_prepend_words_empty(self):
        from pdf_chunker.passes.split_modules.overlap import prepend_words

        result = prepend_words((), "foo bar")
        assert result == "foo bar"

    def test_restore_overlap_words_sequence(self):
        from pdf_chunker.passes.split_modules.overlap import restore_overlap_words

        chunks = ["First chunk ending with overlap.", "overlap. Second chunk here."]
        result = restore_overlap_words(chunks, overlap=2)
        assert len(result) == 2
        # Second chunk should have trimmed duplicate sentence prefix
        assert "overlap." not in result[1] or result[1] != chunks[1]

    def test_trim_sentence_prefix_removes_duplicate(self):
        from pdf_chunker.passes.split_modules.overlap import trim_sentence_prefix

        prev = "Some text. This is the last sentence."
        curr = "This is the last sentence. And this continues."
        result = trim_sentence_prefix(prev, curr)
        # The duplicate sentence should be trimmed
        assert result.startswith("And") or "continues" in result

    def test_trim_boundary_overlap(self):
        from pdf_chunker.passes.split_modules.overlap import trim_boundary_overlap

        prev = "sentence ending here."
        curr = "ending here. New content follows."
        result = trim_boundary_overlap(prev, curr, overlap=3)
        # Should trim the overlapping words
        assert "New content" in result

    def test_should_trim_overlap_sentence_end(self):
        from pdf_chunker.passes.split_modules.overlap import should_trim_overlap

        assert should_trim_overlap("end of sentence.")
        assert should_trim_overlap("question here?")
        assert not should_trim_overlap("no punct")
        assert not should_trim_overlap("")
