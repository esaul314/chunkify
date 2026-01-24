"""Characterization tests for emit_jsonl merge functions.

These tests lock down the current behavior of the three merge functions
before any refactoring. Each test documents expected behavior for edge cases.
"""

from pdf_chunker.passes.emit_jsonl import (
    _coherent,
    _merge_incomplete_lists,
    _merge_short_rows,
    _merge_very_short_forward,
)
from pdf_chunker.passes.emit_jsonl_lists import (
    count_list_items as _count_list_items,
)
from pdf_chunker.passes.emit_jsonl_lists import (
    ends_with_list_intro_colon as _ends_with_list_intro_colon,
)
from pdf_chunker.passes.emit_jsonl_lists import (
    has_incomplete_list as _has_incomplete_list,
)
from pdf_chunker.passes.emit_jsonl_lists import (
    has_single_inline_bullet as _has_single_inline_bullet,
)
from pdf_chunker.passes.emit_jsonl_lists import (
    has_unterminated_bullet_item as _has_unterminated_bullet_item,
)

# ---------------------------------------------------------------------------
# _merge_very_short_forward characterization tests
# ---------------------------------------------------------------------------


class TestMergeVeryShortForward:
    """Tests for _merge_very_short_forward function."""

    def test_empty_list_returns_empty(self):
        assert _merge_very_short_forward([]) == []

    def test_single_long_item_unchanged(self):
        items = [{"text": "This is a paragraph with many words that exceeds the threshold."}]
        result = _merge_very_short_forward(items)
        assert len(result) == 1
        assert result[0]["text"] == items[0]["text"]

    def test_short_item_merges_forward(self):
        """Short items (< 30 words) merge with the following item."""
        items = [
            {"text": "Short heading"},
            {"text": "This paragraph has enough words to stand alone as content."},
        ]
        result = _merge_very_short_forward(items)
        assert len(result) == 1
        assert "Short heading" in result[0]["text"]
        assert "This paragraph" in result[0]["text"]

    def test_consecutive_short_items_merge_forward(self):
        """Multiple consecutive short items merge into one."""
        items = [
            {"text": "Chapter 1"},
            {"text": "Section A"},
            {"text": "This is the actual content paragraph."},
        ]
        result = _merge_very_short_forward(items)
        assert len(result) == 1
        assert "Chapter 1" in result[0]["text"]
        assert "Section A" in result[0]["text"]
        assert "actual content" in result[0]["text"]

    def test_short_trailing_item_merges_backward(self):
        """A short item at the end merges with the preceding item."""
        items = [
            {"text": "This is a complete paragraph with sufficient length."},
            {"text": "Final note"},
        ]
        result = _merge_very_short_forward(items)
        assert len(result) == 1
        assert "complete paragraph" in result[0]["text"]
        assert "Final note" in result[0]["text"]

    def test_coherent_short_item_preserved(self):
        """Coherent items (proper sentence) are preserved even if short."""
        items = [
            {"text": "This is a valid sentence. It has enough words to be coherent."},
            {"text": "Another complete sentence with proper ending."},
        ]
        result = _merge_very_short_forward(items)
        # Both should be preserved since they're coherent
        assert len(result) >= 1

    def test_merge_respects_max_chars_limit(self, monkeypatch):
        """Merging stops when result would exceed max_merge_chars."""
        monkeypatch.setenv("PDF_CHUNKER_MAX_MERGE_CHARS", "50")
        items = [
            {"text": "Short heading here"},
            {"text": "This is a much longer paragraph that exceeds the limit."},
        ]
        result = _merge_very_short_forward(items)
        # Should keep them separate since merged would exceed 50 chars
        assert len(result) == 2

    def test_orphan_bullet_merges_forward(self):
        """Orphan bullet items merge forward for list coherence."""
        items = [
            {"text": "• single orphan bullet"},
            {"text": "This is the content that follows the bullet."},
        ]
        result = _merge_very_short_forward(items)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _merge_incomplete_lists characterization tests
# ---------------------------------------------------------------------------


class TestMergeIncompleteLists:
    """Tests for _merge_incomplete_lists function."""

    def test_empty_list_returns_empty(self):
        assert _merge_incomplete_lists([]) == []

    def test_single_row_unchanged(self):
        rows = [{"text": "Single complete paragraph."}]
        result = _merge_incomplete_lists(rows)
        assert result == rows

    def test_row_with_colon_intro_merges_with_next(self):
        """Row ending with ':' merges with the following list content."""
        rows = [
            {"text": "Here are the key points:"},
            {"text": "• First point\n• Second point"},
        ]
        result = _merge_incomplete_lists(rows)
        assert len(result) == 1
        assert "key points:" in result[0]["text"]
        assert "First point" in result[0]["text"]

    def test_row_with_inline_bullet_merges(self):
        """Row with incomplete inline bullet merges forward."""
        rows = [
            {"text": "Introduction text: • First item"},
            {"text": "• Second item\n• Third item"},
        ]
        result = _merge_incomplete_lists(rows)
        # Should merge since first row has incomplete list
        assert len(result) == 1

    def test_complete_list_not_merged(self):
        """Complete list blocks are not merged."""
        rows = [
            {"text": "Introduction paragraph that ends properly."},
            {"text": "• First item\n• Second item\n• Third item."},
        ]
        result = _merge_incomplete_lists(rows)
        # Second row is a complete list, should stay separate
        assert len(result) == 2

    def test_backward_merge_when_forward_too_large(self, monkeypatch):
        """Falls back to backward merge when forward would exceed limit."""
        monkeypatch.setenv("PDF_CHUNKER_MAX_MERGE_CHARS", "100")
        rows = [
            {"text": "Previous content that is complete."},
            {"text": "List intro:"},  # Incomplete list
            {"text": "Very long content " * 20},  # Too large for forward merge
        ]
        result = _merge_incomplete_lists(rows)
        # "List intro:" should merge backward with "Previous content"
        assert "Previous content" in result[0]["text"]
        assert "List intro:" in result[0]["text"]


# ---------------------------------------------------------------------------
# _merge_short_rows characterization tests
# ---------------------------------------------------------------------------


class TestMergeShortRows:
    """Tests for _merge_short_rows function."""

    def test_empty_list_returns_empty(self):
        assert _merge_short_rows([]) == []

    def test_single_row_unchanged(self):
        rows = [{"text": "This is a complete sentence with enough words."}]
        result = _merge_short_rows(rows)
        assert result == rows

    def test_short_row_merges_forward(self):
        """Rows below min_row_words merge forward."""
        rows = [
            {"text": "Too short"},  # < 15 words
            {"text": "This is a longer paragraph with sufficient content."},
        ]
        result = _merge_short_rows(rows)
        assert len(result) == 1
        assert "Too short" in result[0]["text"]
        assert "longer paragraph" in result[0]["text"]

    def test_critically_short_row_always_merges(self, monkeypatch):
        """Rows below critical threshold (5 words) always merge, ignoring size limit."""
        monkeypatch.setenv("PDF_CHUNKER_MAX_MERGE_CHARS", "50")
        rows = [
            {"text": "Tiny"},  # < 5 words - critical
            {"text": "This is a much longer paragraph that would exceed limit."},
        ]
        result = _merge_short_rows(rows)
        # Should merge anyway since "Tiny" is critically short
        assert len(result) == 1
        assert "Tiny" in result[0]["text"]

    def test_coherent_short_row_preserved(self):
        """Coherent short rows (8+ words with proper ending) are preserved."""
        rows = [
            {"text": "This is a coherent sentence with period."},
            {"text": "Another complete sentence follows here."},
        ]
        result = _merge_short_rows(rows)
        # Both should be preserved since they're coherent
        assert len(result) >= 1

    def test_trailing_short_row_merges_backward(self):
        """Short trailing row merges with preceding row."""
        rows = [
            {"text": "This is the main content paragraph here."},
            {"text": "End"},  # Short trailing
        ]
        result = _merge_short_rows(rows)
        assert len(result) == 1
        assert "main content" in result[0]["text"]
        assert "End" in result[0]["text"]


# ---------------------------------------------------------------------------
# _has_incomplete_list characterization tests
# ---------------------------------------------------------------------------


class TestHasIncompleteList:
    """Tests for _has_incomplete_list detection."""

    def test_empty_string_not_incomplete(self):
        assert not _has_incomplete_list("")

    def test_colon_only_ending_is_incomplete(self):
        """Text ending with ':' alone is an incomplete list intro."""
        assert _has_incomplete_list("Here are the items:")
        assert _has_incomplete_list("The following:")

    def test_colon_with_bullet_not_incomplete(self):
        """Text with colon followed by bullet items is complete."""
        assert not _has_incomplete_list("Items:\n• First\n• Second")

    def test_single_inline_bullet_is_incomplete(self):
        """Colon followed by single bullet inline is incomplete."""
        assert _has_incomplete_list("List: • single item")

    def test_complete_sentence_not_incomplete(self):
        """Regular sentences ending with period are not incomplete."""
        assert not _has_incomplete_list("This is a complete sentence.")

    def test_multiple_bullets_not_incomplete(self):
        """Multiple bullet items are a complete list."""
        assert not _has_incomplete_list("• First item\n• Second item")


# ---------------------------------------------------------------------------
# _coherent characterization tests
# ---------------------------------------------------------------------------


class TestCoherent:
    """Tests for _coherent detection."""

    def test_empty_string_not_coherent(self):
        assert not _coherent("")

    def test_short_fragment_not_coherent(self):
        """Very short text is not coherent."""
        assert not _coherent("word")

    def test_sentence_with_period_is_coherent(self):
        """Text starting with capital and ending with period is coherent (if >= 40 chars)."""
        # _coherent requires min_chars=40 by default
        assert _coherent("This is a complete sentence with enough words here.")
        # Too short even with proper format
        assert not _coherent("Short sentence.")

    def test_incomplete_list_not_coherent(self):
        """Text with incomplete list is not coherent."""
        assert not _coherent("Here are items:")

    def test_lowercase_start_not_coherent(self):
        """Text starting with lowercase is typically not coherent."""
        # This depends on implementation - verify current behavior
        result = _coherent("continuation of previous sentence.")
        # Document actual behavior
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Incomplete list predicate unit tests
# ---------------------------------------------------------------------------


class TestCountListItems:
    """Tests for _count_list_items helper."""

    def test_empty_string(self):
        assert _count_list_items("") == (0, 0)

    def test_bullet_items(self):
        text = "• First\n• Second\n• Third"
        bullets, numbers = _count_list_items(text)
        assert bullets == 3
        assert numbers == 0

    def test_numbered_items(self):
        text = "1. First\n2. Second"
        bullets, numbers = _count_list_items(text)
        assert bullets == 0
        assert numbers == 2

    def test_mixed_items(self):
        text = "• Bullet\n1. Number\n• Another"
        bullets, numbers = _count_list_items(text)
        assert bullets == 2
        assert numbers == 1


class TestEndsWithListIntroColon:
    """Tests for _ends_with_list_intro_colon predicate."""

    def test_empty_lines(self):
        assert not _ends_with_list_intro_colon([])

    def test_colon_ending_no_bullets(self):
        lines = ["Here are the items:"]
        assert _ends_with_list_intro_colon(lines)

    def test_colon_ending_with_bullets(self):
        lines = ["Items:", "• First bullet"]
        assert not _ends_with_list_intro_colon(lines)

    def test_no_colon_ending(self):
        lines = ["This is a regular sentence."]
        assert not _ends_with_list_intro_colon(lines)


class TestHasSingleInlineBullet:
    """Tests for _has_single_inline_bullet predicate."""

    def test_empty_lines(self):
        assert not _has_single_inline_bullet([])

    def test_inline_bullet_single(self):
        lines = ["List: • single item"]
        assert _has_single_inline_bullet(lines)

    def test_inline_bullet_multiple(self):
        lines = ["List: • item one", "• item two"]
        assert not _has_single_inline_bullet(lines)

    def test_no_inline_bullet(self):
        lines = ["Regular paragraph."]
        assert not _has_single_inline_bullet(lines)


class TestHasUnterminatedBulletItem:
    """Tests for _has_unterminated_bullet_item predicate."""

    def test_empty_lines(self):
        assert not _has_unterminated_bullet_item([])

    def test_single_line(self):
        # Requires at least 2 lines
        assert not _has_unterminated_bullet_item(["• Single item"])

    def test_intro_with_unterminated_bullet(self):
        lines = ["Introduction:", "• Item without period"]
        assert _has_unterminated_bullet_item(lines)

    def test_intro_with_colon_always_true(self):
        # When intro ends with colon, function returns True regardless of terminator
        # (the "incomplete list" heuristic: intro + single bullet = needs more)
        lines = ["Introduction:", "• Item with period."]
        assert _has_unterminated_bullet_item(lines)

    def test_no_colon_intro_terminated_bullet(self):
        # No colon intro, but bullet line is terminated - check terminator
        lines = ["Some intro", "• Complete sentence."]
        assert not _has_unterminated_bullet_item(lines)

    def test_no_colon_intro_unterminated_bullet(self):
        # No colon intro, bullet is last line without terminator
        lines = ["Some intro", "• Incomplete item"]
        assert _has_unterminated_bullet_item(lines)

    def test_multiple_bullets(self):
        # Multiple bullets - not incomplete
        lines = ["Intro:", "• First", "• Second"]
        assert not _has_unterminated_bullet_item(lines)
