import unittest

from pdf_chunker.heading_detection import (
    _detect_heading_fallback,
    detect_headings_from_font_analysis,
)


class TestHeadingDetectionFallback(unittest.TestCase):
    def test_short_sentence_with_period_not_heading(self) -> None:
        """Short sentences ending with a period should not be treated as headings."""
        self.assertFalse(_detect_heading_fallback("We agree."))

    def test_chapter_reference_with_period_not_heading(self) -> None:
        """References like 'Chapter 10.' should remain body text."""
        self.assertFalse(_detect_heading_fallback("Chapter 10."))

    def test_block_marked_heading_with_period_not_heading(self) -> None:
        """Blocks tagged as headings but ending with punctuation stay paragraphs."""
        blocks = [{"text": "Chapter 10.", "type": "heading"}]
        self.assertEqual(detect_headings_from_font_analysis(blocks), [])


if __name__ == "__main__":
    unittest.main()
