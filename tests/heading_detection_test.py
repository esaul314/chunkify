import unittest

from pdf_chunker.heading_detection import _detect_heading_fallback


class TestHeadingDetectionFallback(unittest.TestCase):
    def test_short_sentence_with_period_not_heading(self) -> None:
        """Short sentences ending with a period should not be treated as headings."""
        self.assertFalse(_detect_heading_fallback("We agree."))

    def test_chapter_reference_with_period_not_heading(self) -> None:
        """References like 'Chapter 10.' should remain body text."""
        self.assertFalse(_detect_heading_fallback("Chapter 10."))


if __name__ == "__main__":
    unittest.main()
