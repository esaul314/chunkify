import unittest

from pdf_chunker.heading_detection import _detect_heading_fallback


class TestHeadingDetectionFallback(unittest.TestCase):
    def test_short_sentence_with_period_not_heading(self) -> None:
        """Short sentences ending with a period should not be treated as headings."""
        self.assertFalse(_detect_heading_fallback("We agree."))


if __name__ == "__main__":
    unittest.main()
