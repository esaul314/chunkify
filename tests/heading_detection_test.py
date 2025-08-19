import unittest

from pdf_chunker.heading_detection import (
    _detect_heading_fallback,
    detect_headings_from_font_analysis,
)
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.heading_detect import heading_detect
from pdf_chunker.pdf_parsing import _spans_indicate_heading


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

    def test_font_flag_heading_with_period_not_heading(self) -> None:
        """Font-emphasized lines ending with punctuation should remain body text."""
        spans = [{"flags": 2}]
        self.assertFalse(
            _spans_indicate_heading(
                spans, "Considering this issue, no decision was made."
            )
        )


class TestHeadingDetectPass(unittest.TestCase):
    def test_pass_adds_heading_metadata(self) -> None:
        blocks = [{"text": "Introduction"}, {"text": "Body"}]
        artifact = Artifact(payload=blocks, meta={})
        result = heading_detect(artifact).payload
        self.assertTrue(result[0]["is_heading"])
        self.assertEqual(result[0]["heading_source"], "fallback")
        self.assertFalse(result[1]["is_heading"])


if __name__ == "__main__":
    unittest.main()
