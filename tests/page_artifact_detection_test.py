import unittest

from pdf_chunker.page_artifacts import (
    is_page_artifact_text,
    strip_page_artifact_suffix,
    remove_page_artifact_lines,
)


class TestPageArtifactDetection(unittest.TestCase):
    def test_footer_with_pipe(self):
        line = "Creating the Platform Teams That Replace Cooperation | 55"
        self.assertTrue(is_page_artifact_text(line, 55))

    def test_footer_without_pipe(self):
        line = "4.1 Develop Project Charter 43"
        self.assertTrue(is_page_artifact_text(line, 43))

    def test_non_artifact_line(self):
        line = "This is page 5 of our analysis"
        self.assertFalse(is_page_artifact_text(line, 5))

    def test_strip_page_artifact_suffix(self):
        line = "An Introduction to Something | 123"
        stripped = strip_page_artifact_suffix(line, 123)
        self.assertEqual(stripped, "An Introduction to Something")

    def test_footer_with_trailing_text(self):
        line = "Footer Text | 55the next paragraph"
        self.assertTrue(is_page_artifact_text(line, 55))
        stripped = strip_page_artifact_suffix(line, 55)
        self.assertEqual(stripped, "Footer Text")

    def test_remove_page_artifact_lines(self):
        text = "Hello\nA Successful Implementation: How To Do It | 123\nWorld"
        cleaned = remove_page_artifact_lines(text, 123)
        self.assertEqual(cleaned, "Hello\nWorld")

    def test_inline_footer_fragment(self):
        text = "Intro paragraph\n\nFooter Text | 55next line"
        cleaned = remove_page_artifact_lines(text, 55)
        self.assertEqual(cleaned, "Intro paragraph\nnext line")

    def test_footer_without_page_context(self):
        line = "Footer Text | 9"
        self.assertTrue(is_page_artifact_text(line, 0))
        cleaned = remove_page_artifact_lines(line, 0)
        self.assertEqual(cleaned, "")

    def test_footnote_detection(self):
        line = "4 This is a sample footnote text."
        self.assertTrue(is_page_artifact_text(line, 2))

    def test_no_false_footnote(self):
        line = "2 a.m. In our experience, this failed"
        self.assertFalse(is_page_artifact_text(line, 0))

    def test_remove_footnote_line(self):
        text = "Paragraph text\n4 This is a sample footnote text.\nNext paragraph"
        cleaned = remove_page_artifact_lines(text, 2)
        self.assertEqual(cleaned, "Paragraph text\nNext paragraph")

    def test_remove_header_and_footnote(self):
        text = (
            "First part of sentence\n\n"
            "115 | Chapter 3: Welcome to the Jungle\n"
            "4 Footnote text. The sentence continues here."
        )
        cleaned = remove_page_artifact_lines(text, 115)
        self.assertEqual(
            cleaned, "First part of sentence\nThe sentence continues here."
        )


if __name__ == "__main__":
    unittest.main()
