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


if __name__ == "__main__":
    unittest.main()
