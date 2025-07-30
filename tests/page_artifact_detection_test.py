import unittest

from pdf_chunker.page_artifacts import (
    is_page_artifact_text,
    strip_page_artifact_suffix,
)
from pdf_chunker.pdf_parsing import _remove_page_artifact_lines


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

    def test_remove_page_artifact_lines(self):
        text = "Hello\nA Successful Implementation: How To Do It | 123\nWorld"
        cleaned = _remove_page_artifact_lines(text, 123)
        self.assertEqual(cleaned, "Hello\nWorld")


if __name__ == "__main__":
    unittest.main()
