import unittest

from pdf_chunker.page_artifacts import is_page_artifact_text


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


if __name__ == "__main__":
    unittest.main()
