import unittest
from pdf_chunker.splitter import _fix_heading_splitting_issues


class TestHeadingBoundaryFix(unittest.TestCase):
    def test_heading_moved_to_next_chunk(self):
        chunks = [
            "Some text about a topic.\n\nNext Section",
            "The section continues here with more details.",
        ]
        fixed = _fix_heading_splitting_issues(chunks)
        self.assertEqual(len(fixed), 2)
        self.assertEqual(fixed[0], "Some text about a topic.")
        self.assertTrue(fixed[1].startswith("Next Section"))
        self.assertIn("The section continues", fixed[1])

    def test_heading_with_punctuation(self):
        chunks = [
            "Preceding section text.\n\nKeep It Fun!",
            "This part explains how to keep it fun.",
        ]
        fixed = _fix_heading_splitting_issues(chunks)
        self.assertEqual(len(fixed), 2)
        self.assertEqual(fixed[0], "Preceding section text.")
        self.assertTrue(fixed[1].startswith("Keep It Fun!"))

    def test_footer_line_not_treated_as_heading(self):
        chunks = [
            "Intro text.\nProduct Discovery and Market Analysis |\n\nAssimilate and expand",
            "The section body follows here.",
        ]
        fixed = _fix_heading_splitting_issues(chunks)
        self.assertEqual(len(fixed), 2)
        self.assertTrue(fixed[0].endswith("|"))
        self.assertTrue(fixed[1].startswith("Assimilate and expand"))


if __name__ == "__main__":
    unittest.main()
