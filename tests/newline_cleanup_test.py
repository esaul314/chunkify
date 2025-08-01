import os
import unittest

from pdf_chunker.text_cleaning import clean_text


class TestNewlineCleanup(unittest.TestCase):
    def test_merge_lowercase_continuation(self):
        text = "My performance as a\n\nmanager."
        self.assertEqual(clean_text(text), "My performance as a manager.")

    def test_merge_sentence_continuation(self):
        text = "be reknit with purpose.\n\nInstead, Stacy listened calmly."
        self.assertEqual(
            clean_text(text),
            "be reknit with purpose. Instead, Stacy listened calmly.",
        )

    def test_preserve_heading_break(self):
        text = "Chapter 1\n\nIntroduction paragraph starts here."
        cleaned = clean_text(text)
        self.assertIn("\n\n", cleaned)

    def test_merge_break_in_quoted_title(self):
        text = (
            '" Vulnerability sounds like truth": Brené Brown, '
            "_Daring Greatly: How the Courage to Be Vulnerable_\n\n"
            "_Transforms the Way We Live..."
        )
        expected = (
            '" Vulnerability sounds like truth": Brené Brown, '
            "_Daring Greatly: How the Courage to Be Vulnerable_ _Transforms the Way We Live..."
        )
        self.assertEqual(clean_text(text), expected)

    def test_merge_break_in_quoted_phrase(self):
        text = (
            'Reese Witherspoon confessing: Reese Witherspoon, " Reese Witherspoon Shares Her Lean In Story," '
            "Lean\n\nIn."
        )
        expected = (
            'Reese Witherspoon confessing: Reese Witherspoon, " Reese Witherspoon Shares Her Lean In Story," '
            "Lean In."
        )
        self.assertEqual(clean_text(text), expected)

    def test_merge_break_in_quoted_headline(self):
        text = '" President Draws Planning Moral: Recalls Army Days to Show\n\nValue of Preparedness in Time of Crisis,"'
        expected = '" President Draws Planning Moral: Recalls Army Days to Show Value of Preparedness in Time of Crisis,"'
        self.assertEqual(clean_text(text), expected)

    def test_collapse_mid_sentence_double_newline(self):
        text = "Moving to modern\n\nenvironments requires care."
        self.assertEqual(
            clean_text(text),
            "Moving to modern environments requires care.",
        )

    def test_collapse_bullet_linebreak(self):
        text = "and also\n• correlated"
        self.assertEqual(clean_text(text), "and also correlated")

    def test_preserve_bullet_list(self):
        text = "* item 1\n* item 2"
        cleaned = clean_text(text)
        self.assertEqual(cleaned.count("*"), 1)


if __name__ == "__main__":
    unittest.main()
