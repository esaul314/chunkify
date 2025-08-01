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

    def test_collapse_mid_sentence_double_newline_with_punctuation(self):
        text = "includes data)\n\nsuch as metrics"
        self.assertEqual(
            clean_text(text),
            "includes data) such as metrics",
        )

    def test_collapse_line_start_bullet_artifact(self):
        text = "and also\n\n• correlated"
        self.assertEqual(clean_text(text), "and also correlated")

    def test_collapse_bullet_linebreak(self):
        text = "and also\n• correlated"
        self.assertEqual(clean_text(text), "and also correlated")

    def test_preserve_bullet_list(self):
        text = "* item 1\n* item 2"
        cleaned = clean_text(text)
        self.assertEqual(cleaned.count("*"), 2)

    def test_join_bullet_continuation_line(self):
        text = "\u2022 item one\n  continuation"
        self.assertEqual(clean_text(text), "\u2022 item one continuation")

    def test_collapse_line_start_bullet_after_quote(self):
        text = 'end of quote"\n\n• continuation'
        self.assertEqual(clean_text(text), 'end of quote" continuation')

    def test_collapse_mid_sentence_double_newline_after_quote(self):
        text = 'this is "quote"\n\ncontinuation'
        self.assertEqual(clean_text(text), 'this is "quote" continuation')


if __name__ == "__main__":
    unittest.main()
