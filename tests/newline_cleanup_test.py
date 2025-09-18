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

    def test_numbered_item_with_numeric_reference(self):
        text = (
            "2. Another item to mention in Chapter 10.\n\n"
            "Considering this issue, no decision was made. The paragraph continues."
        )
        expected = (
            "2. Another item to mention in Chapter 10.\n\n"
            "Considering this issue, no decision was made. The paragraph continues."
        )
        self.assertEqual(clean_text(text), expected)

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

    def test_merge_quote_with_author_line(self):
        text = (
            "What recommends commerce to me is its enterprise and bravery.\n\n"
            "—Author Name, Book Name"
        )
        expected = (
            "What recommends commerce to me is its enterprise and bravery. "
            "—Author Name, Book Name"
        )
        self.assertEqual(clean_text(text), expected)

    def test_stray_bullet_line_with_trailing_space_removed(self):
        text = (
            "• If we were always indeed getting our living, and regulating our lives according \n"
            "• \n"
            "swamp"
        )
        expected = "• If we were always indeed getting our living, and regulating our lives according swamp"
        self.assertEqual(clean_text(text), expected)


if __name__ == "__main__":
    unittest.main()
