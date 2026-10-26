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
            '" A clear label describes a record": Sample Writer, '
            "_Reading Carefully: Why Invented Examples Matter_\n\n"
            "_Changes the Way We Review..."
        )
        expected = (
            '" A clear label describes a record": Sample Writer, '
            "_Reading Carefully: Why Invented Examples Matter_ _Changes the Way We Review..."
        )
        self.assertEqual(clean_text(text), expected)

    def test_merge_break_in_quoted_phrase(self):
        text = (
            'Example Reader confessing: Example Reader, " Example Reader Shares Her Review Notes," '
            "Review\n\nNotes."
        )
        expected = (
            'Example Reader confessing: Example Reader, " Example Reader Shares Her Review Notes," '
            "Review Notes."
        )
        self.assertEqual(clean_text(text), expected)

    def test_merge_break_in_quoted_headline(self):
        text = '" Example Review Connects Separate Notes: Records Small Steps to Show\n\nValue of Clear Labels During a Workshop,"'
        expected = '" Example Review Connects Separate Notes: Records Small Steps to Show Value of Clear Labels During a Workshop,"'
        self.assertEqual(clean_text(text), expected)

    def test_merge_break_in_fully_quoted_title_with_colon(self):
        text = '"The Power of Words: Understanding Language"\n\n' "continues the discussion."
        expected = '"The Power of Words: Understanding Language" continues the discussion.'
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

    def test_bullet_item_continuation_is_joined(self):
        text = (
            "Reminding you of the difference:\n"
            "• With a reusable review record (RR), the editor's tools retain complete notes describing invented examples and original locations without borrowing external passages.\n"
            "• With an ordered review sheet (RS), the editor takes full responsibility for checking\n"
            "The example's labels, which means every recorded location stays available while reviewers compare independent decisions and retain complete invented passages."
        )
        expected = (
            "Reminding you of the difference:\n"
            "• With a reusable review record (RR), the editor's tools retain complete notes describing invented examples and original locations without borrowing external passages.\n"
            "• With an ordered review sheet (RS), the editor takes full responsibility for checking the example's labels, which means every recorded location stays available while reviewers compare independent decisions and retain complete invented passages."
        )
        self.assertEqual(clean_text(text), expected)


if __name__ == "__main__":
    unittest.main()
