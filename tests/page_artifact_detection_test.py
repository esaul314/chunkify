import unittest

from pdf_chunker.page_artifacts import (
    is_page_artifact_text,
    strip_page_artifact_suffix,
    remove_page_artifact_lines,
)
from pdf_chunker.pymupdf4llm_integration import _clean_pymupdf4llm_block


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

    def test_trailing_pipe_footer(self):
        text = (
            "Intro paragraph\n"
            "The Gardening Approach to the Product Management |\n"
            "Next paragraph"
        )
        cleaned = remove_page_artifact_lines(text, 0)
        self.assertEqual(cleaned, "Intro paragraph\nNext paragraph")

    def test_footer_without_page_context(self):
        line = "Footer Text | 9"
        self.assertTrue(is_page_artifact_text(line, 0))
        cleaned = remove_page_artifact_lines(line, 0)
        self.assertEqual(cleaned, "")

    def test_footnote_detection(self):
        line = "4  This is a sample footnote text."
        self.assertTrue(is_page_artifact_text(line, 2))

    def test_no_false_footnote(self):
        line = "2 a.m. In our experience, this failed"
        self.assertFalse(is_page_artifact_text(line, 0))

    def test_numbered_list_item_not_artifact(self):
        line = "3 This item should remain"
        self.assertFalse(is_page_artifact_text(line, 0))

    def test_remove_footnote_line(self):
        text = "Paragraph text\n4  This is a sample footnote text.\nNext paragraph"
        cleaned = remove_page_artifact_lines(text, 2)
        self.assertEqual(cleaned, "Paragraph text\nNext paragraph")

    def test_inline_footnote_marker(self):
        cases = [
            "Can exist.3\n3  Footnote text.\n\nThis is next",
            "Can exist. 3\n3  Footnote text.\n\nThis is next",
        ]
        self.assertTrue(
            all(
                remove_page_artifact_lines(text, 2) == "Can exist[3]. This is next"
                for text in cases
            )
        )

    def test_superscript_footnote_marker(self):
        text = "Can exist.\u00b3\n\u00b3 Footnote text.\n\nThis is next"
        cleaned = remove_page_artifact_lines(text, 2)
        self.assertEqual(cleaned, "Can exist[3]. This is next")

    def test_crlf_footnote_marker(self):
        text = "Can exist.3\r\n3  Footnote text.\r\n\r\nThis is next"
        cleaned = remove_page_artifact_lines(text, 2)
        self.assertEqual(cleaned, "Can exist[3]. This is next")

    def test_remove_header_and_footnote(self):
        text = (
            "First part of sentence\n\n"
            "115 | Chapter 3: Welcome to the Jungle\n"
            "4  Footnote text. The sentence continues here."
        )
        cleaned = remove_page_artifact_lines(text, 115)
        self.assertEqual(
            cleaned, "First part of sentence\nThe sentence continues here."
        )

    def test_header_inserted_mid_sentence(self):
        text = (
            "A sentence on the last line of the page and it continues on the next page\n"
            "| Chapter 3: Welcome to the Jungle\n"
            "4  And then there is a footnote at the bottom of the second page.\n"
            "the sentence continues here, on the next page."
        )
        cleaned = remove_page_artifact_lines(text, 115)
        expected = (
            "A sentence on the last line of the page and it continues on the next page\n"
            "the sentence continues here, on the next page."
        )
        self.assertEqual(cleaned, expected)

    def test_roman_numeral_footer(self):
        line = "Preface | xix"
        self.assertTrue(is_page_artifact_text(line, 19))
        cleaned = remove_page_artifact_lines(line, 19)
        self.assertEqual(cleaned, "")

    def test_markdown_table_header_normalization(self):
        table_text = (
            "|This closed car smells of salt fish|Col2|\n"
            "|---|---|\n"
            "|salt fish||\n"
            "|Person Name, PMP<br>Alma, Quebec, Canada|Person Name, PMP<br>Alma, Quebec, Canada|"
        )
        expected = (
            "This closed car smells of salt fish\n"
            "Person Name, PMP\n"
            "Alma, Quebec, Canada"
        )
        cleaned = remove_page_artifact_lines(table_text, 1)
        self.assertEqual(cleaned, expected)

    def test_pymupdf4llm_block_normalization(self):
        table_text = (
            "|This closed car smells of salt fish|Col2|\n"
            "|---|---|\n"
            "|salt fish||\n"
            "|Person Name, PMP<br>Alma, Quebec, Canada|Person Name, PMP<br>Alma, Quebec, Canada|"
        )
        block = {"text": table_text, "source": {"page": 1}}
        cleaned = _clean_pymupdf4llm_block(block)
        self.assertIsNotNone(cleaned)
        self.assertEqual(
            cleaned["text"],
            "This closed car smells of salt fish\nPerson Name, PMP\nAlma, Quebec, Canada",
        )


if __name__ == "__main__":
    unittest.main()
