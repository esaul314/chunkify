import os
import sys
import unittest

sys.path.insert(0, ".")

from pdf_chunker.text_cleaning import clean_text
import pdf_chunker.pymupdf4llm_integration as p4l


class TestHyphenationFix(unittest.TestCase):
    def test_hyphenation_fix_with_pymupdf4llm(self):
        sample = "a con-\n tainer and special-\n ists in man-\n agement"
        os.environ["PDF_CHUNKER_USE_PYMUPDF4LLM"] = "true"
        original = p4l.is_pymupdf4llm_available
        try:
            p4l.is_pymupdf4llm_available = lambda: True
            cleaned = clean_text(sample)
        finally:
            p4l.is_pymupdf4llm_available = original
            del os.environ["PDF_CHUNKER_USE_PYMUPDF4LLM"]
        self.assertIn("container", cleaned)
        self.assertIn("specialists", cleaned)
        self.assertIn("management", cleaned)
        self.assertNotIn("con- tainer", cleaned)
        self.assertNotIn("special- ists", cleaned)
        self.assertNotIn("man- agement", cleaned)


if __name__ == "__main__":
    unittest.main()
