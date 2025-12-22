#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")
from pdf_chunker.fallbacks import _extract_with_pdfminer


def test_page_exclusion_fallback():
    blocks = _extract_with_pdfminer("test_data/sample_test.pdf", exclude_pages="1")
    combined = " ".join(b.get("text", "") for b in blocks)
    assert "This is a test document" not in combined


if __name__ == "__main__":
    try:
        test_page_exclusion_fallback()
        sys.exit(0)
    except AssertionError:
        sys.exit(1)
