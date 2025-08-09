import sys

sys.path.insert(0, ".")

from pdf_chunker.core import process_document


def test_hyphen_bullet_lists_preserved():
    chunks = process_document("sample_book-footer.pdf", 400, 50)
    text = chunks[0]["text"]
    bullet_lines = [line for line in text.splitlines() if line.startswith("- ")]
    assert bullet_lines == [
        "- Directed to John Smith, Cuttingsville, Vermont",
        "- Some trader among the Green Mountains",
        "- He expects some by the next train of prime quality",
    ]
    assert "Vermont\n\n- Some trader" not in text
