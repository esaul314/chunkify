import sys

sys.path.insert(0, ".")

from pdf_chunker.core import process_document


def test_hyphen_bullet_lists_preserved():
    chunks = process_document("sample_book-footer.pdf", 400, 50)
    text = "\n".join(c["text"] for c in chunks)
    bullet_lines = [line for line in text.splitlines() if line.startswith("• ")]
    expected = [
        "• Directed to John Smith, Cuttingsville, Vermont",
        "• Some trader among the Green Mountains",
        "• He expects some by the next train of prime quality",
    ]
    assert bullet_lines[: len(expected)] == expected
    assert "Vermont\n\n• Some trader" not in text
    bullet_chunks = [
        i
        for i, c in enumerate(chunks)
        if any(line.startswith("• ") for line in c["text"].splitlines())
    ]
    assert len(set(bullet_chunks)) == 1
