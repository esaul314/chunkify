import sys
from pathlib import Path

sys.path.insert(0, ".")

from pdf_chunker.core import process_document


def test_footer_newlines_joined():
    pdf = Path("sample_book-footer.pdf").resolve()
    chunks = process_document(str(pdf), 400, 50)
    text = " ".join(chunk["text"] for chunk in chunks)
    assert "putting the perseverance" in text
    assert "angle of elevation" in text
    assert "tails exhibit is" in text
    assert "practically speaking" in text
