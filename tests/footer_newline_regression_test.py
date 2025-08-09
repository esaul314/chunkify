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
    assert "Spanish main" in text
    assert "and then they will stay" in text
    assert "andthen" not in text

    second_chunk = chunks[1]["text"]
    assert "lambs. A car-load of drovers" in second_chunk
    assert "lambs.\n\nA car-load" not in second_chunk
