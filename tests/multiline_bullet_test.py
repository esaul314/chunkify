import sys

sys.path.insert(0, ".")

from pdf_chunker.core import process_document


def test_multiline_bullet_items():
    chunks = process_document("sample_book-bullets.pdf", 400, 50)
    text = "\n".join(c["text"] for c in chunks)
    assert (
        "\u2022 All sound heard at the greatest possible distance? "
        "Produces one? Vibration? Atmosphere?" in text
    )
    assert "\u2022\n\nAtmosphere?" not in text
    assert "our eyes?\n\u2022\n\u2022 There" not in text
