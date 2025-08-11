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
    assert "our eyes?\n\u2022 There" in text
    assert "gone by. \u2022 They would" not in text
    assert "gone by.\n\n\u2022 They would" in text
    assert (
        "\u2022 They would begin to sing almost with as much precision as a clock,"
        " within five minutes of a particular time, referred to the setting of the sun,"
        " every evening." in text
    )
    assert "by accident one\n\u2022 a bar behind another" not in text
    assert "mourning women\n\u2022 their" not in text
    assert "(Souls\n\u2022 that" not in text
