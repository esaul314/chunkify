import sys

sys.path.insert(0, ".")

from pdf_chunker.core import process_document


def test_footer_table_cleanup():
    chunks = process_document("sample_book-footer.pdf", 400, 50)
    first_lines = chunks[0]["text"].splitlines()[:3]
    assert first_lines == [
        "This closed car smells of salt fish",
        "Person Name, PMP",
        "Alma, Quebec, Canada",
    ]
    assert chunks[0]["text"].count("Person Name, PMP") == 1
