import sys

sys.path.insert(0, ".")

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


def test_numbered_list_after_colon_has_newlines():
    blocks = extract_text_blocks_from_pdf("sample-local-pdf.pdf")
    blob = "\n".join(b["text"] for b in blocks)
    expected = (
        "structured content processing:\n"
        "1. First numbered item\n"
        "2. Second numbered item\n"
        "3. Third numbered item"
    )
    assert expected in blob
