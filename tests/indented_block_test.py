import sys

sys.path.insert(0, ".")

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


def test_indented_block_no_double_newline():
    blocks = extract_text_blocks_from_pdf("sample_book3.pdf")
    blob = "\n\n".join(b["text"] for b in blocks)
    snippet = "reduced coordination.\nA corollary here"
    assert snippet in blob
    assert "reduced coordination.\n\nA corollary here" not in blob
