import sys

sys.path.insert(0, ".")

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


def test_indented_block_no_double_newline():
    blocks = extract_text_blocks_from_pdf("sample-local-pdf.pdf")
    blob = "\n".join(b["text"] for b in blocks)
    snippet = "content to test:\n- Text extraction capabilities"
    assert snippet in blob
    assert "content to test:\n\n- Text extraction capabilities" not in blob
