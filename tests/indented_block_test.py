import sys
from dataclasses import asdict

sys.path.insert(0, ".")

import pytest
from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


@pytest.mark.skip(reason="synthetic-bullets.pdf no longer contains an indented block example")
def test_indented_block_no_double_newline():
    blocks = [asdict(b) for b in extract_text_blocks_from_pdf("synthetic-bullets.pdf")]
    blob = "\n\n".join(b["text"] for b in blocks)
    snippet = "reduced coordination.\nA corollary here"
    assert snippet in blob
    assert "reduced coordination.\n\nA corollary here" not in blob
