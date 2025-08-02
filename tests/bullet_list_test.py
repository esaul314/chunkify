import sys

sys.path.insert(0, ".")

import re

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


def test_bullet_list_preservation():
    blocks = extract_text_blocks_from_pdf("sample_book3.pdf")
    text = "\n".join(b["text"] for b in blocks)
    items = [
        line[line.index("•") :].strip() for line in text.splitlines() if "•" in line
    ]
    assert len(items) == 3
    assert (
        "• How platform engineering manages this complexity and so frees us from the swamp"
        in items
    )
    assert all(not item.rstrip().endswith(".") for item in items)
