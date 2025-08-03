import sys
import re

sys.path.insert(0, ".")

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


def test_numbered_list_preservation():
    blocks = extract_text_blocks_from_pdf("sample_book0-1.pdf")
    blob = "\n\n".join(b["text"] for b in blocks)
    items = [
        line.strip() for line in blob.splitlines() if re.match(r"\d+\.", line.strip())
    ]
    assert len(items) == 4
    assert "\n\n2." not in blob
    assert "\n\n3." not in blob
    assert "\n\n4." not in blob
