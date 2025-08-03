import sys

sys.path.insert(0, ".")

import re

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


def test_bullet_list_preservation():
    blocks = extract_text_blocks_from_pdf("sample-local-pdf.pdf")
    blob = "\n".join(b["text"] for b in blocks)
    lines = blob.splitlines()
    start = lines.index("Bullet points:") + 1
    items = []
    for line in lines[start:]:
        if line.strip().startswith("-"):
            items.append(line.strip())
        else:
            break
    assert len(items) == 3
    assert "- First bullet point" in items
    assert all(not item.rstrip().endswith(".") for item in items)
    assert "\n\n-" not in blob
    assert "-\n\n-" not in blob
