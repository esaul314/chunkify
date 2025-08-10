import re
import sys

sys.path.insert(0, ".")

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


def test_sample_local_pdf_lists_have_newlines():
    blocks = extract_text_blocks_from_pdf("sample-local-pdf.pdf")
    blob = "\n\n".join(b["text"] for b in blocks)

    assert ":\n1. First numbered item" in blob
    assert "\n2. Second numbered item" in blob
    assert "\n3. Third numbered item\n\nBullet points:" in blob

    numbered = [
        line.strip() for line in blob.splitlines() if re.match(r"\d+\.", line.strip())
    ]
    assert numbered[0] == "1. First numbered item"
    assert numbered[1] == "2. Second numbered item"
    assert numbered[2] == "3. Third numbered item"

    bullets = [
        line.strip() for line in blob.splitlines() if line.strip().startswith("•")
    ]
    assert "• First bullet point" in bullets
    assert "• Second bullet point" in bullets
    assert any(b.startswith("• Third bullet point") for b in bullets)
