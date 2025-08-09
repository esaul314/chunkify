import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdf_chunker.core import process_document


def test_footer_and_subfooter_removed():
    pdf = Path(__file__).resolve().parent.parent / "sample_book-footer.pdf"
    texts = [c["text"] for c in process_document(str(pdf), 400, 50)]
    assert len(texts) == 1
    assert all("spam.com" not in t.lower() for t in texts)
    assert all("Bearings of Cattle Like Leaves Know" not in t for t in texts)
