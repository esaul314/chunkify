from pathlib import Path

from pdf_chunker.pdf_blocks import read_pages, PagePayload


def test_read_pages_counts_pages():
    pdf_path = Path(__file__).resolve().parent.parent / "platform-eng-excerpt.pdf"
    pages = list(read_pages(str(pdf_path), set()))
    assert len(pages) == 55
    assert all(isinstance(p, PagePayload) for p in pages)
