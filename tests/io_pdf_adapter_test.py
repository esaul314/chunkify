from pathlib import Path

from pdf_chunker.adapters import io_pdf


def test_io_pdf_read_returns_page_blocks():
    pdf_path = Path("test_data") / "sample_test.pdf"
    doc = io_pdf.read(str(pdf_path))
    assert doc["type"] == "page_blocks"
    assert doc["pages"]
    assert all(isinstance(p.get("blocks"), list) for p in doc["pages"])
    assert all("text" in b for p in doc["pages"] for b in p["blocks"])
