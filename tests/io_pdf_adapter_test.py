import pytest

pytest.importorskip("fitz")

from pdf_chunker.adapters.io_pdf import read


def test_read_returns_page_blocks():
    doc = read("test_data/sample_test.pdf")
    assert doc["type"] == "page_blocks"
    assert doc["pages"] and doc["pages"][0]["page"] == 1


def test_exclude_pages_list():
    doc = read("test_data/sample_test.pdf", exclude_pages=[1])
    assert all(p["page"] != 1 for p in doc["pages"])
