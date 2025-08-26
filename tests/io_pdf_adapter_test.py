import os
import pytest

pytest.importorskip("fitz")

from pdf_chunker.adapters.io_pdf import read  # noqa: E402


def test_read_returns_page_blocks():
    doc = read("test_data/sample_test.pdf")
    pages = [p["page"] for p in doc["pages"]]
    assert doc["type"] == "page_blocks"
    assert pages == sorted(pages)


def test_exclude_pages_list():
    doc = read("test_data/sample_test.pdf", exclude_pages=[1])
    assert all(p["page"] != 1 for p in doc["pages"])


def test_use_pymupdf4llm_env(monkeypatch):
    captured = {}

    def fake_extract(path, exclude):
        captured["env"] = os.getenv("PDF_CHUNKER_USE_PYMUPDF4LLM")
        return [{"source": {"page": 1, "index": 0}}]

    monkeypatch.setenv("PDF_CHUNKER_USE_PYMUPDF4LLM", "0")
    target = "pdf_chunker.pdf_parsing.extract_text_blocks_from_pdf"
    monkeypatch.setattr(target, fake_extract)

    read("test_data/sample_test.pdf", use_pymupdf4llm=True)

    assert captured["env"] == "1"
    assert os.getenv("PDF_CHUNKER_USE_PYMUPDF4LLM") == "0"
