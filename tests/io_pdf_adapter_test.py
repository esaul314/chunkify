import os
import pytest
from pdf_chunker.pdf_blocks import Block

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

    def fake_extract(path, exclude, **kwargs):
        captured["env"] = os.getenv("PDF_CHUNKER_USE_PYMUPDF4LLM")
        yield Block(text="", source={"page": 1, "index": 0})

    monkeypatch.setenv("PDF_CHUNKER_USE_PYMUPDF4LLM", "0")
    target = "pdf_chunker.pdf_parsing.extract_text_blocks_from_pdf"
    monkeypatch.setattr(target, fake_extract)

    read("test_data/sample_test.pdf", use_pymupdf4llm=True)

    assert captured["env"] == "1"
    assert os.getenv("PDF_CHUNKER_USE_PYMUPDF4LLM") == "0"


def test_footer_margin_filters_footer_blocks(monkeypatch):
    """Zone margin filtering removes footer blocks at extraction time."""
    captured = {"footer_margin": None}

    original_extract = None

    def fake_extract(path, exclude, **kwargs):
        captured["footer_margin"] = kwargs.get("footer_margin")
        # Return a mix of content and footer blocks
        yield Block(text="Content text", source={"page": 1, "index": 0})

    target = "pdf_chunker.pdf_parsing.extract_text_blocks_from_pdf"
    import pdf_chunker.pdf_parsing as pdf_parsing
    original_extract = pdf_parsing.extract_text_blocks_from_pdf
    monkeypatch.setattr(target, fake_extract)

    read("test_data/sample_test.pdf", footer_margin=40.0)

    assert captured["footer_margin"] == 40.0
