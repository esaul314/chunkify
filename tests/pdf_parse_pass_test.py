import pdf_chunker.pdf_parsing as pdf_parsing
import pdf_chunker.passes.pdf_parse as pdf_parse_mod
from pdf_chunker.framework import Artifact


def test_pdf_parse_normalizes_blocks(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("legacy extractor called")

    monkeypatch.setattr(pdf_parsing, "_legacy_extract_text_blocks_from_pdf", boom)

    blocks = [
        {"source": {"page": 2}, "text": "two"},
        {"source": {"page": None}, "text": "none"},
        {"source": {"page": 1}, "text": "one"},
    ]

    artifact = Artifact(payload=blocks, meta={"input": "dummy.pdf"})
    result = pdf_parse_mod.pdf_parse(artifact)
    expected = pdf_parse_mod._to_page_blocks(blocks, "dummy.pdf")

    assert result.payload == expected
    assert result.meta["metrics"]["pdf_parse"]["pages"] == len(expected["pages"])
