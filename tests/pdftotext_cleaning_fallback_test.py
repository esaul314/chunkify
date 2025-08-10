from pdf_chunker.extraction_fallbacks import _text_to_blocks


def test_text_to_blocks_uses_raw_when_clean_empty(monkeypatch):
    monkeypatch.setattr("pdf_chunker.extraction_fallbacks.clean_text", lambda p: "")
    blocks = _text_to_blocks("Paragraph one\n\nParagraph two", "file.pdf", "pdftotext")
    assert [b["text"] for b in blocks] == ["Paragraph one", "Paragraph two"]


def test_text_to_blocks_handles_single_paragraph(monkeypatch):
    monkeypatch.setattr("pdf_chunker.extraction_fallbacks.clean_text", lambda p: "")
    text = "Solo paragraph without breaks"
    blocks = _text_to_blocks(text, "file.pdf", "pdftotext")
    assert [b["text"] for b in blocks] == [text]
