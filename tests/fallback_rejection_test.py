import pdf_chunker.fallbacks as mod
from pdf_chunker.pdf_blocks import Block


def test_single_block_fallback_rejected(monkeypatch):
    original = [
        Block(text="ok", source={"page": 1}),
        Block(text="more", source={"page": 2}),
    ]

    def fake_assess(text: str) -> dict[str, float]:
        return {"quality_score": 0.9 if "fallback" in text else 0.1}

    def fake_extractor(filepath: str, exclude_pages: str | None):
        return [{"text": "fallback", "source": {"page": 1}}]

    monkeypatch.setattr(mod, "_assess_text_quality", fake_assess)
    monkeypatch.setattr(mod, "_extract_with_pdftotext", fake_extractor)
    monkeypatch.setattr(mod, "_extract_with_pdfminer", fake_extractor)

    result = mod.apply_fallbacks(original, "dummy.pdf", set())
    assert list(result) == original


def test_page_truncating_fallback_rejected(monkeypatch):
    original = [
        Block(text="a", source={"page": 1}),
        Block(text="b", source={"page": 2}),
    ]
    fallback = [
        {"text": "f1", "source": {"page": 1}},
        {"text": "f2", "source": {"page": 1}},
    ]

    def fake_assess(text: str) -> dict[str, float]:
        return {"quality_score": 0.9 if "f" in text else 0.1}

    def fake_extractor(filepath: str, exclude_pages: str | None):
        return fallback

    monkeypatch.setattr(mod, "_assess_text_quality", fake_assess)
    monkeypatch.setattr(mod, "_extract_with_pdftotext", fake_extractor)
    monkeypatch.setattr(mod, "_extract_with_pdfminer", fake_extractor)

    result = mod.apply_fallbacks(original, "dummy.pdf", set())
    assert list(result) == original
