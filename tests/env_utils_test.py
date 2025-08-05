import pytest
from pdf_chunker.env_utils import use_pymupdf4llm


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        ("true", True),
        ("false", False),
        ("0", False),
        ("off", False),
    ],
)
def test_use_pymupdf4llm(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("PDF_CHUNKER_USE_PYMUPDF4LLM", raising=False)
    else:
        monkeypatch.setenv("PDF_CHUNKER_USE_PYMUPDF4LLM", value)
    assert use_pymupdf4llm() is expected
