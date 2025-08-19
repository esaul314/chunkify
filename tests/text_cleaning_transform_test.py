from pdf_chunker.framework import Artifact
from pdf_chunker.passes.text_clean import text_clean


def test_text_cleaning_transform(pdf_case):
    raw, func, expected = pdf_case
    assert func(raw).rstrip() == expected


def test_text_clean_pass_normalizes_blocks(pdf_case):
    raw, _, expected = pdf_case
    doc = {"type": "page_blocks", "pages": [{"page": 1, "blocks": [{"text": raw}]}]}
    result = text_clean(Artifact(payload=doc))
    cleaned = result.payload["pages"][0]["blocks"][0]["text"].rstrip()
    assert cleaned == expected
    assert result.meta["metrics"]["normalized"] is True
