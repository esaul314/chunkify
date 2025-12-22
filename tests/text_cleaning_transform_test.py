from pdf_chunker.framework import Artifact
from pdf_chunker.passes.text_clean import text_clean
from pdf_chunker.text_cleaning import clean_paragraph, clean_text


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


def test_clean_paragraph_rejoins_hyphenated_words():
    assert clean_paragraph("hy-\nphen") == "hyphen"


def test_clean_paragraph_strips_headers_and_footers():
    text = "My Book | 1\ncontent\n1 | My Book"
    assert clean_paragraph(text) == "content"


def test_clean_paragraph_cleans_bullet_fragments():
    text = "foo\n\u2022\nbar"
    assert clean_paragraph(text) == "foo bar"


def test_clean_paragraph_removes_underscore_wrapping():
    text = "This is __bold__ and _italics_"
    assert clean_paragraph(text) == "This is bold and italics"


def test_clean_paragraph_drops_dangling_underscores():
    text = "_start __bold__ mid_ end__"
    assert clean_paragraph(text) == "start bold mid end"


def test_quotes_and_control_chars():
    text = "Here\u202d are “quotes”\u202c"
    expected = 'Here are "quotes"'
    assert clean_paragraph(text) == expected

    doc = {"type": "page_blocks", "pages": [{"page": 1, "blocks": [{"text": text}]}]}
    cleaned = text_clean(Artifact(payload=doc)).payload["pages"][0]["blocks"][0]["text"]
    assert cleaned == expected


def test_clean_text_preserves_diacritics_without_underscores():
    sample = "ˠ"
    first_pass = clean_text(sample)
    second_pass = clean_text(first_pass)

    assert first_pass == sample
    assert second_pass == sample
    assert not first_pass.startswith("_")


def test_clean_text_does_not_reintroduce_bullet_markers():
    sample = "Intro:\n\n• Item"
    first_pass = clean_text(sample)
    second_pass = clean_text(first_pass)

    assert first_pass == second_pass
    assert "•" in first_pass
