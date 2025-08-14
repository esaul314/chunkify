import sys

sys.path.insert(0, ".")

from pdf_chunker.text_cleaning import clean_text


def test_inline_footnote_marker_converted() -> None:
    raw = "Some text can exist.3 Another sentence starts."
    cleaned = clean_text(raw)
    assert cleaned == "Some text can exist[3]. Another sentence starts."


def test_inline_footnote_at_end() -> None:
    raw = "They deserve.3"
    cleaned = clean_text(raw)
    assert cleaned == "They deserve[3]."


def test_bracketed_footnote_no_extra_newline() -> None:
    raw = "They deserve[3].\n\nThat follows"
    cleaned = clean_text(raw)
    assert cleaned == "They deserve[3]. That follows"
