import sys

sys.path.insert(0, ".")

from pdf_chunker.page_artifacts import remove_page_artifact_lines
from pdf_chunker.text_cleaning import clean_text


def test_numbered_item_with_footnote_continuation() -> None:
    raw = "1. Some text can exist.3\n\nThis is still part of the same item"
    cleaned = clean_text(remove_page_artifact_lines(raw, None))
    assert "exist.[3] This" in cleaned
    assert "\n\nThis is" not in cleaned
