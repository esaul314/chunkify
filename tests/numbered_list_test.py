import sys
import re
import pytest

sys.path.insert(0, ".")

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf
from pdf_chunker.text_cleaning import (
    clean_text,
    collapse_single_newlines,
    insert_numbered_list_newlines,
)


def test_numbered_list_preservation():
    blocks = extract_text_blocks_from_pdf("sample_book0-1.pdf")
    blob = "\n\n".join(b["text"] for b in blocks)
    items = [
        line.strip() for line in blob.splitlines() if re.match(r"\d+\.", line.strip())
    ]
    assert len(items) == 4
    assert "\n\n2." not in blob
    assert "\n\n3." not in blob
    assert "\n\n4." not in blob


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("scope of Tier\n1.", "scope of Tier 1."),
        ("scope of Tier\n1.\nNext", "scope of Tier 1. Next"),
        (
            "scope of Tier\n1. T2 support engineers",
            "scope of Tier 1. T2 support engineers",
        ),
    ],
)
def test_number_suffix_not_list(raw: str, expected: str) -> None:
    assert collapse_single_newlines(raw) == expected


def test_abbreviation_inside_numbered_item() -> None:
    text = (
        "1. First item.\n"
        "2. Second item references the SaaS paradigm for clarity.\n"
        "Following paragraph."
    )
    cleaned = insert_numbered_list_newlines(text)
    cleaned = collapse_single_newlines(cleaned)
    assert "the\n\nSaaS" not in cleaned
    assert "paradigm for clarity.\n\nFollowing" in cleaned


def test_embedded_sentence_with_quotes() -> None:
    text = (
        '1. whosoever that can answer the question "Why did this need to happen at all?"\n'
        "Then come towards"
    )
    cleaned = clean_text(text)
    assert "\n\nThen" not in cleaned
    assert '"Why did this need to happen at all?" Then' in cleaned
