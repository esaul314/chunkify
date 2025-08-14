import sys
import re
import pytest

sys.path.insert(0, ".")

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf
from pdf_chunker.text_cleaning import (
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


def test_quoted_question_inside_numbered_item() -> None:
    text = (
        '1. Item with a quote "Why did this need to happen at all?" '
        "Then proceed with more details."
    )
    cleaned = insert_numbered_list_newlines(text)
    cleaned = collapse_single_newlines(cleaned)
    assert "\n\nThen" not in cleaned
    assert '"Why did this need to happen at all?" Then' in cleaned


@pytest.mark.parametrize("punct", list(".…"))
def test_quoted_sentence_endings_inside_numbered_item(punct: str) -> None:
    text = (
        f'1. Item with a quote "quoted sentence{punct}" '
        "Then proceed with more details."
    )
    cleaned = insert_numbered_list_newlines(text)
    cleaned = collapse_single_newlines(cleaned)
    assert "\n\nThen" not in cleaned
    assert f'"quoted sentence{punct}" Then' in cleaned


def test_long_inline_numbered_items() -> None:
    text = (
        "1. This item is fairly long and continues without a newline before the next number "
        "2. Second item should begin on its own line."
    )
    cleaned = insert_numbered_list_newlines(text)
    cleaned = collapse_single_newlines(cleaned)
    assert "next number 2." not in cleaned
    assert "next number\n2." in cleaned


def test_item_ending_with_chapter_preserves_newline() -> None:
    text = (
        "1. Intro text spanning lines\n"
        "continues and ends with Chapter 2. Second item"
    )
    cleaned = insert_numbered_list_newlines(text)
    cleaned = collapse_single_newlines(cleaned)
    assert "Chapter 2." not in cleaned
    assert "Chapter\n2." in cleaned
