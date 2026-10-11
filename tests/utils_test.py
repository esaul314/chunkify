import pytest
from haystack.dataclasses import Document

from pdf_chunker.utils import (
    _build_char_map,
    _find_source_block,
    _normalize_sentence_boundaries,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("", ""),
        ("A single line\n", "A single line\n"),
        (
            "A line without punctuation\nNext begins uppercase",
            "A line without punctuation. \nNext begins uppercase",
        ),
        (
            "A line without punctuation\n  Next begins uppercase",
            "A line without punctuation. \n  Next begins uppercase",
        ),
        ("A finished sentence.\nNext sentence.", "A finished sentence.\nNext sentence."),
        ("A finished sentence!\nNext sentence.", "A finished sentence!\nNext sentence."),
        ("A finished sentence?\nNext sentence.", "A finished sentence?\nNext sentence."),
        ("Short title\nNext paragraph starts", "Short title\nNext paragraph starts"),
        ("A line of prose\nlowercase continuation", "A line of prose\nlowercase continuation"),
        ("A line of prose\n3 numbered items", "A line of prose\n3 numbered items"),
        ("A line of prose\n\nNext paragraph starts", "A line of prose\n\nNext paragraph starts"),
        ("   \nNext paragraph starts", "   \nNext paragraph starts"),
    ),
)
def test_normalize_sentence_boundaries_preserves_existing_readability_rules(
    text: str, expected: str
) -> None:
    assert _normalize_sentence_boundaries(text) == expected


def test_build_char_map_returns_charspan_tuple():
    blocks = [
        {"text": "abc", "source": {"page": 1, "filename": "file.pdf"}},
        {"text": "defg", "source": {"page": 2, "filename": "file.pdf"}},
    ]
    result = _build_char_map(blocks)
    positions = result["char_positions"]
    assert isinstance(positions, tuple)
    first, second = positions
    assert (first.start, first.end, first.original_index) == (0, 3, 0)
    assert (second.start, second.end, second.original_index) == (5, 9, 1)


def test_find_source_block_substring_match():
    blocks = [
        {"text": "Hello world", "source": {"page": 1, "filename": "file.pdf"}},
        {"text": "Another block", "source": {"page": 2, "filename": "file.pdf"}},
    ]
    char_map = _build_char_map(blocks)
    chunk = Document(content="Hello world continues")
    block = _find_source_block(chunk, char_map, blocks)
    assert block["source"]["page"] == 1
