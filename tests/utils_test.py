from haystack.dataclasses import Document

from pdf_chunker.utils import _build_char_map, _find_source_block


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
