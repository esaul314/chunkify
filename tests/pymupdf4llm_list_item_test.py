import sys

sys.path.insert(0, ".")

from pdf_chunker.pymupdf4llm_integration import (
    _convert_markdown_to_blocks,
    _clean_pymupdf4llm_block,
)


def test_list_item_metadata_and_cleanup():
    markdown = """# Title\n- Bullet item\n1. Numbered item\nPlain text"""
    blocks = _convert_markdown_to_blocks(markdown, "dummy.pdf")
    assert [(b["type"], b.get("list_kind")) for b in blocks] == [
        ("heading", None),
        ("list_item", "bullet"),
        ("list_item", "numbered"),
        ("paragraph", None),
    ]
    assert blocks[1]["text"].startswith("-")
    assert blocks[2]["text"].startswith("1.")

    cleaned = [_clean_pymupdf4llm_block(b) for b in blocks]
    texts = [b["text"] for b in cleaned if b]
    assert texts[1] == "Bullet item"
    assert texts[2] == "Numbered item"
