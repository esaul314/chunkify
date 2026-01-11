import sys

sys.path.insert(0, ".")

from pdf_chunker.pdf_blocks import Block, merge_continuation_blocks


def test_numbered_list_continuation_across_page_break() -> None:
    blocks = [
        Block(
            text="1. First item ends here.",
            source={"page": 1},
            bbox=(0.0, 700.0, 500.0, 710.0),
        ),
        Block(
            text="    Continuation on next page.",
            source={"page": 2},
            bbox=(24.0, 40.0, 520.0, 50.0),
        ),
    ]

    merged = list(merge_continuation_blocks(blocks))

    assert len(merged) == 1
    assert "1. First item ends here. Continuation on next page." in merged[0].text
