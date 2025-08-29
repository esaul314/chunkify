import sys

sys.path.insert(0, ".")

from pdf_chunker.splitter import semantic_chunker


def test_multiline_numbered_items() -> None:
    text = (
        "Intro text that will fill the chunk boundaries. " * 2
        + "1. First item is long and continues across boundaries. "
        + "Continuation of first item to ensure split. "
        + "2. Second item is long and continues as well. "
        + "Continuation of second item to ensure merging."
    )
    chunks = semantic_chunker(text, chunk_size=40, overlap=0)
    assert any(
        "1. First item is long" in chunk and "Continuation of first item to ensure split." in chunk
        for chunk in chunks
    )
    assert any(
        "2. Second item is long" in chunk
        and "Continuation of second item to ensure merging." in chunk
        for chunk in chunks
    )
