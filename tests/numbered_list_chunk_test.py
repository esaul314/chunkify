import sys

sys.path.insert(0, ".")

from pdf_chunker.splitter import semantic_chunker


def test_numbered_list_not_split_across_chunks():
    text = (
        "Paragraph begins. Then the numbered list: "
        "1. The first list item is here. "
        "2. The second list item is here. "
        "3. The third list item is here. "
        "4. The fourth list item is here."
    )
    chunks = semantic_chunker(text, chunk_size=20, overlap=0)
    assert len(chunks) == 1
    chunk = chunks[0]
    for n in range(1, 5):
        assert str(n) in chunk
