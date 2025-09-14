import sys

sys.path.insert(0, ".")

from pdf_chunker.splitter import semantic_chunker  # noqa: E402
from pdf_chunker.framework import Artifact  # noqa: E402
from pdf_chunker.passes.list_detect import list_detect  # noqa: E402
from pdf_chunker.passes.split_semantic import split_semantic  # noqa: E402


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


def test_numbered_list_merge_collapses_blank_lines():
    text = "1. first\n\n2. second"
    assert semantic_chunker(text, chunk_size=3, overlap=0) == ["1. first\n2. second"]


def test_list_kind_propagates_to_chunk_metadata():
    doc = {
        "type": "page_blocks",
        "pages": [{"page": 1, "blocks": [{"text": "1. first"}]}],
    }
    items = split_semantic(list_detect(Artifact(payload=doc))).payload["items"]
    assert items[0]["meta"]["list_kind"] == "numbered"
