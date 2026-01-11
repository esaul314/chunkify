import sys

sys.path.insert(0, ".")

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.list_detect import list_detect
from pdf_chunker.passes.split_semantic import make_splitter


def test_numbered_list_across_pages_stays_together() -> None:
    doc = {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [
            {"page": 1, "blocks": [{"text": "1. First item ends here."}]},
            {
                "page": 2,
                "blocks": [
                    {"text": "2. Second item starts next page."},
                    {"text": "3. Third item follows."},
                ],
            },
        ],
    }

    annotated = list_detect(Artifact(payload=doc))
    splitter = make_splitter(chunk_size=200, overlap=0, generate_metadata=True)
    items = splitter(annotated).payload["items"]

    assert len(items) == 1
    text = items[0]["text"]
    assert "1. First item ends here." in text
    assert "2. Second item starts next page." in text
    assert "3. Third item follows." in text


def test_numbered_list_overflow_allows_merge() -> None:
    doc = {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [
            {"page": 1, "blocks": [{"text": "1. First item ends here."}]},
            {"page": 2, "blocks": [{"text": "2. Second item starts next page."}]},
        ],
    }

    annotated = list_detect(Artifact(payload=doc))
    splitter = make_splitter(chunk_size=5, overlap=0, generate_metadata=True)
    items = splitter(annotated).payload["items"]

    assert len(items) == 1
    text = items[0]["text"]
    assert "1. First item ends here." in text
    assert "2. Second item starts next page." in text
