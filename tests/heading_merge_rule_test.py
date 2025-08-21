from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_semantic import split_semantic


def _doc(blocks):
    return {
        "type": "page_blocks",
        "source_path": "mem.pdf",
        "pages": [{"page": 1, "blocks": blocks}],
    }


def _run(blocks):
    artifact = Artifact(payload=_doc(blocks), meta={})
    return split_semantic(artifact).payload["items"]


def test_heading_followed_by_paragraph():
    blocks = [
        {"text": "Heading", "type": "heading"},
        {"text": "Body text", "type": "paragraph"},
    ]
    chunks = _run(blocks)
    assert [c["text"] for c in chunks] == ["Heading\nBody text"]


def test_heading_followed_by_list_item():
    blocks = [
        {"text": "Intro", "type": "heading"},
        {"text": "Bullet", "type": "list_item", "list_kind": "bullet"},
    ]
    chunks = _run(blocks)
    assert [c["text"] for c in chunks] == ["Intro\nBullet"]
