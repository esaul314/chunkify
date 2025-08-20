from pdf_chunker.framework import Artifact
from pdf_chunker.passes.list_detect import list_detect


def _run(blocks):
    doc = {"type": "page_blocks", "pages": [{"page": 1, "blocks": blocks}]}
    result = list_detect(Artifact(payload=doc))
    page_blocks = result.payload["pages"][0]["blocks"]
    metrics = result.meta["metrics"]["list_detect"]
    return page_blocks, metrics


def test_bullet_items_annotated_and_counted():
    blocks = [{"text": "• a"}, {"text": "• b"}, {"text": "plain"}]
    annotated, metrics = _run(blocks)
    assert [b.get("list_kind") for b in annotated] == ["bullet", "bullet", None]
    assert [b.get("type") for b in annotated][:2] == ["list_item", "list_item"]
    assert metrics == {"bullet_items": 2, "numbered_items": 0}


def test_numbered_items_annotated_and_counted():
    blocks = [{"text": "1. a."}, {"text": "2. b."}, {"text": "end"}]
    annotated, metrics = _run(blocks)
    assert [b.get("list_kind") for b in annotated] == ["numbered", "numbered", None]
    assert [b.get("type") for b in annotated][:2] == ["list_item", "list_item"]
    assert metrics == {"bullet_items": 0, "numbered_items": 2}
