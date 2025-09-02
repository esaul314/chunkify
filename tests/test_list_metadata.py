from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_semantic import split_semantic


def _list_kind(text: str) -> str | None:
    doc = {"type": "page_blocks", "pages": [{"page": 1, "blocks": [{"text": text}]}]}
    items = split_semantic(Artifact(payload=doc)).payload["items"]
    return items[0]["meta"].get("list_kind")


def test_bullet_list_kind_inferred() -> None:
    assert _list_kind("* first item") == "bullet"


def test_numbered_list_kind_inferred() -> None:
    assert _list_kind("1. first") == "numbered"
