from pdf_chunker.framework import Artifact
from pdf_chunker.passes.text_clean import text_clean


def _build_doc() -> dict:
    return {
        "type": "page_blocks",
        "pages": [
            {
                "page": 1,
                "blocks": [
                    {"text": "Foo\u00a0bar"},
                    {"text": "One-\nline"},
                ],
            }
        ],
    }


def test_text_clean_idempotent() -> None:
    doc = _build_doc()
    first = text_clean(Artifact(payload=doc))
    second = text_clean(first)
    assert second.payload == first.payload
    assert second.meta == first.meta


def test_text_clean_normalizes_blocks() -> None:
    doc = _build_doc()
    result = text_clean(Artifact(payload=doc))
    blocks = [b["text"] for b in result.payload["pages"][0]["blocks"]]
    assert blocks == ["Foo bar", "Oneline"]
    metrics = result.meta["metrics"]
    assert metrics["normalized"] is True
    assert metrics["text_clean"]["blocks"] == 2
