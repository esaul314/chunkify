from pdf_chunker.framework import Artifact
from pdf_chunker.passes.text_clean import text_clean
from pdf_chunker.passes.detect_page_artifacts import detect_page_artifacts


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
    first, second = (block["text"] for block in result.payload["pages"][0]["blocks"])
    assert first == "Foo bar"
    assert second == "Oneline"
    metrics = result.meta["metrics"]
    assert metrics["normalized"] is True
    assert metrics["text_clean"]["blocks"] == 2


def test_text_clean_preserves_table_header_removal() -> None:
    table_text = (
        "|This closed car smells of salt fish|Col2|\n"
        "|---|---|\n"
        "|salt fish||\n"
        "|Person Name, PMP<br>Alma, Quebec, Canada|Person Name, PMP<br>Alma, Quebec, Canada|"
    )
    doc = {
        "type": "page_blocks",
        "pages": [
            {"page": 1, "blocks": [{"text": table_text}]},
        ],
    }
    artifact = Artifact(payload=doc)
    flattened = detect_page_artifacts(artifact)
    result = text_clean(flattened)
    cleaned_block = result.payload["pages"][0]["blocks"][0]["text"]
    assert cleaned_block == "This closed car smells of salt fish"
    assert "Person Name" not in cleaned_block
