from pdf_chunker.framework import Artifact
from pdf_chunker.passes.detect_page_artifacts import detect_page_artifacts

TABLE_TEXT = (
    "|This closed car smells of salt fish|Col2|\n"
    "|---|---|\n"
    "|salt fish||\n"
    "|Person Name, PMP<br>Alma, Quebec, Canada|Person Name, PMP<br>Alma, Quebec, Canada|"
)

EXPECTED = "This closed car smells of salt fish"


def _doc() -> dict:
    return {
        "type": "page_blocks",
        "pages": [{"page": 1, "blocks": [{"text": TABLE_TEXT}]}],
    }


def test_markdown_table_flattened() -> None:
    result = detect_page_artifacts(Artifact(payload=_doc()))
    blocks = result.payload["pages"][0]["blocks"]
    assert blocks[0]["text"] == EXPECTED
    assert "Person Name" not in blocks[0]["text"]
    assert (
        result.meta["metrics"]["detect_page_artifacts"]["blocks_cleaned"] == 1
    )
