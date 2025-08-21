import sys

sys.path.insert(0, ".")

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.heading_detect import heading_detect
from pdf_chunker.passes.list_detect import list_detect
from pdf_chunker.passes.split_semantic import split_semantic


def test_metadata_propagation() -> None:
    doc = {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [
            {"page": 1, "blocks": [{"text": "1. first"}]},
            {"page": 2, "blocks": [{"text": "Chapter One"}]},
        ],
    }
    doc["pages"] = [
        {**p, "blocks": heading_detect(Artifact(payload=p["blocks"])).payload} for p in doc["pages"]
    ]
    chunks = split_semantic(list_detect(Artifact(payload=doc))).payload["items"]

    list_meta = chunks[0]["meta"]
    assert list_meta["list_kind"] == "numbered"
    assert list_meta["page"] == 1
    assert list_meta["source"] == "src.pdf"

    heading_meta = chunks[1]["meta"]
    assert heading_meta["is_heading"] is True
    assert heading_meta["heading_level"] == 1
    assert heading_meta["page"] == 2
    assert heading_meta["source"] == "src.pdf"
