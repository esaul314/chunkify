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
            {"page": 1, "blocks": [{"text": "- bullet"}]},
            {"page": 2, "blocks": [{"text": "1. first"}]},
        ],
    }
    doc["pages"] = [
        {**p, "blocks": heading_detect(Artifact(payload=p["blocks"])).payload} for p in doc["pages"]
    ]
    chunks = split_semantic(list_detect(Artifact(payload=doc))).payload["items"]
    bullet_meta = chunks[0]["meta"]
    assert bullet_meta["list_kind"] == "bullet"
    assert bullet_meta["page"] == 1
    assert bullet_meta["source"] == "src.pdf"

    numbered_meta = chunks[1]["meta"]
    assert numbered_meta["list_kind"] == "numbered"
    assert numbered_meta["page"] == 2
    assert numbered_meta["source"] == "src.pdf"
