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
        "document_metadata": {
            "doc_id": "episode-001",
            "source_kind": "podcast_episode",
            "source_system": "podcast_addict",
            "podcast_name": "Signal",
            "episode_name": "Episode One",
        },
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
    assert bullet_meta["block_type"] == "list_item"
    assert bullet_meta["language"] == "un"
    assert bullet_meta["location"] is None
    assert bullet_meta["doc_id"] == "episode-001"
    assert bullet_meta["source_kind"] == "podcast_episode"
    assert bullet_meta["source_system"] == "podcast_addict"
    assert bullet_meta["podcast_name"] == "Signal"
    assert bullet_meta["episode_name"] == "Episode One"

    numbered_meta = chunks[1]["meta"]
    assert numbered_meta["list_kind"] == "numbered"
    assert numbered_meta["page"] == 2
    assert numbered_meta["source"] == "src.pdf"
    assert numbered_meta["block_type"] == "list_item"
    assert numbered_meta["language"] == "un"
    assert numbered_meta["location"] is None
    assert numbered_meta["doc_id"] == "episode-001"
    assert numbered_meta["source_kind"] == "podcast_episode"
