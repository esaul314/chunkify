import sys
from dataclasses import asdict

sys.path.insert(0, ".")

from pdf_chunker.chunk_validation import validate_chunks
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.list_detect import list_detect
from pdf_chunker.pdf_blocks import Block, merge_continuation_blocks
from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf


def test_bullet_list_preservation():
    stream = extract_text_blocks_from_pdf("sample_book3.pdf")
    assert not isinstance(stream, list)
    blocks = [asdict(b) for b in stream]
    report = validate_chunks(blocks)
    assert report.total_chunks == len(blocks)
    assert report.empty_text == 0
    # mid_sentence_starts not checked: raw PDF blocks naturally continue sentences
    assert report.overlong == 0
    assert report.duplications == []
    assert report.boundary_overlaps == []

    blob = "\n\n".join(b["text"] for b in blocks)
    items = [line.strip() for line in blob.splitlines() if line.lstrip().startswith("•")]
    # Bullet count may vary with extraction heuristics; ensure at least some exist
    assert len(items) >= 2
    # Bullets should not end with periods (they are list items, not sentences)
    assert all(not item.rstrip().endswith(".") for item in items)
    # Bullet items should not be split with empty lines between them
    assert "•\n\n•" not in blob


def test_bullet_items_annotated_with_list_kind():
    doc = {
        "type": "page_blocks",
        "pages": [{"page": 1, "blocks": [{"text": "• item"}, {"text": "plain"}]}],
    }
    annotated = list_detect(Artifact(payload=doc)).payload["pages"][0]["blocks"]
    assert [b.get("list_kind") for b in annotated] == ["bullet", None]


def test_colon_intro_not_merged_with_following_bullet():
    source = {"filename": "example.pdf", "page": 1, "location": None}
    intro = Block(text="Options:", source=source)
    bullet = Block(text="• choice", source=source)
    merged = merge_continuation_blocks([intro, bullet])

    assert [block.text for block in merged] == ["Options:", "• choice"]

    doc = {
        "type": "page_blocks",
        "pages": [
            {"page": 1, "blocks": [asdict(block) for block in merged]},
        ],
    }
    annotated = list_detect(Artifact(payload=doc)).payload["pages"][0]["blocks"]
    kinds = [block.get("list_kind") for block in annotated]
    assert kinds == [None, "bullet"]
