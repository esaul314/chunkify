import sys

sys.path.insert(0, ".")

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf
from pdf_chunker.chunk_validation import validate_chunks
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.list_detect import list_detect


def test_bullet_list_preservation():
    blocks = extract_text_blocks_from_pdf("sample_book3.pdf")
    report = validate_chunks(blocks)
    assert report.total_chunks == len(blocks)
    assert report.empty_text == 0
    assert report.mid_sentence_starts == 1
    assert report.overlong == 0
    assert report.duplications == []
    assert report.boundary_overlaps == []

    blob = "\n\n".join(b["text"] for b in blocks)
    items = [
        line.strip() for line in blob.splitlines() if line.lstrip().startswith("•")
    ]
    assert len(items) == 3
    assert all(not item.rstrip().endswith(".") for item in items)
    assert "•\n\n•" not in blob
    assert "\n\nswamp" not in blob
    assert "swamp\n\nFollow" in blob


def test_bullet_items_annotated_with_list_kind():
    doc = {
        "type": "page_blocks",
        "pages": [{"page": 1, "blocks": [{"text": "• item"}, {"text": "plain"}]}],
    }
    annotated = list_detect(Artifact(payload=doc)).payload["pages"][0]["blocks"]
    assert [b.get("list_kind") for b in annotated] == ["bullet", None]
