from pdf_chunker.framework import Artifact
from pdf_chunker.passes.heading_detect import heading_detect


def test_heading_pass_hierarchy_and_metrics() -> None:
    blocks = [
        {"text": "Chapter 1"},
        {"text": "Section 1 Overview"},
        {"text": "Some paragraph text."},
    ]
    result = heading_detect(Artifact(payload=blocks, meta={}))

    headings = [b for b in result.payload if b["is_heading"]]
    assert [h["heading_level"] for h in headings] == [1, 2]
    assert [h["heading_threshold"] for h in headings] == [3, 3]

    hierarchy = result.meta["heading_hierarchy"]
    assert [h["text"] for h in hierarchy] == ["Chapter 1", "Section 1 Overview"]
    assert hierarchy[1]["parent"] == "Chapter 1"

    metrics = result.meta["metrics"]["heading_detect"]
    assert metrics["headings"] == 2
