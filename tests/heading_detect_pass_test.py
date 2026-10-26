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


def test_canonical_pages_are_annotated_without_losing_source_or_mutating_input() -> None:
    from copy import deepcopy

    source = {"filename": "pilot.pdf", "page": 12, "index": 4}
    doc = {
        "type": "page_blocks",
        "source_path": "/library/pilot.pdf",
        "pages": [
            {
                "page": 12,
                "custom": 7,
                "blocks": [
                    {"text": "SYNTHETIC HEADING FOR TESTS", "source": source},
                    {"text": "Ordinary explanatory prose follows this heading.", "source": source},
                ],
            }
        ],
    }
    artifact = Artifact(
        payload=doc,
        meta={
            "options": {"heading_detect": {"enabled": True}},
            "metrics": {"heading_detect": {"custom": 9}},
        },
    )
    before = deepcopy(artifact)

    result = heading_detect(artifact)

    assert artifact == before
    assert result.payload["source_path"] == doc["source_path"]
    assert result.payload["pages"][0]["custom"] == 7
    first, second = result.payload["pages"][0]["blocks"]
    assert first["type"] == "heading"
    assert first["source"] == source
    assert not second["is_heading"]
    assert result.meta["metrics"]["heading_detect"] == {"custom": 9, "headings": 1}
    assert result.meta["heading_hierarchy"][0]["text"] == first["text"]


def test_heading_and_body_block_provenance_survives_emission() -> None:
    from pdf_chunker.config import PipelineSpec
    from pdf_chunker.core_new import _run_passes

    blocks = [
        {
            "text": "SAMPLE HEADING",
            "source": {"filename": "pilot.pdf", "page": 3},
            "bbox": (1, 2, 3, 4),
        },
        {
            "text": "This is explanatory prose about normal disagreement in groups.",
            "source": {"filename": "pilot.pdf", "page": 3},
            "bbox": (5, 6, 7, 8),
        },
    ]
    for block in blocks:
        import hashlib

        block["source"]["block_provenance"] = [
            {
                "page": 3,
                "bbox": block["bbox"],
                "text_sha256": hashlib.sha256(block["text"].encode()).hexdigest(),
            }
        ]
    artifact = Artifact(
        {
            "type": "page_blocks",
            "source_path": "pilot.pdf",
            "pages": [{"page": 3, "blocks": blocks}],
        },
        {},
    )
    spec = PipelineSpec(
        pipeline=["heading_detect", "split_semantic", "emit_jsonl"],
        options={"heading_detect": {"enabled": True}},
    )
    result, _ = _run_passes(spec, artifact)
    provenance = result.payload[0]["metadata"]["block_provenance"]
    assert {(x["page"], tuple(x["bbox"])) for x in provenance} == {
        (3, (1, 2, 3, 4)),
        (3, (5, 6, 7, 8)),
    }
    assert all(len(x["text_sha256"]) == 64 for x in provenance)


def test_sentence_continuation_keeps_all_supplied_block_origins() -> None:
    from pdf_chunker.config import PipelineSpec
    from pdf_chunker.core_new import _run_passes

    blocks = [
        {
            "text": text,
            "type": "paragraph",
            "source": {
                "filename": "pilot.pdf",
                "page": 3,
                "block_provenance": [
                    {"page": 3, "bbox": (index, 2, 3, 4), "text_sha256": str(index) * 64}
                ],
            },
        }
        for index, text in enumerate(
            ("This source starts", "with a continuation", "and ends here.")
        )
    ]
    artifact = Artifact(
        {
            "type": "page_blocks",
            "source_path": "pilot.pdf",
            "pages": [{"page": 3, "blocks": blocks}],
        },
        {},
    )
    result, _ = _run_passes(PipelineSpec(pipeline=["split_semantic", "emit_jsonl"]), artifact)
    assert len(result.payload[0]["metadata"]["block_provenance"]) == 3
