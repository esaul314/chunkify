"""Ordered occurrence accounting rejects loss, reorder and invented digit joins."""

from copy import deepcopy

import pytest

from pdf_chunker.preservation import (
    finalize_rows,
    identity,
    initialize,
    normalize_source_text,
    record_stage,
    sha,
)


def bound_document(texts):
    raw_blocks, blocks = [], []
    for index, text in enumerate(texts):
        bbox = [72, 72 + index * 50, 500, 110 + index * 50]
        raw = {
            "page": 1,
            "block_index": index,
            "bbox": bbox,
            "text_sha256": sha(text),
            "span": [0, len(text)],
        }
        raw_blocks.append({"index": index, "bbox": bbox, "text": text, "raw_text": text})
        blocks.append(
            {
                "text": text,
                "type": "paragraph",
                "bbox": bbox,
                "source": {"page": 1, "raw_source": raw},
                "section": {"section_id": "body", "section_role": "body"},
            }
        )
    return {
        "type": "page_blocks",
        "pages": [{"page": 1, "blocks": blocks}],
        "source_structure": {
            "source_sha256": "a" * 64,
            "pipeline_settings_sha256": "b" * 64,
            "pages": [{"page": 1, "height": 792, "blocks": raw_blocks}],
        },
    }


def row(text, *, part=None):
    metadata = {"chunk_id": "synthetic", "section_id": "body", "preservation_record_id": "0" * 24}
    if part is not None:
        metadata["chunk_part"] = part
    return {"text": text, "metadata": metadata}


def report_for(texts):
    doc = bound_document(texts)
    return record_stage(initialize(doc, {}), "section_detect", doc), doc


@pytest.mark.parametrize(
    "texts,output",
    [
        (
            ["A repeated useful paragraph.", "A repeated useful paragraph."],
            ["A repeated useful paragraph."],
        ),
        (
            ["First useful paragraph.", "Second useful paragraph."],
            ["Second useful paragraph.", "First useful paragraph."],
        ),
        (["One two three four LAST."], ["One two three", "two three four"]),
        (["Citation 3682–3694 stays exact."], ["Citation 368 2–3694 stays exact."]),
    ],
)
def test_missing_repeated_reordered_tail_and_digit_damage_refuse(texts, output):
    report, _ = report_for(texts)
    result, _ = finalize_rows(report, [row(text) for text in output])
    assert not result["accepted"]
    assert result["pending"]


def test_identical_original_occurrences_are_counted_separately():
    text = "An invented repeated paragraph."
    report, _ = report_for([text, text])
    result, rows = finalize_rows(report, [row(text + "\n\n" + text)])
    assert result["accepted"]
    assert len({span["source_id"] for span in result["records"][0]["spans"]}) == 2
    assert rows[0]["metadata"]["preservation_record_id"] == result["records"][0]["record_id"]


@pytest.mark.parametrize("mutation", ["identity", "duplicate", "reorder"])
def test_changed_or_repeated_source_units_refuse(mutation):
    report, doc = report_for(["First.", "Second."])
    changed = deepcopy(doc)
    blocks = changed["pages"][0]["blocks"]
    if mutation == "identity":
        blocks[0]["source"]["raw_source"]["span"] = [0, 2]
    elif mutation == "duplicate":
        blocks.append(deepcopy(blocks[0]))
    else:
        blocks.reverse()
    result = record_stage(report, "cleanup", changed)
    assert not result["accepted"]


def test_letter_hyphens_ligatures_and_unicode_have_explicit_witnesses():
    original = "An ofﬁce keeps preser-\nvation notes and café."
    report, doc = report_for([original])
    changed = deepcopy(doc)
    changed["pages"][0]["blocks"][0]["text"] = normalize_source_text(original)
    report = record_stage(report, "text_clean", changed)
    assert report["accepted"]
    witness = report["stages"][-1]["transformations"][0]
    assert witness["source_span"] == [0, len(original)]
    assert {"unicode_nfkc", "letter_only_line_hyphenation"} <= set(witness["operations"])
    result, _ = finalize_rows(report, [row(normalize_source_text(original))])
    assert result["accepted"]


def test_serialized_siblings_can_split_one_word_with_exact_source_spans():
    report, _ = report_for(["preservation"])
    result, rows = finalize_rows(report, [row("preser", part=0), row("vation", part=1)])
    assert result["accepted"]
    assert result["records"][0]["spans"][0]["source_span"] == [0, 6]
    assert result["records"][1]["spans"][0]["source_span"] == [6, 12]
    assert "".join(item["text"] for item in rows) == "preservation"


def test_annotation_surface_is_retained_as_an_unresolved_gap():
    doc = bound_document(["Printed body."])
    annotation = {"xref": 12, "bbox": [20, 20, 30, 30], "text": "An invented margin note."}
    doc["source_structure"]["pages"][0]["annotations"] = [annotation]
    result = initialize(doc, {})
    assert not result["accepted"]
    assert result["pending"][0]["annotation"] == annotation


def test_unit_limits_refuse_instead_of_running_unbounded_alignment(monkeypatch):
    import pdf_chunker.preservation as module

    monkeypatch.setattr(module, "MAX_UNIT_CHARACTERS", 5)
    with pytest.raises(ValueError, match="unit"):
        report_for(["Too many source characters."])


def test_source_hashes_and_record_ids_bind_the_exact_occurrence():
    report, _ = report_for(["A passage."])
    result, _ = finalize_rows(report, [row("A passage.")])
    record = result["records"][0]
    assert (
        identity({key: value for key, value in record.items() if key != "record_id"})[:24]
        == record["record_id"]
    )
    assert all(sha(text) == key for key, text in result["texts"].items())


@pytest.mark.parametrize("overlap", [0, 50])
def test_extra_output_record_is_not_approved_as_context(overlap):
    doc = bound_document(["A complete source paragraph."])
    report = initialize(doc, {"split_semantic": {"overlap": overlap}})
    result, _ = finalize_rows(
        report, [row("A complete source paragraph."), row("A complete source paragraph.")]
    )
    assert not result["accepted"]


def test_whole_missing_page_is_not_inferred_to_be_excluded():
    doc = bound_document(["First page."])
    second = bound_document(["Second page."])
    second_raw = second["source_structure"]["pages"][0]
    second_raw["page"] = 2
    doc["source_structure"]["pages"].append(second_raw)
    result = initialize(doc, {})
    assert not result["accepted"]
    assert not any(value["reason"] == "explicit_page_exclusion" for value in result["exclusions"])


@pytest.mark.parametrize("where", ["container", "source"])
def test_page_citation_must_match_exact_raw_identity(where):
    report, doc = report_for(["A page-bound passage."])
    if where == "container":
        doc["pages"][0]["page"] = 99
    else:
        doc["pages"][0]["blocks"][0]["source"]["page"] = 99
    assert not record_stage(report, "cleanup", doc)["accepted"]


def test_serialized_wrong_source_and_page_refuse():
    report, _ = report_for(["A source-bound passage."])
    output = row("A source-bound passage.")
    output["metadata"].update(source_file="wrong.pdf", page=99, page_range="99")
    assert not finalize_rows(report, [output])[0]["accepted"]


def test_combining_mark_composition_keeps_its_full_raw_span():
    report, _ = report_for(["Cafe\u0301"])
    result, _ = finalize_rows(report, [row("Café")])
    assert result["accepted"]
    assert result["records"][0]["spans"][0]["source_span"] == [0, 5]


def test_distinct_sections_cannot_be_reordered():
    report, doc = report_for(["First section.", "Second section."])
    doc["pages"][0]["blocks"][0]["section"]["section_id"] = "s1"
    doc["pages"][0]["blocks"][1]["section"]["section_id"] = "s2"
    report = record_stage(report, "section_detect", doc)
    outputs = [row("Second section."), row("First section.")]
    outputs[0]["metadata"]["section_id"] = "s2"
    outputs[1]["metadata"]["section_id"] = "s1"
    assert not finalize_rows(report, outputs)[0]["accepted"]


def test_parent_page_range_covers_union_and_siblings_keep_exact_spans():
    doc = bound_document(["First page."])
    second = bound_document(["Second page."])
    page = second["pages"][0]
    page["page"] = 2
    page["blocks"][0]["source"]["page"] = 2
    page["blocks"][0]["source"]["raw_source"]["page"] = 2
    doc["pages"].append(page)
    raw = second["source_structure"]["pages"][0]
    raw["page"] = 2
    doc["source_structure"]["pages"].append(raw)
    report = record_stage(initialize(doc, {}), "section_detect", doc)
    rows = [row("First page.", part=0), row("\nSecond page.", part=1)]
    for output in rows:
        output["metadata"].update(page=1, page_range="1-2")
    accepted, _ = finalize_rows(report, rows)
    assert accepted["accepted"]
    assert [{span["source_id"] for span in record["spans"]} for record in accepted["records"]] == [
        {unit["source_id"]} for unit in report["source_units"]
    ]
    forged = row("First page.\nSecond page.")
    forged["metadata"].update(page=1, page_range="1")
    assert not finalize_rows(report, [forged])[0]["accepted"]


def synthetic_classified_chunks(count=24):
    from pdf_chunker.preservation import bind_source_metadata
    from pdf_chunker.sections import classify_document

    doc = bound_document([f"Invented occurrence {n} remains in order." for n in range(count)])
    report = initialize(doc, {})
    classified, _, _ = classify_document(doc, settings={})
    report = record_stage(report, "section_detect", classified)
    blocks = classified["pages"][0]["blocks"]
    payload = {
        "type": "chunks",
        "items": [
            {
                "text": "\n".join(block["text"] for block in blocks),
                "meta": {
                    "chunk_id": "synthetic",
                    "section_id": blocks[0]["section"]["section_id"],
                    "section_role": blocks[0]["section"]["section_role"],
                    "block_provenance": [
                        entry for block in blocks for entry in block["source"]["block_provenance"]
                    ],
                },
            }
        ],
    }
    payload = bind_source_metadata(report, payload)
    report = record_stage(report, "split_semantic", payload)
    assert report["accepted"]
    return payload["items"], report


def test_generated_metadata_and_reserved_enrichment_partition_together():
    import json

    from pdf_chunker.passes.ai_enrich import _admit_chunks_for_jsonl
    from pdf_chunker.passes.emit_jsonl import _canonical_rows

    chunks, report = synthetic_classified_chunks()
    source = chunks[0]["text"]
    limit = 1800
    parts = _admit_chunks_for_jsonl(
        chunks,
        source_preservation=report,
        max_chars=limit,
        metadata_key="metadata",
        tag_configs={"topic": ["bounded-tag"]},
    )
    assert len(parts) > 1
    assert "".join(part["text"] for part in parts) == source
    assert all(len(part["meta"]["block_provenance"]) < 24 for part in parts)
    for part in parts:
        part["meta"].update(
            tags=["bounded-tag"], utterance_type="statement_of_fact", enrichment_status="complete"
        )
        assert (
            len(json.dumps({"text": part["text"], "metadata": part["meta"]}, ensure_ascii=False))
            <= limit
        )
    accepted, rows = finalize_rows(
        report,
        _canonical_rows({"type": "chunks", "items": parts}, preserve_text=True),
        max_chars=limit,
    )
    assert accepted["accepted"], accepted["pending"]
    assert "".join(output["text"] for output in rows) == source
    assert all(
        "raw_source" not in entry
        for output in rows
        for entry in output["metadata"]["block_provenance"]
    )
    changed = deepcopy(chunks)
    changed[0]["meta"]["arbitrary"] = "x" * 5000
    with pytest.raises(ValueError, match="metadata exceeds"):
        _admit_chunks_for_jsonl(
            changed,
            source_preservation=report,
            max_chars=limit,
            metadata_key="metadata",
            tag_configs={},
        )


def test_actual_stub_enrichment_uses_source_partition_and_same_limits():
    from pdf_chunker.config import PipelineSpec
    from pdf_chunker.core_new import _run_passes
    from pdf_chunker.framework import Artifact

    class Client:
        def __init__(self):
            self.inputs = []

        def classify_chunk_utterance(self, text, *, tag_configs):
            self.inputs.append(text)
            return {"classification": "question", "tags": ["bounded-tag"]}

    chunks, report = synthetic_classified_chunks()
    client = Client()
    result, _ = _run_passes(
        PipelineSpec(
            pipeline=["ai_enrich", "emit_jsonl"],
            options={
                "ai_enrich": {
                    "enabled": True,
                    "client": client,
                    "tag_configs": {"topic": ["bounded-tag"]},
                },
                "emit_jsonl": {"max_chars": 1800},
            },
        ),
        Artifact(
            payload={"type": "chunks", "items": chunks},
            meta={"preservation": report, "semantic_pipeline": "canonical-v1"},
        ),
    )
    assert len(client.inputs) > 1
    assert "".join(client.inputs) == chunks[0]["text"]
    assert result.meta["preservation"]["accepted"]
    assert all(output["metadata"]["preservation_status"] == "checked" for output in result.payload)


def test_section_parent_span_cannot_claim_unrelated_offsets():
    report, doc = report_for(["An intact source passage."])
    block = doc["pages"][0]["blocks"][0]
    block["_section_parent_sha256"] = sha(block["text"])
    block["_section_span"] = [0, 1]
    assert not record_stage(report, "section_detect", doc)["accepted"]


def test_boolean_page_is_not_a_truthful_integer_citation():
    report, _ = report_for(["One page."])
    output = row("One page.")
    output["metadata"]["page"] = True
    assert not finalize_rows(report, [output])[0]["accepted"]
