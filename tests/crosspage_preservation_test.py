"""Original generated crosspage failure: physical source identity precedes roles."""

import hashlib

import pytest

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _input_artifact

BODY = "The following page contains substantive explanatory text. " * 8
CONTENTS = "Contents\nChapter One  1\nChapter Two  2\nChapter Three  3"


def original_crosspage_source(tmp_path, first=CONTENTS):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "synthetic-crosspage.pdf"
    with fitz.open() as document:
        for text in (first, "Introduction\n" + BODY):
            page = document.new_page()
            page.insert_textbox((72, 72, 523, 770), text, fontsize=11)
        document.save(path)
    return path


@pytest.mark.parametrize(
    "first", [CONTENTS, "Acknowledgments\nWe thank the reviewers and remain grateful."]
)
def test_structure_ingress_keeps_physical_pages_before_classification(tmp_path, first):
    path = original_crosspage_source(tmp_path, first)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = _input_artifact(
        str(path),
        PipelineSpec(options={"section_detect": {"enabled": True}}),
    )
    pages = artifact.payload["pages"]
    assert pages[1]["blocks"], "following-page body must not live in a preceding section"
    assert "substantive explanatory text" not in "\n".join(b["text"] for b in pages[0]["blocks"])
    assert "substantive explanatory text" in "\n".join(b["text"] for b in pages[1]["blocks"])
    assert all(b["source"]["page"] == p["page"] for p in pages for b in p["blocks"])
    assert all(b["source"]["raw_source"] for p in pages for b in p["blocks"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def structural_spec(**extra):
    return PipelineSpec(
        pipeline=[
            "pdf_parse",
            "text_clean",
            "heading_detect",
            "section_detect",
            "split_semantic",
            "emit_jsonl",
        ],
        options={
            "heading_detect": {"enabled": True},
            "section_detect": {"enabled": True, "recover_structural_text": True, **extra},
            "split_semantic": {"chunk_size": 400, "overlap": 100},
        },
    )


def test_annotation_and_isolated_digit_are_observable_through_library(tmp_path):
    import json

    from pdf_chunker.core_new import convert

    fitz = pytest.importorskip("fitz")
    source = tmp_path / "synthetic-note.pdf"
    note = "An invented margin note must remain inspectable."
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Introduction\n" + BODY, fontsize=10)
        page.insert_text((72, 820), "1", fontsize=8)
        page.add_text_annot((35, 120), note)
        pdf.save(source)
    spec = structural_spec()
    with pytest.raises(ValueError, match="preservation_path"):
        convert(str(source), spec)
    ledger = tmp_path / "private-preservation.json"
    rows = convert(str(source), structural_spec(preservation_path=str(ledger)))
    proof = json.loads(ledger.read_text())
    assert not proof["accepted"]
    assert any(value.get("annotation", {}).get("text") == note for value in proof["pending"])
    assert any(
        value["text"] in proof["texts"] and proof["texts"][value["text"]].strip() == "1"
        for value in proof["source_units"]
    )
    assert any("1" in output["text"] for output in rows)
    assert all(output["metadata"]["preservation_status"] == "pending" for output in rows)
    assert ledger.stat().st_mode & 0o777 == 0o600


def test_terminal_runner_derives_bound_private_ledger_from_existing_output(tmp_path):
    import json

    from pdf_chunker.core_new import run_convert

    source = original_crosspage_source(tmp_path)
    output = tmp_path / "out.jsonl"
    spec = structural_spec()
    spec.options["emit_jsonl"] = {"output_path": str(output)}
    spec.options["run_report"] = {"output_path": str(tmp_path / "report.json")}
    result, _ = run_convert(_input_artifact(str(source), spec), spec)
    ledger = tmp_path / "out.jsonl.preservation.json"
    assert json.loads(ledger.read_text())["accepted"]
    assert result.meta["preservation_path"] == str(ledger)
    assert output.exists()
    assert all(row["metadata"]["preservation_status"] == "checked" for row in result.payload)


def test_multiblock_contents_recovery_retains_all_physical_identities(tmp_path):
    from pdf_chunker.core_new import _run_passes

    fitz = pytest.importorskip("fitz")
    source = tmp_path / "synthetic-multiblock.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        for index, text in enumerate(
            ["Contents", "Chapter One  1", "Chapter Two  2", "Chapter Three  3"]
        ):
            page.insert_text((72, 72 + index * 60), text, fontsize=11)
        pdf.save(source)
    spec = structural_spec()
    initial = _input_artifact(str(source), spec)
    raw_ids = {unit["source_id"] for unit in initial.meta["preservation"]["source_units"]}
    assert len(raw_ids) == 4
    result, _ = _run_passes(spec, initial)
    proof = result.meta["preservation"]
    assert proof["accepted"], proof["pending"]
    assert len(proof["block_identities"]) == 4
    emitted_ids = {span["source_id"] for record in proof["records"] for span in record["spans"]}
    assert emitted_ids == raw_ids
    assert all(row["metadata"]["section_role"] == "contents" for row in result.payload)
