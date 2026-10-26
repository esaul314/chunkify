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


def test_native_crosspage_lists_footnotes_citation_digits_and_unicode_keep_occurrences(tmp_path):
    from pdf_chunker.core_new import _run_passes

    fitz = pytest.importorskip("fitz")
    source = tmp_path / "synthetic-native-controls.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            (72, 72, 520, 220), "Introduction\n" + BODY + "\nA line ends with preser-", fontsize=11
        )
        page.insert_text((72, 310), "1. Keep the first list item.", fontsize=11)
        page.insert_text((72, 820), "1", fontsize=7)
        page.insert_text((84, 750), "An invented footnote stays on its source page.", fontsize=8)
        page = pdf.new_page()
        page.insert_font(fontname="synthetic", fontbuffer=fitz.Font("helv").buffer)
        page.insert_textbox(
            (72, 72, 520, 200),
            "vation continues without losing its source boundaries.\n"
            "2. Keep the second list item.\n"
            "A café passage cites 3682–3694 and remains short but useful.",
            fontsize=11,
            fontname="synthetic",
        )
        pdf.save(source)
    spec = structural_spec()
    initial = _input_artifact(str(source), spec)
    raw_texts = [
        initial.meta["preservation"]["texts"][unit["text"]]
        for unit in initial.meta["preservation"]["source_units"]
    ]
    assert any("3682–3694" in text for text in raw_texts)
    assert any(text.strip() == "1" for text in raw_texts)
    result, _ = _run_passes(spec, initial)
    proof = result.meta["preservation"]
    assert proof["accepted"], proof["pending"]
    joined = " ".join(row["text"] for row in result.payload)
    for marker in (
        "preser-",
        "vation",
        "3682–3694",
        "café",
        "first list",
        "second list",
        "footnote",
    ):
        assert marker in joined
    assert {unit["source_id"] for unit in proof["source_units"]} == {
        span["source_id"] for record in proof["records"] for span in record["spans"]
    }


def test_explicit_alternative_reader_is_not_silently_replaced(tmp_path, monkeypatch):
    source = original_crosspage_source(tmp_path)
    monkeypatch.setenv("PDF_CHUNKER_PDF_READER", "pdftotext")
    with pytest.raises(ValueError, match="native PDF reader"):
        _input_artifact(str(source), structural_spec())


def test_repeated_title_paragraph_and_footer_are_bound_by_source_windows(tmp_path):
    from pdf_chunker.core_new import _run_passes

    fitz = pytest.importorskip("fitz")
    source = tmp_path / "synthetic-repeated.pdf"
    repeated = "An invented repeated sentence remains a separate source occurrence. " * 8
    with fitz.open() as pdf:
        for index in range(3):
            page = pdf.new_page()
            page.insert_text((72, 72), "A Synthetic Heading", fontsize=14)
            page.insert_textbox((72, 120, 520, 350), repeated, fontsize=11)
            page.insert_text((72, 760), f"Footer marker {index + 1}", fontsize=8)
        pdf.save(source)
    spec = structural_spec()
    spec.options["split_semantic"] = {"chunk_size": 40, "overlap": 10, "min_chunk_size": 10}
    result, _ = _run_passes(spec, _input_artifact(str(source), spec))
    proof = result.meta["preservation"]
    assert proof["accepted"], proof["pending"]
    assert all("preservation_source_window" in row["metadata"] for row in result.payload)
    assert {unit["source_id"] for unit in proof["source_units"]} == {
        span["source_id"] for record in proof["records"] for span in record["spans"]
    }


def test_native_source_windows_admit_reserved_ai_metadata_before_stub_calls(tmp_path):
    import json

    from pdf_chunker.core_new import _run_passes

    class Client:
        def __init__(self):
            self.inputs = []

        def classify_chunk_utterance(self, text, *, tag_configs):
            self.inputs.append(text)
            return {"classification": "question", "tags": ["bounded-tag"]}

    fitz = pytest.importorskip("fitz")
    source = tmp_path / "synthetic-admission.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 35), "Introduction", fontsize=12)
        for index in range(24):
            page.insert_text(
                (72, 65 + index * 26),
                f"Invented paragraph {index} retains its exact occurrence.",
                fontsize=9,
            )
        pdf.save(source)
    spec = structural_spec()
    client = Client()
    spec.pipeline.insert(spec.pipeline.index("emit_jsonl"), "ai_enrich")
    spec.options["ai_enrich"] = {
        "enabled": True,
        "client": client,
        "tag_configs": {"topic": ["bounded-tag"]},
    }
    spec.options["emit_jsonl"] = {"max_chars": 1800}
    result, _ = _run_passes(spec, _input_artifact(str(source), spec))
    assert len(client.inputs) > 1
    assert result.meta["preservation"]["accepted"], result.meta["preservation"]["pending"]
    assert all(len(json.dumps(row, ensure_ascii=False)) <= 1800 for row in result.payload)
    assert all(
        "raw_source" not in entry
        for row in result.payload
        for entry in row["metadata"]["block_provenance"]
    )
    assert all(row["metadata"]["preservation_status"] == "checked" for row in result.payload)
