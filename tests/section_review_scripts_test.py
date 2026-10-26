"""Portable review helpers use only generated, synthetic documents."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_source(tmp_path):
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "synthetic-sections.pdf"
    with fitz.open() as pdf:
        for text in ("Contents\nChapter One  1\nChapter Two  2\nChapter Three  3",):
            page = pdf.new_page()
            page.insert_textbox((72, 72, 523, 770), text, fontsize=11)
        pdf.save(source)
    return source


def manifest_for(source, tmp_path, digest=None):
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "example": {
                    "source": str(source),
                    "source_sha256": digest or hashlib.sha256(source.read_bytes()).hexdigest(),
                    "artifacts": [],
                }
            }
        )
    )
    return manifest


def test_cpu_helpers_preserve_synthetic_source_and_report_all_pages(tmp_path):
    source = synthetic_source(tmp_path)
    original = source.read_bytes()
    manifest = manifest_for(source, tmp_path)
    output = tmp_path / "local-section-reviews" / "run"
    helper("prepare_section_pilots").prepare(manifest, output)
    result = helper("verify_section_pilots").verify(output)

    assert result["accepted"]
    assert source.read_bytes() == original
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir() if p.is_file())
    receipt = json.loads((output / "cpu-acceptance.json").read_text())
    assert {section["page"] for section in receipt["books"]["example"]["sections"]} == {1}
    assert receipt["books"]["example"]["enrichment_status"] == "pending"


def test_cpu_preparation_refuses_changed_source_before_extraction(tmp_path):
    source = tmp_path / "synthetic-sections.pdf"
    source.write_bytes(b"synthetic input; extraction must never run")
    manifest = manifest_for(source, tmp_path, digest="0" * 64)
    with pytest.raises(AssertionError, match="source drift"):
        helper("prepare_section_pilots").prepare(manifest, tmp_path / "failed-run")


def test_independent_verification_refuses_changed_output(tmp_path):
    source = synthetic_source(tmp_path)
    manifest = manifest_for(source, tmp_path)
    output = tmp_path / "local-section-reviews" / "run"
    helper("prepare_section_pilots").prepare(manifest, output)
    artifact = output / "example-cpu.jsonl"
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(AssertionError):
        helper("verify_section_pilots").verify(output)


def test_original_generated_crosspage_source_preserves_physical_body(tmp_path):
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "synthetic-crosspage.pdf"
    body = "The following page contains substantive explanatory text. " * 8
    with fitz.open() as pdf:
        for text in (
            "Contents\nChapter One  1\nChapter Two  2\nChapter Three  3",
            "Introduction\n" + body,
        ):
            page = pdf.new_page()
            page.insert_textbox((72, 72, 523, 770), text, fontsize=11)
        pdf.save(source)
    original = source.read_bytes()
    output = tmp_path / "local-section-reviews" / "crosspage"
    helper("prepare_section_pilots").prepare(manifest_for(source, tmp_path), output)

    extraction = json.loads((output / "example-extraction-blocks.json").read_text())
    assert [block["page"] for block in extraction] == [1, 2]
    assert "substantive explanatory text" not in extraction[0]["text"]
    assert "substantive explanatory text" in extraction[1]["text"]
    classified = json.loads((output / "example-classified-blocks.json").read_text())
    assert classified[0]["blocks"][0]["section"]["section_role"] == "contents"
    assert classified[1]["blocks"][0]["section"]["section_role"] == "body"
    assert "substantive explanatory text" in classified[1]["blocks"][0]["text"]
    pending = json.loads((output / "example-questions.json").read_text())
    assert not pending
    receipt = json.loads((output / "cpu-acceptance.json").read_text())
    assert receipt["offline_accepted"]
    assert receipt["books"]["example"]["pending_questions"] == 0
    preservation = json.loads((output / "example-preservation.json").read_text())
    assert preservation["accepted"]
    assert not preservation["pending"]
    assert not receipt["books"]["example"]["unique_normalized_words_missing_after_cleanup"]
    assert source.read_bytes() == original


def test_native_contents_header_prefix_is_never_recovered_twice(tmp_path):
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "synthetic-prefix.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            (72, 72, 523, 770),
            "viii\\t\\nCONTENTS\\nChapter One  1\\nChapter Two  2\\nChapter Three  3",
            fontsize=11,
        )
        pdf.save(source)
    output = tmp_path / "prefix-run"
    helper("prepare_section_pilots").prepare(manifest_for(source, tmp_path), output)
    proof = json.loads((output / "example-preservation.json").read_text())
    assert proof["accepted"], proof["pending"]
    text = "".join(
        json.loads(line)["text"] for line in (output / "example-cpu.jsonl").read_text().splitlines()
    )
    assert text.count("viii") == 1
    assert text.count("CONTENTS") == 1
    assert helper("verify_section_pilots").verify(output)["accepted"]


@pytest.mark.parametrize("surface", ["raw", "stage", "exclusion", "role"])
def test_independent_reader_and_stage_replay_reject_rehashed_forged_sidecars(tmp_path, surface):
    source = synthetic_source(tmp_path)
    output = tmp_path / "forged-run"
    helper("prepare_section_pilots").prepare(manifest_for(source, tmp_path), output)
    receipt_path = output / "cpu-acceptance.json"
    receipt = json.loads(receipt_path.read_text())
    name = {
        "raw": "example-original-structure.json",
        "stage": "example-preservation.json",
        "exclusion": "settings.json",
        "role": "example-classified-blocks.json",
    }[surface]
    path = output / name
    value = json.loads(path.read_text())
    if surface == "raw":
        value["pages"][0]["blocks"][0]["raw_text"] += " Invented forged glyphs."
    elif surface == "stage":
        value["stages"][-1]["records"][0]["spans"][0]["source_span"] = [0, 1]
    elif surface == "exclusion":
        value["options"]["pdf_parse"]["exclude_pages"] = [1]
    else:
        value[0]["blocks"][0]["section"]["section_role"] = "body"
    path.write_text(json.dumps(value))
    receipt["books"]["example"]["sidecars"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((AssertionError, ValueError)):
        helper("verify_section_pilots").verify(output)


def test_external_review_answers_use_exact_canonical_replay_and_stale_rejection(tmp_path):
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "synthetic-ambiguous.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            (72, 72, 523, 770), "Acknowledgements\nWe thank our friends.\n1\n20\n30", fontsize=11
        )
        pdf.set_toc([[1, "Contents", 1]])
        pdf.save(source)
    manifest = manifest_for(source, tmp_path)
    draft = tmp_path / "pending"
    helper("prepare_section_pilots").prepare(manifest, draft)
    questions = json.loads((draft / "example-questions.json").read_text())
    assert questions
    answers = [
        {
            **question,
            "origin": "user",
            "reason": "Synthetic reviewed ambiguity fixture.",
            **({"role": "unknown"} if question["kind"] == "section_role" else {"boundary": True}),
        }
        for question in questions
    ]
    answer_path = tmp_path / "answers.json"
    answer_path.write_text(json.dumps(answers))
    controls = tmp_path / "answer-controls.json"
    controls.write_text(
        json.dumps(
            {
                "example": {
                    "path": str(answer_path),
                    "sha256": hashlib.sha256(answer_path.read_bytes()).hexdigest(),
                }
            }
        )
    )
    resolved = tmp_path / "resolved"
    helper("prepare_section_pilots").prepare(manifest, resolved, answers_manifest_path=controls)
    assert helper("verify_section_pilots").verify(resolved)["accepted"]
    assert not json.loads((resolved / "example-questions.json").read_text())
    assert all(
        value["origin"] == "user"
        for value in json.loads((resolved / "example-decisions.json").read_text())
    )
    answers[0]["source_sha256"] = "0" * 64
    answer_path.write_text(json.dumps(answers))
    controls.write_text(
        json.dumps(
            {
                "example": {
                    "path": str(answer_path),
                    "sha256": hashlib.sha256(answer_path.read_bytes()).hexdigest(),
                }
            }
        )
    )
    with pytest.raises(ValueError, match="stale"):
        helper("prepare_section_pilots").prepare(
            manifest, tmp_path / "stale", answers_manifest_path=controls
        )
