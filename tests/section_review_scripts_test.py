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


def test_generated_crosspage_contents_conflict_stays_pending_and_preserves_body(tmp_path):
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
    assert extraction[0]["source"]["page_range"] == [1, 2]
    assert "substantive explanatory text" in extraction[0]["text"]
    classified = json.loads((output / "example-classified-blocks.json").read_text())
    assert classified[0]["blocks"][0]["text"] == extraction[0]["text"]
    assert classified[0]["blocks"][0]["section"]["section_role"] == "unknown"
    pending = json.loads((output / "example-questions.json").read_text())
    assert len(pending) == 2
    assert all("substantive explanatory text" in q["extracted_text"] for q in pending)
    receipt = json.loads((output / "cpu-acceptance.json").read_text())
    assert not receipt["offline_accepted"]
    assert receipt["books"]["example"]["pending_questions"] == 2
    assert not receipt["books"]["example"]["unique_normalized_words_missing_after_cleanup"]
    assert source.read_bytes() == original
