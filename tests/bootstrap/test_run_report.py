from __future__ import annotations

import json
from pathlib import Path

import pdf_chunker.pdf_parsing as pdf_parsing
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import run_convert
from pdf_chunker.framework import Artifact


def test_run_report_emitted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pdf_parsing,
        "_legacy_extract_text_blocks_from_pdf",
        lambda path, exclude_pages=None: [],
    )
    spec = PipelineSpec(
        pipeline=["pdf_parse"],
        options={"run_report": {"output_path": str(tmp_path / "run_report.json")}},
    )
    pdf_path = Path("test_data") / "sample_test.pdf"
    artifact = Artifact(payload=str(pdf_path), meta={"metrics": {}, "input": str(pdf_path)})
    run_convert(artifact, spec)
    report_file = tmp_path / "run_report.json"
    assert report_file.exists()
    data = json.loads(report_file.read_text())
    assert set(data) == {"timings", "metrics", "warnings"}
