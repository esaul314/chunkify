from __future__ import annotations

import json
from pathlib import Path

import pdf_chunker.adapters.io_pdf as io_pdf
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import assemble_report, run_convert, write_run_report
from pdf_chunker.framework import Artifact


def test_run_report_emitted(tmp_path, monkeypatch):
    def fake_read(path, exclude_pages=None):
        return {"type": "page_blocks", "source_path": path, "pages": []}

    monkeypatch.setattr(io_pdf, "read", fake_read)
    spec = PipelineSpec(
        pipeline=["pdf_parse"],
        options={"run_report": {"output_path": str(tmp_path / "run_report.json")}},
    )
    pdf_path = Path("test_data") / "sample_test.pdf"
    payload = io_pdf.read(str(pdf_path))
    artifact = Artifact(payload=payload, meta={"metrics": {}, "input": str(pdf_path)})
    artifact, timings = run_convert(artifact, spec)
    report = assemble_report(timings, artifact.meta or {})
    write_run_report(spec, report)
    report_file = tmp_path / "run_report.json"
    data = json.loads(report_file.read_text())
    assert set(data) == {"timings", "metrics", "warnings"}
