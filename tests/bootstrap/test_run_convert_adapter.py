from pathlib import Path

import pdf_chunker.adapters.io_pdf as io_pdf
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import run_convert


def test_run_convert_writes_jsonl(tmp_path, monkeypatch):
    def fake_read(path, exclude_pages=None):
        return {"type": "page_blocks", "source_path": path, "pages": []}

    monkeypatch.setattr(io_pdf, "read", fake_read)
    spec = PipelineSpec(
        pipeline=["pdf_parse", "emit_jsonl"],
        options={
            "emit_jsonl": {"output_path": str(tmp_path / "out.jsonl")},
            "run_report": {"output_path": str(tmp_path / "run_report.json")},
        },
    )
    pdf_path = Path("test_data") / "sample_test.pdf"
    artifact = run_convert(str(pdf_path), spec)
    out_file = tmp_path / "out.jsonl"
    report_file = tmp_path / "run_report.json"
    assert out_file.exists()
    assert report_file.exists()
    assert artifact.meta.get("input") == str(pdf_path)
