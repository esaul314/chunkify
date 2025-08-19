import json

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import assemble_report, run_convert, write_run_report
from pdf_chunker.framework import Artifact


def test_run_convert_writes_jsonl(tmp_path):
    spec = PipelineSpec(
        pipeline=["emit_jsonl"],
        options={
            "emit_jsonl": {"output_path": str(tmp_path / "out.jsonl")},
            "run_report": {"output_path": str(tmp_path / "run_report.json")},
        },
    )
    doc = {"type": "chunks", "items": [{"text": "a"}, {"text": "b"}]}
    artifact = Artifact(payload=doc, meta={"metrics": {}, "input": "doc.pdf"})
    artifact, timings = run_convert(artifact, spec)
    report = assemble_report(timings, artifact.meta or {})
    write_run_report(spec, report)

    out_file = tmp_path / "out.jsonl"
    lines = out_file.read_text(encoding="utf-8").splitlines()
    expected = [json.dumps(r, ensure_ascii=False) for r in artifact.payload]
    assert lines == expected

    report_file = tmp_path / "run_report.json"
    assert report_file.exists()
