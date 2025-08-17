from __future__ import annotations

import json
from pathlib import Path

from pdf_chunker.adapters import io_pdf
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import assemble_report, run_convert, write_run_report
from pdf_chunker.framework import Artifact

BASE_DIR = Path(__file__).resolve().parent


def _spec(tmp: Path) -> PipelineSpec:
    """Return a minimal PDF pipeline spec bound to temporary paths."""
    return PipelineSpec(
        pipeline=["pdf_parse", "text_clean", "split_semantic", "emit_jsonl"],
        options={
            "emit_jsonl": {"output_path": str(tmp / "out.jsonl")},
            "run_report": {"output_path": str(tmp / "report.json")},
        },
    )


def _jsonl(chunks: list[dict[str, object]]) -> str:
    """Serialize ``chunks`` into deterministic JSONL."""
    return "\n".join(json.dumps(c, sort_keys=True) for c in chunks)


def test_conversion(file_regression, tmp_path: Path) -> None:
    pdf = BASE_DIR / "samples" / "sample.pdf"
    spec = _spec(tmp_path)
    payload = io_pdf.read(str(pdf))
    artifact = Artifact(payload=payload, meta={"metrics": {}, "input": str(pdf)})
    artifact, timings = run_convert(artifact, spec)
    report = assemble_report(timings, artifact.meta or {})
    write_run_report(spec, report)
    jsonl = _jsonl(artifact.payload if isinstance(artifact.payload, list) else [])
    expected = BASE_DIR / "expected" / "sample.jsonl"
    file_regression.check(jsonl, fullpath=expected, encoding="utf-8")
