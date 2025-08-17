from __future__ import annotations

import json
from pathlib import Path

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import run_convert

BASE_DIR = Path(__file__).resolve().parent


def _spec(tmp: Path) -> PipelineSpec:
    """Return a minimal PDF pipeline spec bound to temporary paths."""
    return PipelineSpec(
        pipeline=["pdf_parse", "text_clean", "split_semantic"],
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
    artifact = run_convert(str(pdf), spec)
    jsonl = _jsonl(artifact.payload if isinstance(artifact.payload, list) else [])
    expected = BASE_DIR / "expected" / "sample.jsonl"
    file_regression.check(jsonl, fullpath=expected, encoding="utf-8")
