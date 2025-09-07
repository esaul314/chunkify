import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:  # pragma: no cover - skip if dependency missing
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "platform-eng-excerpt.pdf"
SENTINEL = "Alignment and trust are challenging"


def test_page_count_regression(tmp_path: Path) -> None:
    out = tmp_path / "platform-eng.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pdf_chunker.cli",
            "convert",
            str(PDF),
            "--chunk-size",
            "1000",
            "--overlap",
            "0",
            "--out",
            str(out),
            "--no-enrich",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "run_report.json").read_text())
    page_count = report["metrics"]["page_count"]
    pytest.importorskip("pypdf")
    truth = len(PdfReader(str(PDF)).pages)
    assert truth == page_count
    text = out.read_text()
    assert SENTINEL in text

    lines = [json.loads(line)["text"] for line in text.splitlines()]
    toc = [t for t in lines if "Why Platform Engineering Is Becoming Essential" in t]
    assert len(toc) == 1
    assert ". ." not in toc[0]
    assert not toc[0].rstrip().endswith("1")
