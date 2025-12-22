from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
from pathlib import Path
from shutil import which

import pytest


def _is_cli_prerequisite_available() -> bool:
    return all(
        (
            importlib.util.find_spec("fitz") is not None,
            which("pdf_chunker") is not None,
        )
    )


pytestmark = pytest.mark.skipif(
    not _is_cli_prerequisite_available(),
    reason="PyMuPDF (fitz) and pdf_chunker CLI required for smoke test",
)


def _materialize_sample_pdf() -> Path:
    """Ensure the sample PDF exists by decoding its base64 form."""
    samples_dir = Path("tests/golden/samples")
    pdf_path = samples_dir / "sample.pdf"
    if not pdf_path.exists():
        b64_path = samples_dir / "sample.pdf.b64"
        pdf_path.write_bytes(base64.b64decode(b64_path.read_text()))
    return pdf_path


def _cleanup(*paths: Path) -> None:
    tuple(p.unlink(missing_ok=True) for p in paths)


def test_cli_smoke() -> None:
    pdf_path = _materialize_sample_pdf()
    out_path = Path("tmp.jsonl")
    report_path = Path("run_report.json")
    _cleanup(out_path, report_path)
    result = subprocess.run(
        [
            "pdf_chunker",
            "convert",
            str(pdf_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        assert result.returncode == 0
        assert out_path.exists()
        rows = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows
        assert report_path.exists()
    finally:
        _cleanup(out_path, report_path)
