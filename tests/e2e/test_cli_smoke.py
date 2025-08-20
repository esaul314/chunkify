from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from shutil import which

import pytest

# Skip if heavy PDF dependency or CLI is missing
pytest.importorskip("fitz")
if which("pdf_chunker") is None:
    pytest.skip("pdf_chunker CLI not installed", allow_module_level=True)


def _materialize_sample_pdf() -> Path:
    """Ensure the sample PDF exists by decoding its base64 form."""
    samples_dir = Path("tests/golden/samples")
    pdf_path = samples_dir / "sample.pdf"
    if not pdf_path.exists():
        b64_path = samples_dir / "sample.pdf.b64"
        pdf_path.write_bytes(base64.b64decode(b64_path.read_text()))
    return pdf_path


def test_cli_smoke() -> None:
    pdf_path = _materialize_sample_pdf()
    out_path = Path("tmp.jsonl")
    if out_path.exists():
        out_path.unlink()
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
    finally:
        out_path.unlink(missing_ok=True)
