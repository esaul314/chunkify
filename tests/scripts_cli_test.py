from __future__ import annotations
import base64
import json
import os
import subprocess
from pathlib import Path


def _materialize(src: Path, dst: Path, ext: str) -> Path:
    data = base64.b64decode(src.read_text())
    target = dst / f"sample.{ext}"
    target.write_bytes(data)
    return target


def test_convert_cli_writes_jsonl(tmp_path: Path) -> None:
    b64 = Path("tests/golden/samples/sample.pdf.b64")
    pdf_path = _materialize(b64, tmp_path, "pdf")
    out_file = tmp_path / "out.jsonl"
    cmd = [
        "python",
        "-m",
        "pdf_chunker.cli",
        "convert",
        str(pdf_path),
        "--chunk-size",
        "1000",
        "--overlap",
        "0",
        "--out",
        str(out_file),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert result.returncode == 0
    rows = [
        json.loads(line)
        for line in out_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
