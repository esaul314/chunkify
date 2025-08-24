from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path

from tests.utils.materialize import materialize_base64

BASE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]


def test_conversion_epub_cli(tmp_path: Path) -> None:
    epub_path = materialize_base64(
        BASE / "samples" / "sample.epub.b64", tmp_path, "sample.epub"
    )
    out_file = tmp_path / "out.jsonl"
    cmd = [
        "python",
        "-m",
        "pdf_chunker.cli",
        "convert",
        str(epub_path),
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
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        cwd=tmp_path,
    )
    assert result.returncode == 0
    actual = [
        json.loads(line)
        for line in out_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [
        json.loads(line)
        for line in (BASE / "expected" / "epub.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert actual == expected
