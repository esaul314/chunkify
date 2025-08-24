from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path


def _materialize(src: Path, dst: Path) -> Path:
    data = base64.b64decode(src.read_text())
    target = dst / "sample.epub"
    target.write_bytes(data)
    return target


def test_conversion_epub_cli(tmp_path: Path) -> None:
    b64_path = Path("tests/golden/samples/sample.epub.b64")
    epub_path = _materialize(b64_path, tmp_path)
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
        env={**os.environ, "PYTHONPATH": "."},
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
        for line in Path("tests/golden/expected/epub.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert actual == expected
