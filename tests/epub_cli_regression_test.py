from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from tests.utils.materialize import materialize_base64

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_cli_epub_matches_golden(tmp_path: Path) -> None:
    epub = materialize_base64(
        Path("tests/golden/samples/sample.epub.b64"), tmp_path, "sample.epub"
    )
    out_file = tmp_path / "out.jsonl"
    cmd = [
        "python",
        "-m",
        "pdf_chunker.cli",
        "convert",
        str(epub),
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
    actual = _read_jsonl(out_file)
    expected = _read_jsonl(Path("tests/golden/expected/epub.jsonl"))
    assert actual == expected

