from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.utils.materialize import materialize_base64

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pdf_chunker.cli", *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        cwd=cwd,
    )


def test_convert_is_deterministic(tmp_path: Path) -> None:
    pdf_src = Path("tests/golden/samples/sample.pdf.b64")
    dirs = tuple(tmp_path / name for name in ("first", "second"))
    [d.mkdir() for d in dirs]
    pdf_paths = tuple(materialize_base64(pdf_src, d, "sample.pdf") for d in dirs)
    out_files = tuple(d / "out.jsonl" for d in dirs)
    results = tuple(
        _run_cli(
            "convert",
            str(pdf),
            "--chunk-size",
            "1000",
            "--overlap",
            "0",
            "--out",
            str(out),
            cwd=out.parent,
        )
        for pdf, out in zip(pdf_paths, out_files)
    )
    assert all(r.returncode == 0 for r in results)
    assert out_files[0].read_bytes() == out_files[1].read_bytes()
