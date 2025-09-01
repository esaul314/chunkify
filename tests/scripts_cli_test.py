from __future__ import annotations

import json
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


def _run_chunk_pdf(
    *args: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.chunk_pdf", *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        cwd=cwd,
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_convert_cli_writes_jsonl(tmp_path: Path) -> None:
    pdf_path = materialize_base64(
        Path("tests/golden/samples/sample.pdf.b64"), tmp_path, "sample.pdf"
    )
    out_file = tmp_path / "out.jsonl"
    result = _run_cli(
        "convert",
        str(pdf_path),
        "--chunk-size",
        "1000",
        "--overlap",
        "0",
        "--out",
        str(out_file),
        cwd=tmp_path,
    )
    assert result.returncode == 0
    rows = [
        json.loads(line)
        for line in out_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    report = json.loads((tmp_path / "run_report.json").read_text())
    assert {"timings", "metrics", "warnings"} <= report.keys()
    assert report["metrics"]["page_count"] == 3
    assert report["metrics"]["chunk_count"] == len(rows)
    assert report["warnings"] == ["metadata_gaps"]


def test_root_help_lists_commands() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    assert all(cmd in result.stdout for cmd in ("convert", "inspect"))


def test_convert_help_lists_expected_flags() -> None:
    result = _run_cli("convert", "--help")
    assert result.returncode == 0
    out = result.stdout
    flags = (
        "--enrich",
        "--no-enrich",
        "--exclude-pages",
        "--chunk-size",
        "--overlap",
        "--no-metadata",
        "--spec",
        "--verbose",
    )
    assert all(f in out for f in flags)


def test_chunk_pdf_accepts_flags(tmp_path: Path) -> None:
    pdf_path = materialize_base64(
        Path("tests/golden/samples/sample.pdf.b64"), tmp_path, "sample.pdf"
    )
    out_file = tmp_path / "out.jsonl"
    result = _run_chunk_pdf(
        str(pdf_path),
        "--chunk-size",
        "1000",
        "--overlap",
        "0",
        "--exclude-pages",
        "2",
        "--no-metadata",
        "--out",
        str(out_file),
        cwd=tmp_path,
    )
    assert result.returncode == 0
    rows = [
        json.loads(line)
        for line in out_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows and all("metadata" not in row for row in rows)


def test_cli_exclude_pages_flag(tmp_path: Path) -> None:
    pdf_path = materialize_base64(
        Path("tests/golden/samples/sample.pdf.b64"), tmp_path, "sample.pdf"
    )
    out_file = tmp_path / "out.jsonl"
    result = _run_cli(
        "convert",
        str(pdf_path),
        "--exclude-pages",
        "1",
        "--out",
        str(out_file),
        cwd=tmp_path,
    )
    assert result.returncode == 0
    rows = _rows(out_file)
    assert rows and all(r.get("meta", {}).get("page") != 1 for r in rows)


def test_cli_no_metadata_flag(tmp_path: Path) -> None:
    pdf_path = materialize_base64(
        Path("tests/golden/samples/sample.pdf.b64"), tmp_path, "sample.pdf"
    )
    out_file = tmp_path / "out.jsonl"
    result = _run_cli(
        "convert",
        str(pdf_path),
        "--chunk-size",
        "1000",
        "--overlap",
        "0",
        "--no-metadata",
        "--out",
        str(out_file),
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert _rows(out_file) and all(r.keys() == {"text"} for r in _rows(out_file))


def test_cli_chunk_size_overlap_flags(tmp_path: Path) -> None:
    pdf_path = materialize_base64(
        Path("tests/golden/samples/sample.pdf.b64"), tmp_path, "sample.pdf"
    )
    out_file = tmp_path / "out.jsonl"
    result = _run_cli(
        "convert",
        str(pdf_path),
        "--chunk-size",
        "5",
        "--overlap",
        "2",
        "--out",
        str(out_file),
        cwd=tmp_path,
    )
    assert result.returncode == 0
    tokens = [r["text"].split() for r in _rows(out_file)]
    assert tokens and len(tokens[0]) <= 5
    if len(tokens) >= 2:
        assert tokens[1][:2] == tokens[0][-2:]

