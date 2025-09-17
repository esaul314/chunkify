from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from pdf_chunker.pdf_parsing import extract_text_blocks_from_pdf
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


def _jsonl_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rows_from_lines(lines: Iterable[str]) -> list[dict[str, object]]:
    return [json.loads(line) for line in lines]


def _rows(path: Path, lines: Iterable[str] | None = None) -> list[dict[str, object]]:
    return _rows_from_lines(lines if lines is not None else _jsonl_lines(path))


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
    lines = _jsonl_lines(out_file)
    rows = _rows(out_file, lines)
    assert lines and rows
    # Exercise streaming extraction without relying on chunk counts
    assert sum(1 for _ in extract_text_blocks_from_pdf(str(pdf_path))) > 0
    report = json.loads((tmp_path / "run_report.json").read_text())
    assert {"timings", "metrics", "warnings"} <= report.keys()
    metrics = report["metrics"]
    assert metrics["page_count"] == 3
    assert metrics["split_semantic"]["chunks"] == metrics["chunk_count"]
    emit_rows = metrics.get("emit_jsonl", {}).get("rows")
    if emit_rows is not None:
        assert emit_rows == len(rows)
    assert not report["warnings"]


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
    lines = _jsonl_lines(out_file)
    rows = _rows(out_file, lines)
    assert lines and rows and all("metadata" not in row for row in rows)


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
    lines = _jsonl_lines(out_file)
    rows = _rows(out_file, lines)
    assert lines and rows and all(r.get("meta", {}).get("page") != 1 for r in rows)


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
    lines = _jsonl_lines(out_file)
    rows = _rows(out_file, lines)
    assert lines and rows and all(r.keys() == {"text"} for r in rows)


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
    lines = _jsonl_lines(out_file)
    rows = _rows(out_file, lines)
    assert lines and rows
    tokens = [r["text"].split() for r in rows]
    report = json.loads((tmp_path / "run_report.json").read_text())
    metrics = report["metrics"]
    emit_rows = metrics.get("emit_jsonl", {}).get("rows")
    if emit_rows is not None:
        assert emit_rows == len(rows)
    chunk_count = metrics["split_semantic"]["chunks"]
    assert chunk_count == metrics["chunk_count"]
    assert chunk_count >= 2
    assert len(rows) == chunk_count


