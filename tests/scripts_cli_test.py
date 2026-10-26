from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from pdf_chunker.cli import ConvertOptions, run_conversion
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


def _run_chunk_pdf(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.chunk_pdf", *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        cwd=cwd,
    )


def _jsonl_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


@pytest.mark.parametrize(
    "terminal",
    [
        pytest.param({}, id="plain"),
        pytest.param({"CI": "true", "GITHUB_ACTIONS": "true"}, id="github"),
        pytest.param(
            {"CI": "true", "GITHUB_ACTIONS": "true", "FORCE_COLOR": "1"},
            id="forced-color",
        ),
    ],
)
def test_convert_help_lists_expected_flags(
    monkeypatch: pytest.MonkeyPatch, terminal: dict[str, str]
) -> None:
    for key in ("CI", "GITHUB_ACTIONS", "FORCE_COLOR", "NO_COLOR", "CLICOLOR_FORCE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "80")
    for key, value in terminal.items():
        monkeypatch.setenv(key, value)
    result = _run_cli("convert", "--help")
    assert result.returncode == 0
    # Rich styles can split an option token into several ANSI-colored spans.
    out = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stdout)
    flags = (
        "--metadata-file",
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
    assert ".pdf" in out and ".epub" in out and ".txt" in out


def test_convert_cli_chunks_text_transcript_with_metadata(tmp_path: Path) -> None:
    transcript = tmp_path / "episode.txt"
    transcript.write_text(
        "\n".join(
            (
                "WEBVTT",
                "",
                "1",
                "00:00:01.000 --> 00:00:03.000",
                "HOST:",
                "Welcome back.",
                "",
                "00:00:04.000 --> 00:00:06.000",
                "GUEST:",
                "Thanks for having me.",
            )
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "episode-metadata.yaml"
    metadata_path.write_text(
        "\n".join(
            (
                "doc_id: episode-001",
                "source: podcasts/signal/episode-001",
                "source_file: episode.txt",
                "source_kind: podcast_episode",
                "source_system: podcast_addict",
                "podcast_name: Signal",
                "episode_name: Episode One",
                "corpus: podcasts",
                "publication_date_utc: 2026-07-16T00:00:00Z",
                "played_at_utc: 2026-07-17T10:30:00Z",
                "best_link: https://example.com/episode-001",
                "transcript_url: https://example.com/episode-001/transcript",
                "transcript_source_url: https://example.com/episode-001/source",
                "transcript_source_kind: webvtt",
            )
        ),
        encoding="utf-8",
    )
    out_file = tmp_path / "out.jsonl"

    result = _run_cli(
        "convert",
        str(transcript),
        "--metadata-file",
        str(metadata_path),
        "--chunk-size",
        "1000",
        "--overlap",
        "0",
        "--out",
        str(out_file),
        cwd=tmp_path,
    )

    assert result.returncode == 0
    rows = _rows(out_file)
    assert rows
    merged_text = " ".join(row["text"] for row in rows)
    assert "WEBVTT" not in merged_text
    assert "00:00:01.000" not in merged_text
    assert "HOST: Welcome back." in merged_text
    assert "GUEST: Thanks for having me." in merged_text
    for row in rows:
        metadata = row["metadata"]
        assert metadata["doc_id"] == "episode-001"
        assert metadata["source"] == "podcasts/signal/episode-001"
        assert metadata["source_file"] == "episode.txt"
        assert metadata["source_kind"] == "podcast_episode"
        assert metadata["source_system"] == "podcast_addict"
        assert metadata["podcast_name"] == "Signal"
        assert metadata["episode_name"] == "Episode One"
        assert metadata["corpus"] == "podcasts"
        assert metadata["publication_date_utc"] == "2026-07-16T00:00:00Z"
        assert metadata["played_at_utc"] == "2026-07-17T10:30:00Z"
        assert metadata["best_link"] == "https://example.com/episode-001"
        assert metadata["transcript_url"] == "https://example.com/episode-001/transcript"
        assert metadata["transcript_source_url"] == "https://example.com/episode-001/source"
        assert metadata["transcript_source_kind"] == "webvtt"
        assert metadata["chunk_id"].startswith("episode.txt_p0_c")
    assert rows[0]["metadata"]["timestamp_start"] == "00:00:01.000"
    assert rows[0]["metadata"]["timestamp_end"] == "00:00:06.000"


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


def test_legacy_frontend_matches_modern_frontend(tmp_path: Path) -> None:
    pdf_path = materialize_base64(
        Path("tests/golden/samples/sample.pdf.b64"), tmp_path, "sample.pdf"
    )
    modern_out = tmp_path / "modern.jsonl"
    legacy_out = tmp_path / "legacy.jsonl"

    modern = _run_cli(
        "convert",
        str(pdf_path),
        "--chunk-size",
        "1000",
        "--overlap",
        "0",
        "--no-metadata",
        "--out",
        str(modern_out),
        cwd=tmp_path,
    )
    legacy = _run_chunk_pdf(
        str(pdf_path),
        "--chunk-size",
        "1000",
        "--overlap",
        "0",
        "--no-metadata",
        "--out",
        str(legacy_out),
        cwd=tmp_path,
    )

    assert modern.returncode == legacy.returncode == 0
    assert _rows(modern_out) == _rows(legacy_out)


def test_missing_explicit_spec_fails_before_conversion(tmp_path: Path) -> None:
    pdf_path = materialize_base64(
        Path("tests/golden/samples/sample.pdf.b64"), tmp_path, "sample.pdf"
    )
    result = _run_cli(
        "convert",
        str(pdf_path),
        "--spec",
        str(tmp_path / "missing.yaml"),
        "--out",
        str(tmp_path / "out.jsonl"),
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "pipeline spec does not exist" in result.stderr
    assert not (tmp_path / "out.jsonl").exists()


def test_max_chars_override_restores_environment(tmp_path: Path, monkeypatch) -> None:
    pdf_path = materialize_base64(
        Path("tests/golden/samples/sample.pdf.b64"), tmp_path, "sample.pdf"
    )
    output = tmp_path / "out.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MAX_CHARS", "original")

    run_conversion(
        ConvertOptions(
            input_path=pdf_path,
            out=output,
            chunk_size=1000,
            overlap=0,
            max_chars=2000,
        )
    )

    assert output.exists()
    assert os.environ["PDF_CHUNKER_JSONL_MAX_CHARS"] == "original"


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
    report = json.loads((tmp_path / "run_report.json").read_text())
    metrics = report["metrics"]
    emit_rows = metrics.get("emit_jsonl", {}).get("rows")
    if emit_rows is not None:
        assert emit_rows == len(rows)
    chunk_count = metrics["split_semantic"]["chunks"]
    assert chunk_count == metrics["chunk_count"]
    assert chunk_count >= 2
    assert len(rows) == chunk_count
