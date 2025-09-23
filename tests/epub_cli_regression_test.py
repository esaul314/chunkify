from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.materialize import materialize_base64

pytest.importorskip("ebooklib")

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cli_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT)}


@pytest.fixture
def epub_cli_rows(tmp_path: Path) -> list[dict[str, object]]:
    epub = materialize_base64(Path("tests/golden/samples/sample.epub.b64"), tmp_path, "sample.epub")
    out_file = tmp_path / "out.jsonl"
    cmd = [
        sys.executable,
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
        env=_cli_env(),
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "convert: OK" in result.stdout
    rows = _read_jsonl(out_file)
    assert rows, "CLI conversion should emit rows"
    return rows


def test_cli_epub_matches_expected_structure(epub_cli_rows: list[dict[str, object]]) -> None:
    assert len(epub_cli_rows) == 2
    metadata = [row["metadata"] for row in epub_cli_rows]
    chunk_ids = [meta["chunk_id"] for meta in metadata]
    assert chunk_ids == ["sample.epub_p0_c0", "sample.epub_p2_c1"]
    assert {meta.get("source") for meta in metadata} == {"sample.epub"}
    assert metadata[0]["block_type"] == "list_item"
    assert metadata[0].get("list_kind") == "numbered"
    assert metadata[1]["block_type"] == "paragraph"
    assert "list_kind" not in metadata[1]

    first_text, second_text = (row["text"] for row in epub_cli_rows)

    intro_prefix = (
        "1. Chapter 1: Introduction\n2. Chapter 2: Sample Content\n3. Chapter 3: Conclusion"
    )
    intro_suffix = (
        "Regular paragraph text continues here. This text should be processed as a normal paragraph block, separate from the dialogue above."
    )
    assert first_text.startswith(intro_prefix)
    assert first_text.endswith(intro_suffix)
    assert "\n\nChapter 2: Sample Content" in first_text
    assert "\"Yes, this helps test dialogue detection in EPUB format,\" replied the second." in first_text

    list_prefix = (
        "Subsection with Lists\n\nThis section tests structured content processing:\n"
        "1. First numbered item\n2. Second numbered item\n3. Third numbered item\n\n"
        "Another paragraph with some technical terms like PyMuPDF4LLM and text processing algorithms to test specialized handling."
    )
    conclusion_suffix = (
        "HTML content extraction\n\nMulti-chapter document handling\n\n"
        "Text cleaning and normalization End of test document."
    )
    expected_scaffolding = {
        "sample.epub_p0_c0": (intro_prefix, intro_suffix),
        "sample.epub_p2_c1": (list_prefix, conclusion_suffix),
    }
    actual_scaffolding = [
        (
            chunk_id,
            text[: len(prefix)],
            text[-len(suffix):] if suffix else "",
        )
        for chunk_id, text in (
            (row["metadata"]["chunk_id"], row["text"]) for row in epub_cli_rows
        )
        for prefix, suffix in (expected_scaffolding[chunk_id],)
    ]
    assert actual_scaffolding == [
        (chunk_id, prefix, suffix)
        for chunk_id, (prefix, suffix) in expected_scaffolding.items()
    ]
    assert second_text.startswith(list_prefix)
    assert second_text.endswith(conclusion_suffix)
    assert "\nChapter 3: Conclusion" in second_text
    assert "\n\nThe EPUB format allows for rich HTML content" in second_text
    assert (
        "1. First numbered item\n2. Second numbered item\n3. Third numbered item"
        in second_text
    )
    assert "\n\n1. First numbered item" not in second_text
