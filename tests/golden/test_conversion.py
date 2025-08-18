from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

import pdf_chunker.text_cleaning as tc
from pdf_chunker.core import process_document

BASE_DIR = Path(__file__).resolve().parent

DOCS: dict[str, Path] = {
    "pdf": BASE_DIR / "samples" / "sample.pdf.b64",
    "epub": BASE_DIR / "samples" / "sample.epub.b64",
}


def _jsonl(chunks: list[dict[str, object]]) -> str:
    """Serialize chunks into stable JSONL."""
    return "\n".join(json.dumps(c, sort_keys=True) for c in chunks)


def _materialize(kind: str, b64_path: Path, tmp_dir: Path) -> Path:
    target = tmp_dir / f"sample.{kind}"
    target.write_bytes(base64.b64decode(b64_path.read_text()))
    return target


@pytest.mark.parametrize("kind,b64_path", DOCS.items())
def test_conversion(kind: str, b64_path: Path, file_regression, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tc, "clean_text", lambda text: text)
    path = _materialize(kind, b64_path, tmp_path)
    chunks = process_document(
        str(path),
        chunk_size=1000,
        overlap=0,
        generate_metadata=True,
        ai_enrichment=False,
    )
    expected = BASE_DIR / "expected" / f"{kind}.jsonl"
    file_regression.check(_jsonl(chunks), fullpath=expected, encoding="utf-8")
