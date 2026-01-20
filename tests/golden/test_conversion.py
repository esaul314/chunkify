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
}


def _sorted_mapping(value: object) -> object:
    if isinstance(value, dict):
        return {k: _sorted_mapping(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted_mapping(item) for item in value]
    return value


def _ordered_dump(chunk: dict[str, object]) -> str:
    """Serialize chunk JSON with text first, then metadata, then other keys."""
    ordered: dict[str, object] = {}
    if "text" in chunk:
        ordered["text"] = chunk["text"]
    if "metadata" in chunk:
        ordered["metadata"] = _sorted_mapping(chunk["metadata"])
    for key in sorted(k for k in chunk.keys() if k not in {"text", "metadata"}):
        ordered[key] = _sorted_mapping(chunk[key])
    return json.dumps(ordered, ensure_ascii=False)


def _jsonl(chunks: list[dict[str, object]]) -> str:
    """Serialize chunks into stable JSONL."""
    return "\n".join(_ordered_dump(c) for c in chunks)


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
