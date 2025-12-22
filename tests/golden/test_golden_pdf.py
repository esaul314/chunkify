from __future__ import annotations

import json
from pathlib import Path

import pdf_chunker.text_cleaning as tc
from pdf_chunker.core import process_document

BASE_DIR = Path(__file__).resolve().parent


def _jsonl(chunks: list[dict[str, object]]) -> str:
    """Serialize chunks into stable JSONL."""
    return "\n".join(json.dumps(c, sort_keys=True) for c in chunks)


def test_golden_pdf(file_regression, monkeypatch) -> None:
    monkeypatch.setattr(tc, "clean_text", lambda text: text)
    path = BASE_DIR / "samples" / "tiny.pdf"
    chunks = process_document(
        str(path),
        chunk_size=1000,
        overlap=0,
        generate_metadata=True,
        ai_enrichment=False,
    )
    expected = BASE_DIR / "expected" / "tiny.jsonl"
    file_regression.check(_jsonl(chunks), fullpath=expected, encoding="utf-8")
