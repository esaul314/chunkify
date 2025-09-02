import json

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import emit_jsonl


def test_emit_jsonl_splits_and_clamps_rows():
    long_text = "a" * 9000
    artifact = Artifact(payload={"type": "chunks", "items": [{"text": long_text}]})
    rows = emit_jsonl(artifact).payload
    assert len(rows) > 1
    assert "".join(r["text"] for r in rows) == long_text
    assert all(len(json.dumps(r, ensure_ascii=False)) <= 8000 for r in rows)
