from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import emit_jsonl


def test_emit_jsonl_truncates_oversized_text():
    long_text = "a" * 9000
    artifact = Artifact(payload={"type": "chunks", "items": [{"text": long_text}]})
    rows = emit_jsonl(artifact).payload
    assert len(rows[0]["text"]) <= 8000
