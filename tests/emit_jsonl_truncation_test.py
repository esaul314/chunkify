import json
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import emit_jsonl
from pdf_chunker.passes.split_semantic import split_semantic


def test_emit_jsonl_splits_and_clamps_rows():
    long_text = "A" + "a" * 8998 + "."
    artifact = Artifact(payload={"type": "chunks", "items": [{"text": long_text}]})
    rows = emit_jsonl(artifact).payload
    assert len(rows) > 1
    assert all(len(json.dumps(r, ensure_ascii=False)) <= 8000 for r in rows)


def test_split_semantic_produces_bounded_chunks():
    long_text = " ".join(f"w{i}" for i in range(2500))
    doc = {"type": "page_blocks", "pages": [{"blocks": [{"text": long_text}]}]}
    artifact = Artifact(payload=doc)
    items = split_semantic(artifact).payload["items"]
    texts = [c["text"] for c in items]
    assert len(texts) > 1
    assert all(len(t.split()) <= 400 for t in texts)
