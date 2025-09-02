from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import emit_jsonl


def test_emit_jsonl_merges_heading_with_text():
    doc = {
        "type": "chunks",
        "items": [
            {"text": "Chapter 1"},
            {"text": "This paragraph has enough words to be valid."},
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert len(rows) == 1
    assert rows[0]["text"].startswith("Chapter 1\n\nThis paragraph has enough words")


def test_emit_jsonl_merges_incomplete_sentences():
    doc = {
        "type": "chunks",
        "items": [
            {"text": "This is the beginning of a sentence that lacks an ending"},
            {"text": "and adds more context to meet length rules."},
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert len(rows) == 1
    assert rows[0]["text"].startswith("This is the beginning of a sentence")
    assert rows[0]["text"].endswith("rules.")
