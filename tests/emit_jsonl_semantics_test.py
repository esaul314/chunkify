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


def test_emit_jsonl_merges_tail_fragment():
    doc = {
        "type": "chunks",
        "items": [
            {"text": "All prior context resolves a sentence."},
            {"text": "continues the thought and ends properly."},
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert len(rows) == 1
    assert rows[0]["text"].startswith("All prior context")
    assert rows[0]["text"].endswith("ends properly.")


def test_emit_jsonl_trims_overlap_without_merging(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    first = "An act of parallel evolution, in about 2004 Google moved away."
    second = "Act of parallel\n\nCaused a lot of excitement across the team."
    doc = {
        "type": "chunks",
        "items": [
            {"text": first},
            {"text": second},
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert rows == [
        {"text": first},
        {"text": "Caused a lot of excitement across the team."},
    ]


def test_emit_jsonl_drops_incoherent_tail():
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": (
                    "This opening sentence is intentionally long to satisfy the"
                    " coherence heuristic and ends properly."
                )
            },
            {
                "text": (
                    "and lacks terminal punctuation while being sufficiently long to"
                    " trigger validation logic"
                )
            },
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    expected = "\n\n".join(
        (
            "This opening sentence is intentionally long to satisfy the coherence heuristic and ends properly.",
            "and lacks terminal punctuation while being sufficiently long to trigger validation logic",
        )
    )
    assert rows == [{"text": expected}]


def test_emit_jsonl_strips_leading_newline():
    text = "\nThis " + "word " * 54 + "ends."
    doc = {"type": "chunks", "items": [{"text": text}]}
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert not rows[0]["text"].startswith("\n")


def test_emit_jsonl_merges_short_items():
    short = "This item is short."
    long = "This sentence has many words " * 20 + "and ends here."
    doc = {"type": "chunks", "items": [{"text": short}, {"text": long}]}
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert len(rows) == 1
    assert len(rows[0]["text"].split()) >= 50


def test_emit_jsonl_retains_numbered_punctuated_item(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": (
                    "1. First item has enough words to be considered coherent and ends."
                )
            },
            {
                "text": (
                    "2. Second item has enough words to be coherent as well."
                )
            },
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert rows[1]["text"].startswith("2. Second item")


def test_emit_jsonl_retains_numbered_tail_without_punctuation(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": (
                    "1. First item has enough words to be considered coherent and ends."
                )
            },
            {
                "text": (
                    "2. Second item continues with sufficient words but lacks punctuation"
                )
            },
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert rows[1]["text"].startswith("2. Second item continues")
