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


def test_emit_jsonl_trims_overlap_without_merging():
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
    assert rows == [
        {
            "text": (
                "This opening sentence is intentionally long to satisfy the coherence"
                " heuristic and ends properly."
            )
        }
    ]
