from pathlib import Path

from functools import reduce

import pytest

from pdf_chunker.adapters.io_pdf import read
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import _rebalance_lists, emit_jsonl
from pdf_chunker.passes.split_semantic import DEFAULT_SPLITTER, _merge_blocks


def _sentence(words: int, *, start: int = 0) -> str:
    return " ".join(f"Word{i}" for i in range(start, start + words)) + "."


def test_emit_jsonl_collapses_sentence_soft_wraps(monkeypatch) -> None:
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": (
                    "There is no getting away from the needs of operating\n\nsoftware. "
                    "This continuation keeps flowing."
                )
            }
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert rows == [
        {
            "text": (
                "There is no getting away from the needs of operating software. "
                "This continuation keeps flowing."
            )
        }
    ]


def test_emit_jsonl_retains_paragraph_breaks(monkeypatch) -> None:
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": (
                    "Teams presented it in two ways:\n\nSplit\n\nThe summary "
                    "continues elsewhere."
                )
            }
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert rows == [
        {
            "text": (
                "Teams presented it in two ways:\n\nSplit\n\nThe summary "
                "continues elsewhere."
            )
        }
    ]


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


def test_emit_jsonl_retains_caption_label_after_overlap(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    intro = "Teams repeatedly reference the architecture seen in Figure 9-1."
    caption = "Figure 9-1. Platform control plane layering across product areas"
    doc = {
        "type": "chunks",
        "items": [
            {"text": intro},
            {"text": caption},
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    texts = [row["text"] for row in rows]
    assert any(caption in text for text in texts)
    truncated = "Platform control plane layering across product areas"
    assert not any(text.startswith(truncated) for text in texts)


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
    expected = (
        "This opening sentence is intentionally long to satisfy the coherence heuristic and ends properly. "
        "and lacks terminal punctuation while being sufficiently long to trigger validation logic"
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


def test_emit_jsonl_retains_coherent_short_blocks():
    doc = {
        "type": "chunks",
        "items": [
            {"text": _sentence(30, start=0)},
            {"text": _sentence(25, start=100)},
            {"text": _sentence(28, start=200)},
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert len(rows) == 2
    assert rows[1]["text"] == _sentence(28, start=200)


def test_emit_jsonl_retains_numbered_punctuated_item(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    doc = {
        "type": "chunks",
        "items": [
            {"text": ("1. First item has enough words to be considered coherent and ends.")},
            {"text": ("2. Second item has enough words to be coherent as well.")},
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert rows[1]["text"].startswith("2. Second item")


def test_emit_jsonl_retains_numbered_tail_without_punctuation(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    doc = {
        "type": "chunks",
        "items": [
            {"text": ("1. First item has enough words to be considered coherent and ends.")},
            {"text": ("2. Second item continues with sufficient words but lacks punctuation")},
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert rows[1]["text"].startswith("2. Second item continues")


def test_rebalance_lists_moves_intro_to_list_block():
    raw = (
        "The platform story highlights enduring bottlenecks. "
        "Here are the recurring causes—namely:"
    )
    rest = "\n\n1. Teams cannot self-service their needs.\n2. Platform scope is too broad."
    kept, moved = _rebalance_lists(raw, rest)
    assert kept == "The platform story highlights enduring bottlenecks."
    lines = [ln for ln in moved.splitlines() if ln.strip()]
    assert lines[0] == "Here are the recurring causes—namely:"
    assert lines[1].startswith("1. Teams cannot self-service")


def test_emit_jsonl_preserves_caption_sentence_start(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    intro = "Context around prior analysis leading into Figure"
    caption = (
        "Figure 1-3 shows a high-level comparison of the two approaches.\n\n"
        "Figure 1-3. Comparison of IaaS and PaaS models in terms of vendor versus customer responsibility"
        " Initially, it was hoped that application teams would embrace fully supported PaaS offerings."
    )
    doc = {"type": "chunks", "items": [{"text": intro}, {"text": caption}]}
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert len(rows) == 1
    text = rows[0]["text"]
    assert text.count("Figure 1-3 shows") == 1
    assert "\n1-3 shows" not in text
    first_alpha = next((ch for ch in text if ch.isalpha()), "")
    assert first_alpha.isupper()
    assert "Initially, it was hoped" in text


def test_caption_block_merges_after_reference_sentence():
    page = 3
    sequence = tuple(
        (page, {"text": text}, text)
        for text in (
            (
                "You end up with an architecture more like Figure 1-3 "
                "shows a high-level comparison of the two approaches."
            ),
            "Figure 1-3. Comparison of IaaS and PaaS models in terms of vendor versus customer responsibility",
            "Initially, it was hoped that application teams would embrace fully supported PaaS offerings.",
        )
    )
    merged = reduce(_merge_blocks, sequence, [])
    assert len(merged) == 2
    intro_with_caption, following = (record[2] for record in merged)
    assert "Figure 1-3. Comparison" in intro_with_caption
    assert following.startswith("Initially")


@pytest.mark.usefixtures("_nltk_data")
def test_platform_eng_caption_survives_emit_jsonl() -> None:
    pytest.importorskip("fitz")
    doc = read(str(Path("platform-eng-excerpt.pdf")))
    artifact = DEFAULT_SPLITTER(Artifact(payload=doc))
    items = artifact.payload.get("items", [])
    rows = emit_jsonl(Artifact(payload={"type": "chunks", "items": items})).payload
    caption = "Figure 1-1. The over-general swamp, held together by glue"
    texts = [row["text"] for row in rows]
    assert any(caption in text for text in texts)
    truncated = "The over-general swamp, held together by glue"
    offenders = [
        text for text in texts if text.startswith(truncated) and caption not in text
    ]
    assert not offenders, "caption should retain its figure label"
    starters = [text for text in texts if text.lstrip().startswith("Figure 1-1.")]
    assert not starters, "caption should not start a fresh chunk"
    combined = "seen in Figure 1-1.\n\nFigure 1-1. The over-general swamp"
    assert any(combined in text for text in texts), "caption should follow its callout"


def test_emit_jsonl_rebalances_sentence_after_limit(monkeypatch):
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MIN_WORDS", "1")
    monkeypatch.setenv("PDF_CHUNKER_JSONL_MAX_CHARS", "400")
    prefix = (
        "This long introduction describes how teams collaborate across disciplines to ship reliable "
        "software platforms while still iterating quickly on features and fixes, ultimately achieving"
    )
    continuation = (
        " the expected outcomes. Another sentence follows to ensure that the remaining text still "
        "exceeds the artificial limit so emission has to split the chunk while respecting sentence "
        "boundaries and keeping the next sentence intact."
    )
    doc = {
        "type": "chunks",
        "items": [
            {"text": prefix},
            {"text": continuation},
        ],
    }
    rows = emit_jsonl(Artifact(payload=doc)).payload
    assert any(row["text"].endswith("the expected outcomes.") for row in rows)
    target = next(row for row in rows if "Another sentence follows" in row["text"])
    assert target["text"].lstrip().startswith("Another sentence follows")
    first_chars = [row["text"].lstrip()[0] for row in rows if row["text"].strip()]
    assert all(not ch.islower() for ch in first_chars)
