import json
import os
import subprocess
from pathlib import Path

from pdf_chunker.passes.emit_jsonl import (
    _dedupe,
    _flag_potential_duplicates,
    _rows,
)


def test_leading_fragment_not_dropped():
    doc = {
        "type": "chunks",
        "items": [
            {"text": "fragment without punctuation"},
            {"text": "continues the thought without end"},
            {"text": "Final sentence."},
        ],
    }
    rows = _rows(doc)
    assert rows and rows[0]["text"].startswith("fragment without punctuation")


def test_duplicate_sentence_trimmed():
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": (
                    "the developers didn't want to use the interface. "
                    "Developers are picky about the way they work."
                )
            },
            {
                "text": (
                    "Developers are picky about the way they work. "
                    "Many want nothing to do with the interface."
                )
            },
        ],
    }
    rows = _rows(doc)
    assert sum("Developers are picky" in r["text"] for r in rows) == 1


def test_duplicate_sentence_with_whitespace_trimmed():
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": "Alpha. Developers are picky about the way they work.",
            },
            {
                "text": "Developers are  picky   about the way they work.  ",
            },
        ],
    }
    rows = _rows(doc)
    assert sum("Developers are picky" in r["text"] for r in rows) == 1


def test_non_adjacent_duplicate_sentence_trimmed():
    doc = {
        "type": "chunks",
        "items": [
            {"text": "Intro."},
            {
                "text": (
                    "Most engineers don't want to learn a whole new toolset for "
                    "infrequent tasks."
                )
            },
            {"text": "Filler paragraph in between."},
            {
                "text": (
                    "Most engineers don't want to learn a whole new toolset for "
                    "infrequent tasks."
                    " Another sentence follows."
                )
            },
        ],
    }
    rows = _rows(doc)
    assert sum("Most engineers don't want to learn" in r["text"] for r in rows) == 1


def test_prefix_overlap_trimmed():
    sent = " ".join(
        [
            "Most engineers don't want to learn a whole new toolset for",
            "infrequent tasks.",
        ]
    )
    filler1 = " ".join(f"alpha{i}" for i in range(55))
    filler2 = " ".join(f"beta{i}" for i in range(55))
    doc = {
        "type": "chunks",
        "items": [
            {"text": f"{filler1} {sent}"},
            {"text": f"{sent} {filler2}"},
        ],
    }
    rows = _rows(doc)
    combined = " ".join(r["text"] for r in rows)
    assert combined.count(sent) == 1


def test_prefix_overlap_preserves_sentence_start():
    items = [
        {"text": "Intro when application engineering matters."},
        {"text": "When application engineering teams succeed."},
    ]
    deduped = _dedupe(items)
    assert len(deduped) == 2
    assert deduped[1]["text"].startswith("When application engineering")


def test_flag_potential_duplicates():
    items = [
        {"text": " ".join(["word"] * 11) + "."},
        {"text": " ".join(["word"] * 11) + "."},
    ]
    assert _flag_potential_duplicates(items)


def test_numbered_item_fragment_merges_with_parent():
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": (
                    "This progression:\n"
                    "1. Most engineers don't want to learn a whole new toolset.\n"
                    "2. The shortage, combined with people cobbling together their own Terraform, led to chaos."
                )
            },
            {
                "text": (
                    "2. The shortage, combined with people cobbling together their own Terraform, led to chaos.\n"
                    "and forced teams to centralize their efforts."
                )
            },
        ],
    }
    rows = _rows(doc)
    combined = " ".join(row["text"] for row in rows)
    assert combined.count("2. The shortage") == 1
    assert "and forced teams to centralize their efforts." in combined


def test_numbered_item_duplicate_lines_pruned():
    doc = {
        "type": "chunks",
        "items": [
            {
                "text": (
                    "This progression:\n"
                    "1. Most engineers don't want to learn a whole new toolset.\n"
                    "2. The shortage, combined with people cobbling together their own Terraform, led to chaos.\n"
                    "3. These centralized Terraform-writing teams became trapped in a feature shop."
                )
            },
            {
                "text": (
                    "\n2. The shortage, combined with people cobbling together their own Terraform, led to chaos.\n"
                    "3. These centralized Terraform-writing teams became trapped in a feature shop.\n"
                    "A better path is to realize that you need to do something more coherent."
                )
            },
        ],
    }
    rows = _rows(doc)
    assert rows and rows[0]["text"].count("2. The shortage") == 1
    assert "A better path is to realize that you need to do something more coherent." in rows[0]["text"]


def test_split_does_not_duplicate(tmp_path: Path) -> None:
    pdf = Path("platform-eng-excerpt.pdf").resolve()
    spec = Path("pipeline.yaml").resolve()
    out = tmp_path / "out.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "PDF_CHUNKER_DEDUP_DEBUG": "1",
    }
    proc = subprocess.run(
        [
            "python",
            "-m",
            "pdf_chunker.cli",
            "convert",
            str(pdf),
            "--spec",
            str(spec),
            "--out",
            str(out),
            "--no-metadata",
        ],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    text_lines = [line for line in out.read_text().splitlines() if line.strip()]
    rows = [json.loads(line)["text"] for line in text_lines]
    assert rows, "conversion produced no rows"
    shortage_hits = [i for i, chunk in enumerate(rows, 1) if "2. The shortage" in chunk]
    assert len(shortage_hits) == 1, shortage_hits
    shortage_chunk = rows[shortage_hits[0] - 1]
    assert shortage_chunk.count("2. The shortage") == 1
    assert not shortage_chunk.lstrip().startswith("2. The shortage")
    mid_list_prefixes = tuple(f"{n}. " for n in range(2, 10))
    mid_list_starts = [
        (idx, chunk[:60])
        for idx, chunk in enumerate(rows, 1)
        if chunk.lstrip().startswith(mid_list_prefixes)
    ]
    assert not mid_list_starts, mid_list_starts
    word_counts = [len(chunk.split()) for chunk in rows]
    assert sum(count >= 100 for count in word_counts) / len(word_counts) >= 0.75
    matches = sum(chunk.count("Most engineers") for chunk in rows)
    assert matches == 1, proc.stderr
    assert any("Infrastructure setup" in chunk for chunk in rows)
    assert "dedupe dropped" in proc.stderr
