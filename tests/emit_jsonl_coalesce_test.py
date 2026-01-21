import json
import os
import subprocess
from pathlib import Path

from pdf_chunker.passes.emit_jsonl import (
    _flag_potential_duplicates,
    _merge_very_short_forward,
    _rows,
    _starts_with_orphan_bullet,
)


def _convert_platform_eng(
    tmp_path: Path, *, extra_env: dict[str, str] | None = None
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    pdf = Path("platform-eng-excerpt.pdf").resolve()
    spec = Path("pipeline.yaml").resolve()
    out = tmp_path / "out.jsonl"
    base_env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    if extra_env:
        base_env = {**base_env, **extra_env}
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
            "--no-enrich",
        ],
        check=True,
        cwd=tmp_path,
        env=base_env,
        capture_output=True,
        text=True,
    )
    return out, proc


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
                    "Most engineers don't want to learn a whole new toolset for infrequent tasks."
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


def test_flag_potential_duplicates():
    items = [
        {"text": " ".join(["word"] * 11) + "."},
        {"text": " ".join(["word"] * 11) + "."},
    ]
    assert _flag_potential_duplicates(items)


def test_split_does_not_duplicate(tmp_path: Path) -> None:
    out, proc = _convert_platform_eng(tmp_path, extra_env={"PDF_CHUNKER_DEDUP_DEBUG": "1"})
    text = out.read_text()
    matches = text.count("Most engineers")
    assert matches == 1, proc.stderr
    assert "Infrastructure setup" in text
    assert "dedupe dropped" in proc.stderr


def test_platform_eng_sentence_boundary(tmp_path: Path) -> None:
    out, _ = _convert_platform_eng(tmp_path)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    phrase = "It only works because you have one person"
    chunk = next(row for row in rows if phrase in row["text"])
    position = chunk["text"].find(phrase)
    assert position > 0
    prefix = chunk["text"][:position].rstrip()
    assert prefix and prefix[-1] in ".!?:;"
    chunk_id = chunk["metadata"]["chunk_id"]
    intro = next(
        row
        for row in rows
        if row["metadata"].get("chunk_id") == chunk_id and row["metadata"].get("chunk_part", 0) == 0
    )
    assert ":" in intro["text"]


def test_platform_eng_heading_preserved(tmp_path: Path) -> None:
    out, _ = _convert_platform_eng(tmp_path)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    heading = "How Platform Engineering Clears the Swamp"
    target = next(row for row in rows if "Clears the Swamp" in row["text"])
    assert heading in target["text"]
    assert target["text"].lstrip().startswith(heading)


def test_platform_eng_conjunction_chunks_have_context(tmp_path: Path) -> None:
    out, _ = _convert_platform_eng(tmp_path)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]

    def _lead_token(text: str) -> str:
        words = text.lstrip().split()
        return words[0].rstrip(",").lower() if words else ""

    blocked = {"but", "so", "however"}
    leads = {_lead_token(row["text"]) for row in rows}
    assert blocked.isdisjoint(leads)

    phrase = "But building platforms takes significant investment"
    chunk = next(row for row in rows if phrase in row["text"])
    assert not chunk["text"].lstrip().startswith(phrase)


def test_very_short_heading_merged_forward():
    """Very short items like orphaned headings merge into the following chunk."""
    items = [
        {"text": "Chapter One"},
        {"text": "This is the first paragraph of the chapter. " * 5},
    ]
    result = _merge_very_short_forward(items)
    # "Chapter One" (2 words) should merge into the next item
    assert len(result) == 1
    assert result[0]["text"].startswith("Chapter One")
    assert "first paragraph" in result[0]["text"]


def test_very_short_consecutive_headings_merge_forward():
    """Multiple consecutive short items all merge into the first long item."""
    items = [
        {"text": "Part I"},
        {"text": "Introduction"},
        {"text": "This is the actual content of the section. " * 5},
    ]
    result = _merge_very_short_forward(items)
    # Both short items should end up merged into the body
    assert len(result) == 1
    assert "Part I" in result[0]["text"]
    assert "Introduction" in result[0]["text"]
    assert "actual content" in result[0]["text"]


def test_very_short_trailing_item_merges_backward():
    """A very short item at the end merges into the previous chunk."""
    items = [
        {"text": "This is a complete paragraph with enough words to pass the threshold. " * 3},
        {"text": "The End."},
    ]
    result = _merge_very_short_forward(items)
    # "The End." should merge backward into the previous chunk
    assert len(result) == 1
    assert "The End." in result[0]["text"]
    assert "complete paragraph" in result[0]["text"]


def test_rows_merges_short_heading_in_preserve_mode():
    """Even when preserve=True (overlap>0), very short chunks are merged."""
    doc = {
        "type": "chunks",
        "items": [
            {"text": "Foreword"},
            {"text": "The following content is substantial enough to stand alone. " * 3},
        ],
    }
    rows = _rows(doc, preserve=True)
    # "Foreword" should be merged into the next chunk
    assert len(rows) == 1
    assert rows[0]["text"].startswith("Foreword")
    assert "substantial" in rows[0]["text"]


def test_orphan_bullet_detection():
    """Single bullet at start of text is detected as orphaned."""
    # Single bullet line - orphaned
    assert _starts_with_orphan_bullet("• First item only")
    assert _starts_with_orphan_bullet("1. First numbered item")

    # Bullet followed by non-bullet - orphaned
    assert _starts_with_orphan_bullet("• First item\nSome other text")
    assert _starts_with_orphan_bullet("1. First item\nSome paragraph follows")

    # Multiple bullets - NOT orphaned (proper list)
    assert not _starts_with_orphan_bullet("• First item\n• Second item")
    assert not _starts_with_orphan_bullet("1. First\n2. Second")

    # Non-bullet start - NOT orphaned
    assert not _starts_with_orphan_bullet("Regular paragraph text")
    assert not _starts_with_orphan_bullet("Heading\n• Then a bullet")


def test_orphan_bullet_merges_forward():
    """A chunk starting with a single orphaned bullet merges into next chunk."""
    items = [
        {"text": "• Single orphan bullet item"},
        {"text": "This is the main content of the section with enough words. " * 4},
    ]
    result = _merge_very_short_forward(items)
    # Orphan bullet should merge forward
    assert len(result) == 1
    assert "• Single orphan" in result[0]["text"]
    assert "main content" in result[0]["text"]


def test_proper_list_stays_intact():
    """A chunk with a proper multi-item list is NOT merged."""
    items = [
        {
            "text": "Here is a list:\n• First item\n• Second item\n• Third item with more words to reach threshold "
            * 2
        },
        {"text": "This is separate content that follows the list. " * 4},
    ]
    result = _merge_very_short_forward(items)
    # Proper list should stay separate
    assert len(result) == 2
    assert "• First item" in result[0]["text"]
    assert "• Second item" in result[0]["text"]


def test_critical_short_trailing_merges_even_when_large():
    """Critically short items (<5 words) merge even when prev is at max size."""
    from pdf_chunker.passes.emit_jsonl import _merge_short_rows

    # Create a row that's at max size (2000 chars)
    large_text = "word " * 400  # ~2000 chars
    rows = [
        {"text": large_text.strip()},
        {"text": "The End"},  # 2 words - critically short
    ]
    result = _merge_short_rows(rows)
    # Should merge despite exceeding soft limit
    assert len(result) == 1
    assert "The End" in result[0]["text"]
    assert result[0]["text"].startswith("word")


def test_merge_short_rows_respects_min_words():
    """Rows below min_row_words threshold are merged."""
    from pdf_chunker.passes.emit_jsonl import _merge_short_rows

    rows = [
        {"text": "Short heading"},  # 2 words
        {"text": "This is longer content with enough words to stand alone. " * 3},
    ]
    result = _merge_short_rows(rows)
    assert len(result) == 1
    assert "Short heading" in result[0]["text"]
    assert "longer content" in result[0]["text"]
