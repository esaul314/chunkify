import json
import os
import subprocess
from pathlib import Path

from pdf_chunker.passes.emit_jsonl import _rows, _flag_potential_duplicates


def _convert_platform_eng(tmp_path: Path, *, extra_env: dict[str, str] | None = None) -> tuple[Path, subprocess.CompletedProcess[str]]:
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


def test_flag_potential_duplicates():
    items = [
        {"text": " ".join(["word"] * 11) + "."},
        {"text": " ".join(["word"] * 11) + "."},
    ]
    assert _flag_potential_duplicates(items)


def test_split_does_not_duplicate(tmp_path: Path) -> None:
    out, proc = _convert_platform_eng(
        tmp_path, extra_env={"PDF_CHUNKER_DEDUP_DEBUG": "1"}
    )
    text = out.read_text()
    matches = text.count("Most engineers")
    assert matches == 1, proc.stderr
    assert "Infrastructure setup" in text
    assert "dedupe dropped" in proc.stderr


def test_platform_eng_sentence_boundary(tmp_path: Path) -> None:
    out, _ = _convert_platform_eng(tmp_path)
    rows = [
        json.loads(line)
        for line in out.read_text().splitlines()
        if line.strip()
    ]
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
        if row["metadata"].get("chunk_id") == chunk_id
        and row["metadata"].get("chunk_part", 0) == 0
    )
    assert intro["text"].rstrip().endswith(":")


def test_platform_eng_heading_preserved(tmp_path: Path) -> None:
    out, _ = _convert_platform_eng(tmp_path)
    rows = [
        json.loads(line)
        for line in out.read_text().splitlines()
        if line.strip()
    ]
    heading = "How Platform Engineering Clears the Swamp"
    target = next(row for row in rows if "Clears the Swamp" in row["text"])
    assert heading in target["text"]
    assert target["text"].lstrip().startswith(heading)
