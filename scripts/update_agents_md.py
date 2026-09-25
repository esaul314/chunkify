from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path

BEGIN = "<!-- BEGIN AUTO-PASSES -->"
END = "<!-- END AUTO-PASSES -->"

_PASS_RESPONSIBILITIES = {
    "ai_enrich": "Optionally enrich canonical chunks with metadata and tags.",
    "detect_doc_end": "Detect end-of-document markers and truncate trailing pages.",
    "emit_jsonl": "Normalize canonical chunks into JSONL-ready rows.",
    "epub_parse": "Normalize EPUB adapter output into canonical page blocks.",
    "extraction_fallback": "Select the best in-memory extraction candidate using quality metrics.",
    "heading_detect": "Annotate page blocks with heading metadata.",
    "merge_footers": "Merge short trailing footer lines into a stable block.",
    "pdf_parse": "Normalize PDF adapter output into canonical page blocks.",
    "text_clean": "Normalize text, whitespace, quotes, ligatures, and artifacts.",
    "text_parse": "Normalize plain-text transcript input into canonical page blocks.",
}

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _pass_rows() -> list[tuple[str, str, str]]:
    from pdf_chunker import passes

    def _doc(name: str) -> str:
        module = import_module(f"pdf_chunker.passes.{name}")
        doc = module.__doc__ or ""
        return doc.strip().splitlines()[0] if doc else _PASS_RESPONSIBILITIES.get(name, "")

    return [(name, f"pdf_chunker.passes.{name}", _doc(name)) for name in sorted(passes.__all__)]


def _table(rows: Iterable[tuple[str, str, str]]) -> str:
    header = "| Pass | Module | Responsibility |\n| --- | --- | --- |"

    def fmt(row: tuple[str, str, str]) -> str:
        return f"| `{row[0]}` | `{row[1]}` | {row[2]} |"

    return "\n".join([header, *map(fmt, rows), ""])


def _replace(md_path: Path, table: str) -> None:
    text = md_path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit("AGENTS.md missing auto-pass markers")
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    md_path.write_text("".join([pre, BEGIN, "\n", table, END, post]), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update pass table in AGENTS.md")
    parser.add_argument(
        "--md", type=Path, default=Path(__file__).resolve().parents[1] / "AGENTS.md"
    )
    md_path = parser.parse_args().md
    table = _table(_pass_rows())
    _replace(md_path, table)


if __name__ == "__main__":
    main()
