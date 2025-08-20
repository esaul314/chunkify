from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from typing import Iterable, Tuple
import argparse

BEGIN = "<!-- BEGIN AUTO-PASSES -->"
END = "<!-- END AUTO-PASSES -->"

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _pass_rows() -> list[Tuple[str, str, str]]:
    from pdf_chunker import passes

    def _doc(name: str) -> str:
        module = import_module(f"pdf_chunker.passes.{name}")
        doc = module.__doc__ or ""
        return doc.strip().splitlines()[0] if doc else ""

    return [(name, f"pdf_chunker.passes.{name}", _doc(name)) for name in sorted(passes.__all__)]


def _table(rows: Iterable[Tuple[str, str, str]]) -> str:
    header = "| Pass | Module | Responsibility |\n| --- | --- | --- |"
    fmt = lambda r: f"| `{r[0]}` | `{r[1]}` | {r[2]} |"
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
