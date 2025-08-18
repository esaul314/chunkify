from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Iterable, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pdf_chunker.framework import registry
import pdf_chunker.passes  # ensure registration side effects

START_MARK = "<!-- responsibilities-start -->"
END_MARK = "<!-- responsibilities-end -->"


def _pass_rows() -> Iterable[Tuple[str, str, str]]:
    reg = registry()
    doc = lambda o: inspect.getdoc(o) or inspect.getdoc(inspect.getmodule(o)) or ""
    first = lambda s: s.splitlines()[0] if s else ""
    return sorted((name, obj.__class__.__module__, first(doc(obj))) for name, obj in reg.items())


def _format_table(rows: Iterable[Tuple[str, str, str]]) -> str:
    header = "| Pass | Module | Responsibility |\n| --- | --- | --- |\n"
    body = "\n".join(f"| `{n}` | `{m}` | {d} |" for n, m, d in rows)
    return f"{header}{body}\n"


def _inject_table(text: str, table: str) -> str:
    before, _, rest = text.partition(START_MARK)
    _, _, after = rest.partition(END_MARK)
    return f"{before}{START_MARK}\n{table}{END_MARK}{after}"


def main(path: Path) -> None:
    updated = _inject_table(path.read_text(), _format_table(_pass_rows()))
    path.write_text(updated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate responsibilities table in AGENTS.md")
    parser.add_argument(
        "--agents-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "AGENTS.md",
        help="Path to AGENTS.md to update",
    )
    main(parser.parse_args().agents_path)
