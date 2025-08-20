from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from subprocess import run
from typing import Callable, Sequence
from unittest.mock import patch

from scripts import chunk_pdf


def _run_legacy(pdf: Path, out_path: Path, flags: Sequence[str] = ()) -> Path:
    argv = ["chunk_pdf.py", str(pdf), *flags]
    with (
        out_path.open("w", encoding="utf-8") as f,
        redirect_stdout(f),
        patch.object(sys, "argv", argv),
    ):
        chunk_pdf.main()
    return out_path


def _run_new(pdf: Path, out_path: Path, flags: Sequence[str] = ()) -> Path:
    run(
        [
            "pdf_chunker",
            "convert",
            str(pdf),
            *flags,
            "--out",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def run_parity(pdf: Path, tmpdir: Path, flags: Sequence[str] = ()) -> tuple[Path, Path]:
    tmpdir.mkdir(parents=True, exist_ok=True)
    runners: tuple[Callable[[], Path], ...] = (
        partial(_run_legacy, pdf, tmpdir / "legacy.jsonl", flags),
        partial(_run_new, pdf, tmpdir / "new.jsonl", flags),
    )
    return tuple(map(lambda fn: fn(), runners))  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legacy and new pipelines")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("tmpdir", type=Path)
    args = parser.parse_args()
    legacy, new = run_parity(args.pdf, args.tmpdir)
    for p in (legacy, new):
        print(p)


if __name__ == "__main__":
    main()
