"""Deprecated shim for the old ``chunk_pdf`` CLI.

Accepts the historical flag set and forwards them to the modern
``pdf_chunker`` command while emitting a deprecation notice.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
import io
import tempfile

from pdf_chunker import cli


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chunk_pdf.py")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("-o", "--out", type=Path)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--exclude-pages")
    parser.add_argument("--no-metadata", action="store_true")
    return parser


def _to_cli_args(ns: argparse.Namespace) -> list[str]:
    pairs = {
        "--out": ns.out,
        "--chunk-size": ns.chunk_size,
        "--overlap": ns.overlap,
        "--exclude-pages": ns.exclude_pages,
    }
    flags = [item for k, v in pairs.items() for item in (k, str(v)) if v is not None]
    bools = ["--no-metadata"] if ns.no_metadata else []
    return ["convert", str(ns.input_path), *flags, *bools]


def _delegate(argv: Sequence[str]) -> None:
    """Print a deprecation notice and invoke the modern CLI."""
    print("chunk_pdf.py is deprecated; use `pdf_chunker` instead.", file=sys.stderr)
    run = cli.app
    args = list(argv)
    if getattr(cli, "typer", None):
        run(args=args, standalone_mode=False)
    else:
        run(args)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point retaining the old script's interface."""
    ns = _parser().parse_args(argv)
    out = ns.out
    tmp: Path | None = None
    if out is None:
        handle = tempfile.NamedTemporaryFile(delete=False)
        tmp = Path(handle.name)
        handle.close()
        ns.out = tmp
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _delegate(_to_cli_args(ns))
    if tmp is not None:
        print(tmp.read_text(encoding="utf-8"), end="")
        tmp.unlink()
    report = Path("run_report.json")
    if report.exists():
        report.unlink()


if __name__ == "__main__":  # pragma: no cover - exercised in tests
    main()
