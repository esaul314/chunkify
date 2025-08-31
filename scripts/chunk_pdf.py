"""Deprecated shim for the old ``chunk_pdf`` CLI.

This module forwards all arguments to the new ``pdf_chunker`` entry point
while emitting a one‑time deprecation warning.  It exists solely to support
legacy automation scripts that still invoke ``scripts/chunk_pdf.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pdf_chunker import cli


def _delegate(argv: Sequence[str]) -> None:
    """Print a deprecation notice and invoke the modern CLI."""
    print("chunk_pdf.py is deprecated; use `pdf_chunker` instead.", file=sys.stderr)
    run = cli.app
    args = list(argv)
    run(args=args) if getattr(cli, "typer", None) else run(args)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point retaining the old script's interface."""
    _delegate(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":  # pragma: no cover - exercised in tests
    main()
