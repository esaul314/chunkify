"""Deprecated thin frontend for the canonical ``pdf_chunker convert`` command."""

from __future__ import annotations

import io
import sys
import tempfile
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

from pdf_chunker import cli


def _has_help(argv: Sequence[str]) -> bool:
    """Return ``True`` when a help flag is present."""
    return any(flag in argv for flag in ("-h", "--help"))


def _has_output(argv: Sequence[str]) -> bool:
    """Return whether the caller supplied an output path."""
    return any(arg in {"-o", "--out"} or arg.startswith("--out=") for arg in argv)


def _exit_code(exc: BaseException) -> int | None:
    """Return the exit code carried by ``exc`` when available."""

    if isinstance(exc, SystemExit):
        code = exc.code
        if code in (None, 0):
            return 0
        if isinstance(code, int):
            return code
        try:
            return int(code)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    for attr in ("exit_code", "code"):
        value = getattr(exc, attr, None)
        if value is None:
            continue
        if isinstance(value, int):
            return value
        if value == 0:
            return 0
    return None


def _invoke_cli(argv: Sequence[str]) -> int:
    """Execute the canonical CLI while normalising frontend exit codes."""

    args = list(argv)

    try:
        if getattr(cli, "typer", None):
            cli.app(args=args, standalone_mode=False)
        else:
            cli.app(args)
    except BaseException as exc:  # ``typer.Exit`` is not a ``SystemExit``
        code = _exit_code(exc)
        if code == 0:
            return 0
        raise
    return 0


def _delegate(argv: Sequence[str]) -> int:
    """Print a deprecation notice and invoke the modern CLI."""

    print("chunk_pdf.py is deprecated; use `pdf_chunker` instead.", file=sys.stderr)
    return _invoke_cli(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Retain the old entry point while sharing parsing and execution."""
    args = list(sys.argv[1:] if argv is None else argv)
    if _has_help(args):
        _delegate(["--help"])
        return

    temp_path: Path | None = None
    forwarded = ["convert", *args]
    if not _has_output(args):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            temp_path = Path(handle.name)
        forwarded.extend(("--out", str(temp_path)))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _delegate(forwarded)
    if temp_path is not None:
        print(temp_path.read_text(encoding="utf-8"), end="")
        temp_path.unlink()
    report = Path("run_report.json")
    if report.exists():
        report.unlink()


if __name__ == "__main__":  # pragma: no cover - exercised in tests
    main()
