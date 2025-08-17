from __future__ import annotations

import json
from pathlib import Path

import typer

from pdf_chunker.adapters import emit_jsonl, io_pdf
from pdf_chunker.config import load_spec
from pdf_chunker.core_new import (
    assemble_report,
    run_convert,
    run_inspect,
    write_run_report,
)
from pdf_chunker.framework import Artifact

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _adapter_for(path: str):
    """Return IO adapter for ``path`` based on its extension."""
    ext = Path(path).suffix.lower()
    if ext == ".epub":
        from pdf_chunker.adapters import io_epub

        return io_epub
    return io_pdf


def _initial_artifact(path: str) -> Artifact:
    """Load ``path`` via adapter and wrap in an ``Artifact``."""
    adapter = _adapter_for(path)
    payload = adapter.read(path)
    return Artifact(payload=payload, meta={"metrics": {}, "input": path})


@app.command()
def convert(input_path: str, spec: str = "pipeline.yaml"):
    """Run the configured pipeline on ``input_path``."""
    s = load_spec(spec)
    a = _initial_artifact(input_path)
    a, timings = run_convert(a, s)
    emit_jsonl.maybe_write(a, s.options.get("emit_jsonl", {}), timings)
    report = assemble_report(timings, a.meta or {})
    write_run_report(s, report)
    typer.echo("convert: OK")


@app.command()
def inspect():
    """Print registered passes with their declared input/output types."""
    typer.echo(json.dumps(run_inspect(), indent=2))


if __name__ == "__main__":
    app()
