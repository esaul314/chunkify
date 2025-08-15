from __future__ import annotations

import json

import typer

from pdf_chunker.config import load_spec
from pdf_chunker.core_new import run_convert, run_inspect

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def convert(spec: str = "pipeline.yaml"):
    """
    Run the configured pipeline. This command does not perform IO; it only
    exercises registered passes over an empty Artifact until adapters exist.
    """
    s = load_spec(spec)
    _ = run_convert(s)  # result kept in memory for now
    typer.echo("convert: OK")


@app.command()
def inspect():
    """Print registered passes with their declared input/output types."""
    typer.echo(json.dumps(run_inspect(), indent=2))


if __name__ == "__main__":
    app()
