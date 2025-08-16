from __future__ import annotations

import json

import typer

from pdf_chunker.config import load_spec
from pdf_chunker.core_new import run_convert, run_inspect

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def convert(input_path: str, spec: str = "pipeline.yaml"):
    """Run the configured pipeline on ``input_path``."""
    s = load_spec(spec)
    _ = run_convert(input_path, s)
    typer.echo("convert: OK")


@app.command()
def inspect():
    """Print registered passes with their declared input/output types."""
    typer.echo(json.dumps(run_inspect(), indent=2))


if __name__ == "__main__":
    app()
