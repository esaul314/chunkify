from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import typer

from pdf_chunker.config import load_spec
from pdf_chunker.core_new import _input_artifact, run_convert, run_inspect

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _cli_overrides(
    out: Path | None,
    chunk_size: int | None,
    overlap: int | None,
    enrich: bool,
) -> Dict[str, Dict[str, Any]]:
    split_opts = {
        k: v for k, v in {"chunk_size": chunk_size, "overlap": overlap}.items() if v is not None
    }
    emit_opts = {"output_path": str(out)} if out else {}
    enrich_opts = {"enabled": True} if enrich else {}
    return {
        k: v
        for k, v in {
            "split_semantic": split_opts,
            "emit_jsonl": emit_opts,
            "ai_enrich": enrich_opts,
        }.items()
        if v
    }


@app.command()
def convert(
    input_path: str,
    out: Path | None = typer.Option(None, "--out"),
    chunk_size: int | None = typer.Option(None, "--chunk-size"),
    overlap: int | None = typer.Option(None, "--overlap"),
    enrich: bool = typer.Option(False, "--enrich/--no-enrich"),
    spec: str = "pipeline.yaml",
):
    """Run the configured pipeline on ``input_path``."""
    s = load_spec(spec, overrides=_cli_overrides(out, chunk_size, overlap, enrich))
    if enrich:
        s["pipeline"] = [
            step
            for p in s["pipeline"]
            for step in (["ai_enrich", "emit_jsonl"] if p == "emit_jsonl" else [p])
        ]
    run_convert(_input_artifact(input_path), s)
    typer.echo("convert: OK")


@app.command()
def inspect():
    """Print registered passes with their declared input/output types."""
    typer.echo(json.dumps(run_inspect(), indent=2))


if __name__ == "__main__":
    app()
