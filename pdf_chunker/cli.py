from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable

import importlib.util as ilu
import sys
import types
import typer

from pdf_chunker.config import PipelineSpec, load_spec


def _spec_path_candidates(path: str | Path) -> Iterator[Path]:
    """Yield potential spec locations without hitting the filesystem."""
    candidate = Path(path)
    pkg_dir = Path(__file__).resolve().parent
    yield from (
        candidate,
        pkg_dir.parent / candidate,
        pkg_dir / candidate,
    )


def _resolve_spec_path(path: str | Path) -> Path:
    """Pick the first existing pipeline spec from candidate locations."""
    return next((p for p in _spec_path_candidates(path) if p.exists()), Path(path))


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _core_helpers(
    enrich: bool,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Load core functions while stubbing AI adapters when ``enrich`` is False."""
    if enrich:
        from pdf_chunker.core_new import (
            _input_artifact,
            run_convert,
            run_inspect,
        )  # pragma: no cover

        return _input_artifact, run_convert, run_inspect

    base = Path(__file__).resolve().parent / "adapters"
    pkg = types.ModuleType("pdf_chunker.adapters")
    pkg.__path__ = [str(base)]
    sys.modules.setdefault("pdf_chunker.adapters", pkg)

    def _load(name: str) -> None:
        spec = ilu.spec_from_file_location(f"pdf_chunker.adapters.{name}", base / f"{name}.py")
        mod = ilu.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(mod)
        sys.modules[f"pdf_chunker.adapters.{name}"] = mod
        setattr(pkg, name, mod)

    tuple(_load(n) for n in ("emit_jsonl", "io_pdf", "io_epub"))
    from pdf_chunker.core_new import _input_artifact, run_convert, run_inspect

    return _input_artifact, run_convert, run_inspect


def _cli_overrides(
    out: Path | None,
    chunk_size: int | None,
    overlap: int | None,
    enrich: bool,
    exclude_pages: str | None,
    no_metadata: bool,
) -> dict[str, dict[str, Any]]:
    split_opts = {
        k: v
        for k, v in {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "generate_metadata": False if no_metadata else None,
        }.items()
        if v is not None
    }
    emit_opts = {
        k: v
        for k, v in {
            "output_path": str(out) if out else None,
            "drop_meta": True if no_metadata else None,
        }.items()
        if v is not None
    }
    enrich_opts = {"enabled": True} if enrich else {}
    parse_opts = {"exclude_pages": exclude_pages} if exclude_pages else {}
    return {
        k: v
        for k, v in {
            "split_semantic": split_opts,
            "emit_jsonl": emit_opts,
            "ai_enrich": enrich_opts,
            "pdf_parse": parse_opts,
        }.items()
        if v
    }


def _enrich_spec(spec: PipelineSpec) -> PipelineSpec:
    """Insert ``ai_enrich`` before ``emit_jsonl`` without mutating ``spec``."""
    return spec.model_copy(
        update={
            "pipeline": [
                step
                for p in spec.pipeline
                for step in (["ai_enrich", "emit_jsonl"] if p == "emit_jsonl" else [p])
            ]
        }
    )


@app.command()
def convert(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    out: Path | None = typer.Option(None, "--out"),
    chunk_size: int | None = typer.Option(None, "--chunk-size"),
    overlap: int | None = typer.Option(None, "--overlap"),
    enrich: bool = typer.Option(False, "--enrich/--no-enrich"),
    exclude_pages: str | None = typer.Option(None, "--exclude-pages"),
    no_metadata: bool = typer.Option(False, "--no-metadata"),
    spec: str = "pipeline.yaml",
):
    """Run the configured pipeline on ``input_path``."""
    _input_artifact, run_convert, _ = _core_helpers(enrich)
    s = load_spec(
        _resolve_spec_path(spec),
        overrides=_cli_overrides(out, chunk_size, overlap, enrich, exclude_pages, no_metadata),
    )
    s = _enrich_spec(s) if enrich else s
    run_convert(_input_artifact(str(input_path), s), s)
    typer.echo("convert: OK")


@app.command()
def inspect():
    """Print registered passes with their declared input/output types."""
    _, _, run_inspect = _core_helpers(False)
    typer.echo(json.dumps(run_inspect(), indent=2))


if __name__ == "__main__":
    app()
