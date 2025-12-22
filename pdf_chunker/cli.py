from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import os
import sys
import types
from collections.abc import Callable, Iterator, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, cast


try:  # pragma: no cover - exercised via CLI tests
    typer = cast(Any, import_module("typer"))
except ModuleNotFoundError:  # pragma: no cover - fallback when Typer is absent
    typer = None

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


def _format_timings(timings: Mapping[str, float]) -> str:
    """Return ``timings`` as newline-delimited ``name: seconds`` strings."""
    return "\n".join(f"{n}: {t:.2f}s" for n, t in timings.items())


def _exit_with_error(exc: Exception) -> None:
    """Print ``exc`` to stderr and exit with status 1."""
    print(f"error: {exc}", file=sys.stderr)
    raise typer.Exit(1) if typer else SystemExit(1)


def _safe(func: Callable[[], None]) -> None:
    """Invoke ``func`` and exit non-zero on any exception."""
    try:
        func()
    except Exception as exc:  # pragma: no cover - exercised in CLI tests
        _exit_with_error(exc)


def _run_convert(
    input_path: Path,
    out: Path | None,
    chunk_size: int | None,
    overlap: int | None,
    enrich: bool,
    exclude_pages: str | None,
    no_metadata: bool,
    spec: str,
    verbose: bool,
    trace: str | None,
    max_chars: int | None,
) -> None:
    if max_chars:
        os.environ["PDF_CHUNKER_JSONL_MAX_CHARS"] = str(max_chars)
        if chunk_size is None:
            chunk_size = max_chars // 5
        if overlap is None:
            overlap = 0

    _input_artifact, run_convert, _ = _core_helpers(enrich)
    s = load_spec(
        _resolve_spec_path(spec),
        overrides=_cli_overrides(
            out,
            chunk_size,
            overlap,
            enrich,
            exclude_pages,
            no_metadata,
        ),
    )
    s = _enrich_spec(s) if enrich else s
    _, timings = run_convert(_input_artifact(str(input_path), s), s, trace=trace)
    if verbose:
        print(_format_timings(timings))
    print("convert: OK")


def _run_inspect() -> None:
    _, _, run_inspect = _core_helpers(False)
    print(json.dumps(run_inspect(), indent=2))


app = typer.Typer(add_completion=False, no_args_is_help=True) if typer else None


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
        spec = ilu.spec_from_file_location(
            f"pdf_chunker.adapters.{name}",
            base / f"{name}.py",
        )
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load adapter module: {name}")
        module = ilu.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[f"pdf_chunker.adapters.{name}"] = module
        setattr(pkg, name, module)

    for name in ("emit_jsonl", "io_pdf", "io_epub"):
        _load(name)
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
    split_opts: dict[str, Any] = {
        k: v
        for k, v in {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "generate_metadata": False if no_metadata else None,
        }.items()
        if v is not None
    }
    emit_opts: dict[str, Any] = {
        k: v
        for k, v in {
            "output_path": str(out) if out else None,
            "drop_meta": True if no_metadata else None,
        }.items()
        if v is not None
    }
    enrich_opts: dict[str, Any] = {"enabled": True} if enrich else {}
    parse_opts: dict[str, Any] = {
        k: v for k, v in {"exclude_pages": exclude_pages}.items() if v is not None
    }
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


if typer:
    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command()
    def convert(  # pragma: no cover - exercised in parity tests
        input_path: Path = typer.Argument(
            ...,
            exists=True,
            dir_okay=False,
            readable=True,
        ),
        out: Path | None = typer.Option(None, "--out"),
        chunk_size: int | None = typer.Option(None, "--chunk-size"),
        overlap: int | None = typer.Option(None, "--overlap"),
        enrich: bool = typer.Option(False, "--enrich/--no-enrich"),
        exclude_pages: str | None = typer.Option(None, "--exclude-pages"),
        no_metadata: bool = typer.Option(False, "--no-metadata"),
        spec: str = typer.Option("pipeline.yaml", "--spec"),
        verbose: bool = typer.Option(False, "--verbose"),
        trace: str | None = typer.Option(None, "--trace"),
        max_chars: int | None = typer.Option(None, "--max-chars"),
    ) -> None:
        _safe(
            lambda: _run_convert(
                input_path,
                out,
                chunk_size,
                overlap,
                enrich,
                exclude_pages,
                no_metadata,
                spec,
                verbose,
                trace,
                max_chars,
            )
        )

    @app.command()
    def inspect() -> None:  # pragma: no cover - exercised in tests
        _run_inspect()

else:

    def app(argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(prog="pdf_chunker")
        sub = parser.add_subparsers(dest="cmd", required=True)

        conv = sub.add_parser("convert")
        conv.add_argument("input_path", type=Path)
        conv.add_argument("--out", type=Path)
        conv.add_argument("--chunk-size", type=int)
        conv.add_argument("--overlap", type=int)
        conv.add_argument("--enrich", dest="enrich", action="store_true")
        conv.add_argument("--no-enrich", dest="enrich", action="store_false")
        conv.add_argument("--exclude-pages")
        conv.add_argument("--no-metadata", action="store_true")
        conv.add_argument("--spec", default="pipeline.yaml")
        conv.add_argument("--verbose", action="store_true")
        conv.add_argument("--trace")
        conv.add_argument("--max-chars", type=int)
        conv.set_defaults(
            enrich=False,
            func=lambda ns: _safe(
                lambda: _run_convert(
                    ns.input_path,
                    ns.out,
                    ns.chunk_size,
                    ns.overlap,
                    ns.enrich,
                    ns.exclude_pages,
                    ns.no_metadata,
                    ns.spec,
                    ns.verbose,
                    ns.trace,
                    ns.max_chars,
                )
            ),
        )

        insp = sub.add_parser("inspect")
        insp.set_defaults(func=lambda ns: _run_inspect())

        args = parser.parse_args(argv)
        args.func(args)


if __name__ == "__main__":
    app()
