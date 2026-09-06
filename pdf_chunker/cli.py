from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

try:  # pragma: no cover - exercised via CLI tests
    typer = cast(Any, import_module("typer"))
except ModuleNotFoundError:  # pragma: no cover - fallback when Typer is absent
    typer = None

from pdf_chunker.config import PipelineSpec, load_spec


@dataclass(frozen=True)
class ConvertOptions:
    """Normalized conversion options shared by every CLI frontend."""

    input_path: Path
    out: Path | None = None
    metadata_file: Path | None = None
    chunk_size: int | None = None
    overlap: int | None = None
    enrich: bool = False
    exclude_pages: str | None = None
    exclude_spine: str | None = None
    no_metadata: bool = False
    spec: str = "pipeline.yaml"
    verbose: bool = False
    trace: str | None = None
    max_chars: int | None = None
    footer_patterns: tuple[str, ...] | None = None
    interactive: bool = False
    interactive_footers: bool = False
    interactive_lists: bool = False
    footer_margin: float | None = None
    header_margin: float | None = None
    auto_detect_zones: bool = False
    teach: bool = False


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
    resolved = next((p for p in _spec_path_candidates(path) if p.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"pipeline spec does not exist: {path}")
    return resolved


@contextmanager
def _temporary_environment(name: str, value: str):
    """Temporarily set an environment variable and restore its prior value."""
    original = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


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


def run_conversion(options: ConvertOptions) -> None:
    """Execute one normalized conversion request for every CLI frontend."""
    input_path = options.input_path
    out = options.out
    metadata_file = options.metadata_file
    chunk_size = options.chunk_size
    overlap = options.overlap
    enrich = options.enrich
    exclude_pages = options.exclude_pages
    exclude_spine = options.exclude_spine
    no_metadata = options.no_metadata
    spec = options.spec
    verbose = options.verbose
    trace = options.trace
    max_chars = options.max_chars
    footer_patterns = options.footer_patterns
    interactive = options.interactive
    interactive_footers = options.interactive_footers
    interactive_lists = options.interactive_lists
    footer_margin = options.footer_margin
    header_margin = options.header_margin
    auto_detect_zones = options.auto_detect_zones
    teach = options.teach

    # Resolve interactive flags: --interactive enables all
    effective_interactive_footers = interactive or interactive_footers

    # --teach implies --interactive
    if teach:
        interactive = True
        effective_interactive_footers = True

    # Load learned patterns if they exist
    learned_patterns = None
    if teach or interactive or interactive_footers or interactive_lists:
        from pdf_chunker.learned_patterns import LearnedPatterns

        learned_patterns = LearnedPatterns.load()

    if max_chars is not None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        if chunk_size is None:
            chunk_size = max_chars // 5
        if overlap is None:
            overlap = 0

    spec_path = _resolve_spec_path(spec)
    # Validate the spec and CLI overrides before opening or parsing the input.
    preflight_zones = {
        name: value
        for name, value in {
            "footer_margin": footer_margin,
            "header_margin": header_margin,
        }.items()
        if value is not None
    }
    preflight_overrides = _cli_overrides(
        input_path,
        out,
        metadata_file,
        chunk_size,
        overlap,
        enrich,
        exclude_pages,
        exclude_spine,
        no_metadata,
        footer_patterns=footer_patterns,
        interactive=interactive,
        interactive_footers=interactive_footers,
        interactive_lists=interactive_lists,
        zones_config=preflight_zones,
        learned_patterns=learned_patterns,
        teach=teach,
    )
    preflight_spec = load_spec(spec_path, overrides=preflight_overrides)
    from pdf_chunker.core_new import _enforce_invariants

    _enforce_invariants(preflight_spec, input_path=str(input_path))

    # Parse exclude_pages to a set for zone detection
    excluded_page_set: set[int] = set()
    if exclude_pages:
        from pdf_chunker.page_utils import parse_page_ranges

        excluded_page_set = parse_page_ranges(exclude_pages)

    # Auto-detect footer/header zones if requested (PDF only - EPUBs don't have zones)
    zones_config: dict[str, float] = {}
    is_epub = str(input_path).lower().endswith(".epub")
    if not is_epub and (
        auto_detect_zones or (effective_interactive_footers and footer_margin is None)
    ):
        try:
            import fitz

            from pdf_chunker.geometry import detect_document_zones, discover_zones_interactive

            doc = fitz.open(str(input_path))
            if effective_interactive_footers:
                # Interactive zone discovery - respects page exclusions
                zones = discover_zones_interactive(
                    doc,
                    exclude_pages=excluded_page_set,
                )
            else:
                # Automatic detection - respects page exclusions
                zones = detect_document_zones(
                    doc,
                    exclude_pages=excluded_page_set,
                )
            doc.close()

            if zones.footer_margin:
                zones_config["footer_margin"] = zones.footer_margin
                if not effective_interactive_footers:
                    conf = zones.confidence
                    margin = zones.footer_margin
                    print(f"Auto-detected footer margin: {margin:.1f}pt (confidence: {conf:.0%})")
            if zones.header_margin:
                zones_config["header_margin"] = zones.header_margin
        except ImportError:
            pass  # fitz not available
        except Exception as e:
            if effective_interactive_footers:
                print(f"Zone detection failed: {e}")

    # CLI overrides take precedence over auto-detection
    if footer_margin is not None:
        zones_config["footer_margin"] = footer_margin
    if header_margin is not None:
        zones_config["header_margin"] = header_margin

    s = load_spec(
        spec_path,
        overrides=_cli_overrides(
            input_path,
            out,
            metadata_file,
            chunk_size,
            overlap,
            enrich,
            exclude_pages,
            exclude_spine,
            no_metadata,
            footer_patterns=footer_patterns,
            interactive=interactive,
            interactive_footers=interactive_footers,
            interactive_lists=interactive_lists,
            zones_config=zones_config,
            learned_patterns=learned_patterns,
            teach=teach,
        ),
    )
    s = _enrich_spec(s) if enrich else s
    from pdf_chunker.core_new import _input_artifact, run_convert

    env = (
        _temporary_environment("PDF_CHUNKER_JSONL_MAX_CHARS", str(max_chars))
        if max_chars is not None
        else nullcontext()
    )
    with env:
        _, timings = run_convert(_input_artifact(str(input_path), s), s, trace=trace)
        if verbose:
            print(_format_timings(timings))

        # Save learned patterns if --teach mode was enabled
        if teach and learned_patterns is not None:
            learned_patterns.save()
            count = len(learned_patterns.patterns)
            if count > 0:
                print(f"teach: saved {count} learned pattern(s)")

        print("convert: OK")


def _run_convert(
    input_path: Path,
    out: Path | None,
    metadata_file: Path | None,
    chunk_size: int | None,
    overlap: int | None,
    enrich: bool,
    exclude_pages: str | None,
    exclude_spine: str | None,
    no_metadata: bool,
    spec: str,
    verbose: bool,
    trace: str | None,
    max_chars: int | None,
    footer_patterns: tuple[str, ...] | None = None,
    interactive: bool = False,
    interactive_footers: bool = False,
    interactive_lists: bool = False,
    footer_margin: float | None = None,
    header_margin: float | None = None,
    auto_detect_zones: bool = False,
    teach: bool = False,
) -> None:
    """Compatibility adapter for callers using the former positional helper."""
    run_conversion(
        ConvertOptions(
            input_path=input_path,
            out=out,
            metadata_file=metadata_file,
            chunk_size=chunk_size,
            overlap=overlap,
            enrich=enrich,
            exclude_pages=exclude_pages,
            exclude_spine=exclude_spine,
            no_metadata=no_metadata,
            spec=spec,
            verbose=verbose,
            trace=trace,
            max_chars=max_chars,
            footer_patterns=footer_patterns,
            interactive=interactive,
            interactive_footers=interactive_footers,
            interactive_lists=interactive_lists,
            footer_margin=footer_margin,
            header_margin=header_margin,
            auto_detect_zones=auto_detect_zones,
            teach=teach,
        )
    )


def _run_inspect() -> None:
    from pdf_chunker.core_new import run_inspect

    print(json.dumps(run_inspect(), indent=2))


def _run_list_spines(input_path: Path) -> None:
    """List EPUB spine items with indices."""
    if not str(input_path).lower().endswith(".epub"):
        print("Error: list-spines only works with EPUB files", file=sys.stderr)
        sys.exit(1)

    from pdf_chunker.epub_parsing import list_epub_spines

    spines = list_epub_spines(str(input_path))
    print(f"\nSpine items in {input_path.name}:\n")
    for item in spines:
        idx = item["index"]
        filename = item["filename"]
        preview = item["content_preview"]
        print(f"  {idx:3d}. {filename}")
        print(f"       {preview}\n")
    print("Use --exclude-spine to skip items, e.g.: --exclude-spine '1,2'")


app = typer.Typer(add_completion=False, no_args_is_help=True) if typer else None


def _core_helpers(
    enrich: bool,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Return canonical runner helpers without mutating the import registry."""
    del enrich
    from pdf_chunker.core_new import _input_artifact, run_convert, run_inspect

    return _input_artifact, run_convert, run_inspect


def _cli_overrides(
    input_path: Path = Path("input.pdf"),
    out: Path | None = None,
    metadata_file: Path | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    enrich: bool = False,
    exclude_pages: str | None = None,
    exclude_spine: str | None = None,
    no_metadata: bool = False,
    *,
    footer_patterns: tuple[str, ...] | None = None,
    interactive: bool = False,
    interactive_footers: bool = False,
    interactive_lists: bool = False,
    zones_config: dict[str, float] | None = None,
    learned_patterns: Any = None,
    teach: bool = False,
) -> dict[str, dict[str, Any]]:
    suffix = input_path.suffix.lower()
    parser_option_key = {
        ".epub": "epub_parse",
        ".txt": "text_parse",
    }.get(suffix, "pdf_parse")
    # Resolve interactive flags: --interactive enables all, specific flags override
    effective_interactive_footers = interactive or interactive_footers
    effective_interactive_lists = interactive or interactive_lists

    footer_callback = None
    if effective_interactive_footers:
        from pdf_chunker.interactive import make_cli_footer_prompt

        footer_callback = make_cli_footer_prompt(default_is_footer=True)
        if learned_patterns is not None:
            from pdf_chunker.interactive import make_learned_footer_callback

            footer_callback = make_learned_footer_callback(
                footer_callback,
                learned_patterns,
                teach=teach,
                source_file=str(input_path),
            )
    list_continuation_callback = None
    if effective_interactive_lists:
        from pdf_chunker.interactive import make_cli_list_continuation_prompt

        list_continuation_callback = make_cli_list_continuation_prompt()
        if learned_patterns is not None:
            from pdf_chunker.interactive import make_learned_list_continuation_callback

            list_continuation_callback = make_learned_list_continuation_callback(
                list_continuation_callback,
                learned_patterns,
                teach=teach,
                source_file=str(input_path),
            )

    split_opts: dict[str, Any] = {
        k: v
        for k, v in {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "generate_metadata": False if no_metadata else None,
            "interactive_lists": True if effective_interactive_lists else None,
            "list_continuation_callback": list_continuation_callback,
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
    parse_pairs: dict[str, Any] = {}
    if parser_option_key == "pdf_parse" and exclude_pages is not None:
        parse_pairs["exclude_pages"] = exclude_pages
    if parser_option_key == "epub_parse" and exclude_spine is not None:
        parse_pairs["exclude_spine"] = exclude_spine
    if metadata_file is not None:
        parse_pairs["metadata_path"] = str(metadata_file)
    parse_opts = parse_pairs
    # Add zone exclusion margins to pdf_parse options
    if zones_config and parser_option_key == "pdf_parse":
        parse_opts.update(zones_config)
    # Footer detection options (apply even if pass not in pipeline)
    artifact_opts: dict[str, Any] = {}
    if footer_patterns:
        artifact_opts["known_footer_patterns"] = footer_patterns
    if effective_interactive_footers:
        artifact_opts["interactive"] = True
        artifact_opts["footer_callback"] = footer_callback

    # Build options, including artifact_opts in base even if pass isn't in pipeline
    base_opts = {
        "split_semantic": split_opts,
        "emit_jsonl": emit_opts,
        "ai_enrich": enrich_opts,
        parser_option_key: parse_opts,
    }
    if artifact_opts:
        # Merge into text_clean options for footer pattern handling
        base_opts["text_clean"] = {
            **base_opts.get("text_clean", {}),
            "footer_patterns": artifact_opts.get("known_footer_patterns", ()),
            "interactive_footers": artifact_opts.get("interactive", False),
            "footer_callback": artifact_opts.get("footer_callback"),
        }
    return {k: v for k, v in base_opts.items() if v}


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
        out: Path | None = typer.Option(None, "--out", "-o"),
        metadata_file: Path | None = typer.Option(
            None,
            "--metadata-file",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON or YAML metadata to propagate to every emitted chunk",
        ),
        chunk_size: int | None = typer.Option(None, "--chunk-size"),
        overlap: int | None = typer.Option(None, "--overlap"),
        enrich: bool = typer.Option(False, "--enrich/--no-enrich"),
        exclude_pages: str | None = typer.Option(None, "--exclude-pages"),
        exclude_spine: str | None = typer.Option(
            None,
            "--exclude-spine",
            help="EPUB spine items to exclude (e.g., '1,2' for ToC). Use list-spines to discover.",
        ),
        no_metadata: bool = typer.Option(False, "--no-metadata"),
        spec: str = typer.Option("pipeline.yaml", "--spec"),
        verbose: bool = typer.Option(False, "--verbose"),
        trace: str | None = typer.Option(None, "--trace"),
        max_chars: int | None = typer.Option(None, "--max-chars"),
        footer_pattern: list[str] | None = typer.Option(
            None,
            "--footer-pattern",
            help="Regex pattern for known footers (repeatable)",
        ),
        interactive: bool = typer.Option(
            False,
            "--interactive",
            help="Enable all interactive prompts (footers and lists)",
        ),
        interactive_footers: bool = typer.Option(
            False,
            "--interactive-footers",
            help="Prompt for confirmation on ambiguous footers only",
        ),
        interactive_lists: bool = typer.Option(
            False,
            "--interactive-lists",
            help="Prompt for confirmation on ambiguous list continuations only",
        ),
        footer_margin: float | None = typer.Option(
            None,
            "--footer-margin",
            help="Footer zone margin in points from page bottom",
        ),
        header_margin: float | None = typer.Option(
            None,
            "--header-margin",
            help="Header zone margin in points from page top",
        ),
        auto_detect_zones: bool = typer.Option(
            False,
            "--auto-detect-zones",
            help="Auto-detect header/footer zones using geometry",
        ),
        teach: bool = typer.Option(
            False,
            "--teach",
            help="Save interactive decisions for future runs (implies --interactive)",
        ),
    ) -> None:
        _safe(
            lambda: run_conversion(
                ConvertOptions(
                    input_path=input_path,
                    out=out,
                    metadata_file=metadata_file,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    enrich=enrich,
                    exclude_pages=exclude_pages,
                    exclude_spine=exclude_spine,
                    no_metadata=no_metadata,
                    spec=spec,
                    verbose=verbose,
                    trace=trace,
                    max_chars=max_chars,
                    footer_patterns=tuple(footer_pattern) if footer_pattern else None,
                    interactive=interactive,
                    interactive_footers=interactive_footers,
                    interactive_lists=interactive_lists,
                    footer_margin=footer_margin,
                    header_margin=header_margin,
                    auto_detect_zones=auto_detect_zones,
                    teach=teach,
                )
            )
        )

    @app.command(name="list-spines")
    def list_spines(
        input_path: Path = typer.Argument(
            ...,
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ) -> None:
        """List EPUB spine items with indices for use with --exclude-spine."""
        if not str(input_path).lower().endswith(".epub"):
            print("Error: list-spines only works with EPUB files", file=sys.stderr)
            raise typer.Exit(1)

        from pdf_chunker.epub_parsing import list_epub_spines

        spines = list_epub_spines(str(input_path))
        print(f"\nSpine items in {input_path.name}:\n")
        for item in spines:
            idx = item["index"]
            filename = item["filename"]
            preview = item["content_preview"]
            print(f"  {idx:3d}. {filename}")
            print(f"       {preview}\n")
        print("Use --exclude-spine to skip items, e.g.: --exclude-spine '1,2'")

    @app.command()
    def inspect() -> None:  # pragma: no cover - exercised in tests
        _run_inspect()

else:

    def app(argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(prog="pdf_chunker")
        sub = parser.add_subparsers(dest="cmd", required=True)

        conv = sub.add_parser("convert")
        conv.add_argument("input_path", type=Path)
        conv.add_argument("-o", "--out", type=Path)
        conv.add_argument("--metadata-file", type=Path)
        conv.add_argument("--chunk-size", type=int)
        conv.add_argument("--overlap", type=int)
        conv.add_argument("--enrich", dest="enrich", action="store_true")
        conv.add_argument("--no-enrich", dest="enrich", action="store_false")
        conv.add_argument("--exclude-pages")
        conv.add_argument(
            "--exclude-spine",
            help="EPUB spine items to exclude (e.g., '1,2' for ToC)",
        )
        conv.add_argument("--no-metadata", action="store_true")
        conv.add_argument("--spec", default="pipeline.yaml")
        conv.add_argument("--verbose", action="store_true")
        conv.add_argument("--trace")
        conv.add_argument("--max-chars", type=int)
        conv.add_argument(
            "--footer-pattern",
            action="append",
            dest="footer_patterns",
            help="Regex pattern for known footers (repeatable)",
        )
        conv.add_argument(
            "--interactive",
            action="store_true",
            help="Enable all interactive prompts (footers and lists)",
        )
        conv.add_argument(
            "--interactive-footers",
            action="store_true",
            help="Prompt for confirmation on ambiguous footers only",
        )
        conv.add_argument(
            "--interactive-lists",
            action="store_true",
            help="Prompt for confirmation on ambiguous list continuations only",
        )
        conv.add_argument(
            "--footer-margin",
            type=float,
            help="Footer zone margin in points from page bottom",
        )
        conv.add_argument(
            "--header-margin",
            type=float,
            help="Header zone margin in points from page top",
        )
        conv.add_argument(
            "--auto-detect-zones",
            action="store_true",
            help="Auto-detect header/footer zones using geometry",
        )
        conv.add_argument(
            "--teach",
            action="store_true",
            help="Save interactive decisions for future runs (implies --interactive)",
        )
        conv.set_defaults(
            enrich=False,
            exclude_spine=None,
            footer_patterns=None,
            interactive=False,
            interactive_footers=False,
            interactive_lists=False,
            footer_margin=None,
            header_margin=None,
            auto_detect_zones=False,
            teach=False,
            func=lambda ns: _safe(
                lambda: run_conversion(
                    ConvertOptions(
                        input_path=ns.input_path,
                        out=ns.out,
                        metadata_file=ns.metadata_file,
                        chunk_size=ns.chunk_size,
                        overlap=ns.overlap,
                        enrich=ns.enrich,
                        exclude_pages=ns.exclude_pages,
                        exclude_spine=ns.exclude_spine,
                        no_metadata=ns.no_metadata,
                        spec=ns.spec,
                        verbose=ns.verbose,
                        trace=ns.trace,
                        max_chars=ns.max_chars,
                        footer_patterns=tuple(ns.footer_patterns) if ns.footer_patterns else None,
                        interactive=ns.interactive,
                        interactive_footers=ns.interactive_footers,
                        interactive_lists=ns.interactive_lists,
                        footer_margin=ns.footer_margin,
                        header_margin=ns.header_margin,
                        auto_detect_zones=ns.auto_detect_zones,
                        teach=ns.teach,
                    )
                )
            ),
        )

        # list-spines command for EPUB
        list_sp = sub.add_parser("list-spines", help="List EPUB spine items")
        list_sp.add_argument("input_path", type=Path)
        list_sp.set_defaults(
            func=lambda ns: _run_list_spines(ns.input_path),
        )

        insp = sub.add_parser("inspect")
        insp.set_defaults(func=lambda ns: _run_inspect())

        args = parser.parse_args(argv)
        args.func(args)


if __name__ == "__main__":
    app()
