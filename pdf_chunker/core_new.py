from __future__ import annotations

import inspect
import json
import platform
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from datetime import date, datetime
from functools import reduce
from hashlib import md5
from importlib import import_module
from itertools import chain
from pathlib import Path
from typing import Any, Final, cast

import pdf_chunker.adapters.emit_trace as emit_trace
from pdf_chunker.adapters import emit_jsonl
from pdf_chunker.config import PipelineSpec, validate_options
from pdf_chunker.contracts import Chunk
from pdf_chunker.framework import Artifact, Pass, registry


class UnsupportedFormatError(ValueError):
    """Raised when no reader adapter is available for an input suffix."""


class PipelineContractError(ValueError):
    """Raised when a pass receives or returns the wrong canonical payload."""


_PARSER_MAP: Final[Mapping[str, str]] = {
    ext: name
    for name, exts in {
        "pdf_parse": (".pdf",),
        "epub_parse": (".epub",),
        "text_parse": (".txt",),
    }.items()
    for ext in exts
}
_TEXT_UNSUPPORTED_PASSES: Final[frozenset[str]] = frozenset(
    {"detect_page_artifacts", "merge_footers"}
)


def choose_parser(suffix: str) -> str:
    """Return parser step for ``suffix`` using a lookup table."""
    normalized = suffix.lower()
    try:
        return _PARSER_MAP[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_PARSER_MAP))
        raise UnsupportedFormatError(
            f"unsupported input format {normalized!r}; supported formats: {supported}"
        ) from exc


def _pass_steps(spec: PipelineSpec) -> list[str]:
    """Filter pipeline steps to registered passes; error on unknown ones."""
    regs = registry()
    unknown = [s for s in spec.pipeline if s not in regs and s != "emit_jsonl"]
    if unknown:
        available = ", ".join(sorted(regs)) or "<none>"
        raise KeyError(f"unknown pipeline steps: {unknown}; available: {available}")
    return [s for s in spec.pipeline if s in regs]


def _ensure_clean_precedes_split(steps: Sequence[str]) -> None:
    """Raise descriptive error if a split pass precedes ``text_clean``."""
    first_split = next(
        ((s, i) for i, s in enumerate(steps) if s.startswith("split")),
        None,
    )
    if not first_split:
        return
    clean_index = next((i for i, s in enumerate(steps) if s == "text_clean"), None)
    split_name, split_index = first_split
    if clean_index is None or clean_index > split_index:
        raise ValueError(f"{split_name} requires text_clean to run beforehand")


def _ensure_pdf_epub_separation(steps: Sequence[str], ext: str) -> list[str]:
    """Return ``steps`` with parse step aligned to ``ext``; reject mixed media."""
    parser_steps = {"pdf_parse", "epub_parse", "text_parse"}
    prefixes = {s.split("_", 1)[0] for s in steps if s in parser_steps}
    if len(prefixes) > 1:
        raise ValueError(f"pipeline mixes media types: {sorted(prefixes)}")
    parser = choose_parser(ext)
    return [parser if s in parser_steps else s for s in steps]


def _ensure_text_separation(steps: Sequence[str], ext: str) -> None:
    """Reject PDF page-artifact passes when the input is plain text."""
    if ext != ".txt":
        return
    unsupported = sorted(set(steps) & _TEXT_UNSUPPORTED_PASSES)
    if unsupported:
        raise ValueError(
            "PDF artifact passes are not supported for .txt input: " + ", ".join(unsupported)
        )


def _enforce_invariants(spec: PipelineSpec, *, input_path: str) -> list[str]:
    """Return validated steps while enforcing order and media-type invariants."""
    validate_options(spec.options)
    ext = Path(input_path).suffix.lower()
    steps = _ensure_pdf_epub_separation(list(spec.pipeline), ext)
    _ensure_text_separation(steps, ext)
    _ensure_clean_precedes_split(steps)
    return _pass_steps(PipelineSpec(pipeline=steps, options=spec.options))


def _prepare_pass(pass_obj: Pass, overrides: Mapping[str, Any]) -> tuple[Pass, Mapping[str, Any]]:
    """Return a configured pass and sanitized overrides for meta propagation."""

    opts = dict(overrides)
    if pass_obj.name == "ai_enrich" and opts.get("enabled"):
        from pdf_chunker.adapters.ai_enrich import prepare_options

        opts = prepare_options(opts)
    _validate_pass_options(pass_obj, opts)
    if not opts:
        return pass_obj, opts
    if not is_dataclass(pass_obj):
        return pass_obj, opts

    names = {f.name for f in fields(pass_obj)}
    updates = {k: opts[k] for k in opts if k in names}
    if (
        "chunk_size" in updates
        and "min_chunk_size" not in opts
        and hasattr(pass_obj, "min_chunk_size")
    ):
        updates["min_chunk_size"] = None

    if not updates:
        return pass_obj, opts

    new_pass = replace(pass_obj, **updates)
    post = getattr(new_pass, "__post_init__", None)
    if callable(post):
        post()
    return new_pass, opts


def _validate_pass_options(pass_obj: Pass, opts: Mapping[str, Any]) -> None:
    """Reject invalid numeric splitter settings before a pass starts."""
    if pass_obj.name != "split_semantic":
        return

    for name in ("chunk_size", "overlap", "min_chunk_size"):
        value = opts.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"split_semantic.{name} must be an integer")
        if isinstance(value, int) and value < (0 if name == "overlap" else 1):
            raise ValueError(f"split_semantic.{name} must be greater than zero")

    chunk_size = opts.get("chunk_size", getattr(pass_obj, "chunk_size", None))
    overlap = opts.get("overlap", getattr(pass_obj, "overlap", None))
    if isinstance(chunk_size, int) and isinstance(overlap, int) and overlap >= chunk_size:
        raise ValueError("split_semantic.overlap must be smaller than chunk_size")


def configure_pass(pass_obj: Pass, opts: Mapping[str, Any]) -> Pass:
    """Return a new pass with ``opts`` merged without mutating ``pass_obj``."""

    configured, _ = _prepare_pass(pass_obj, opts)
    return configured


_EXPECTED_INPUT_SHAPES: Final[Mapping[str, str]] = {
    "text_clean": "page_blocks",
    "list_detect": "page_blocks",
    "heading_detect": "page_blocks",
    "merge_footers": "page_blocks",
    "detect_doc_end": "page_blocks",
    "detect_page_artifacts": "page_blocks",
    "split_semantic": "page_blocks",
    "ai_enrich": "chunks",
    "emit_jsonl": "chunks",
}
_EXPECTED_OUTPUT_SHAPES: Final[Mapping[str, str]] = {
    "pdf_parse": "page_blocks",
    "epub_parse": "page_blocks",
    "text_parse": "page_blocks",
    "text_clean": "page_blocks",
    "list_detect": "page_blocks",
    "heading_detect": "page_blocks",
    "merge_footers": "page_blocks",
    "detect_doc_end": "page_blocks",
    "detect_page_artifacts": "page_blocks",
    "split_semantic": "chunks",
    "ai_enrich": "chunks",
    "emit_jsonl": "rows",
}


def _matches_shape(payload: object, shape: str) -> bool:
    """Return whether ``payload`` has the required canonical envelope."""
    if shape == "rows":
        return isinstance(payload, list)
    if not isinstance(payload, Mapping) or payload.get("type") != shape:
        return False
    field = "pages" if shape == "page_blocks" else "items"
    return isinstance(payload.get(field), list)


def _validate_payload_shape(payload: object, step: str, *, phase: str) -> None:
    """Validate a known pass boundary with a useful error message."""
    shapes = _EXPECTED_INPUT_SHAPES if phase == "input" else _EXPECTED_OUTPUT_SHAPES
    expected = shapes.get(step)
    if expected is not None and not _matches_shape(payload, expected):
        actual = payload.get("type") if isinstance(payload, Mapping) else type(payload).__name__
        raise PipelineContractError(
            f"{step} {phase} requires {expected!r} payload, received {actual!r}"
        )


def _time_step(
    acc: tuple[Artifact[object], dict[str, float]],
    p: Pass,
) -> tuple[Artifact[object], dict[str, float]]:
    """Apply ``p`` to ``acc`` while recording its execution time."""
    a, timings = acc
    t0 = time.time()
    a = p(a)
    return a, {**timings, p.name: time.time() - t0}


def _with_pass_options(
    meta: Mapping[str, Any] | None, name: str, overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Return ``meta`` with ``overrides`` recorded under ``options[name]``."""

    base = {**(meta or {})}
    existing = dict((meta or {}).get("options") or {})
    if overrides:
        existing[name] = dict(overrides)
    if existing:
        base["options"] = existing
    else:
        base.pop("options", None)
    return base


def _run_passes(
    spec: PipelineSpec, a: Artifact[object]
) -> tuple[Artifact[object], dict[str, float]]:
    """Run pipeline passes declared in ``spec`` capturing per-pass timings."""
    configs = [
        _prepare_pass(registry()[name], spec.options.get(name, {})) for name in spec.pipeline
    ]

    def _apply(
        acc: tuple[Artifact[object], dict[str, float]],
        config: tuple[Pass, Mapping[str, Any]],
    ) -> tuple[Artifact[object], dict[str, float]]:
        artifact, timings = acc
        p, overrides = config
        _validate_payload_shape(artifact.payload, p.name, phase="input")
        seeded = Artifact(
            payload=artifact.payload,
            meta=_with_pass_options(artifact.meta, p.name, overrides),
        )
        result, updated_timings = _time_step((seeded, timings), p)
        _validate_payload_shape(result.payload, p.name, phase="output")
        return result, updated_timings

    initial: tuple[Artifact[object], dict[str, float]] = (a, {})
    return reduce(_apply, configs, initial)


def _adapter_for(path: str):
    """Return IO adapter for ``path`` based on its extension."""
    ext = Path(path).suffix.lower()
    module = {
        ".epub": "pdf_chunker.adapters.io_epub",
        ".txt": "pdf_chunker.adapters.io_text",
    }.get(ext, "pdf_chunker.adapters.io_pdf")
    return import_module(module)


def _excluded_all(path: str, exclude: str | None) -> bool:
    """Return True when ``exclude`` removes every page of ``path``."""
    if not exclude:
        return False
    import fitz

    from pdf_chunker.page_utils import parse_page_ranges

    excluded = parse_page_ranges(exclude)
    with fitz.open(path) as doc:
        return set(range(1, len(doc) + 1)) <= excluded


def _load_document_metadata(path: str | None) -> dict[str, Any] | None:
    """Read document metadata from a JSON or YAML mapping file."""
    if not path:
        return None

    def _json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        if isinstance(value, (datetime, date)):
            return value.isoformat().replace("+00:00", "Z")
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    data = import_module("yaml").safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise TypeError("metadata file must contain a top-level mapping")
    return _json_safe(data)


def _input_artifact(path: str, spec: PipelineSpec | None = None) -> Artifact[object]:
    """Load ``path`` honoring PDF exclusions from ``spec``."""
    spec_opts = (spec or PipelineSpec()).options
    parser_name = choose_parser(Path(path).suffix.lower())
    parse_opts = spec_opts.get(parser_name, {})
    text_clean_opts = spec_opts.get("text_clean", {})

    exclude = parse_opts.get("exclude_pages")
    # EPUB-specific spine exclusion (takes precedence over exclude_pages for EPUB)
    exclude_spine = parse_opts.get("exclude_spine")
    # Propagate interactive mode to PDF extraction so aggressive footer
    # detection is skipped, allowing downstream interactive confirmation
    interactive = text_clean_opts.get("interactive_footers", False)
    # Zone exclusion margins
    footer_margin = parse_opts.get("footer_margin")
    header_margin = parse_opts.get("header_margin")
    document_metadata = _load_document_metadata(parse_opts.get("metadata_path"))

    abs_path = str(Path(path).resolve())
    suffix = Path(path).suffix.lower()
    is_epub = suffix == ".epub"
    is_pdf = suffix == ".pdf"
    adapter = _adapter_for(path)

    # For EPUB, use exclude_spine if provided, otherwise fall back to exclude_pages
    epub_exclude = exclude_spine or exclude if is_epub else None

    # Pass interactive flag and zone margins if the adapter supports them
    try:
        payload = (
            {"type": "page_blocks", "source_path": abs_path, "pages": []}
            if is_pdf and _excluded_all(abs_path, exclude)
            else adapter.read(
                path,
                exclude_pages=epub_exclude if is_epub else exclude,
                interactive=interactive,
                footer_margin=footer_margin,
                header_margin=header_margin,
            )
        )
    except TypeError:
        # Adapter doesn't support interactive/zone parameters (e.g., EPUB)
        payload = (
            {"type": "page_blocks", "source_path": abs_path, "pages": []}
            if is_pdf and _excluded_all(abs_path, exclude)
            else adapter.read(path, exclude_pages=epub_exclude if is_epub else exclude)
        )
    if document_metadata and isinstance(payload, Mapping):
        payload = {**payload, "document_metadata": document_metadata}
    return Artifact(payload=payload, meta={"metrics": {}, "input": abs_path})


def _rows_from_payload(payload: Any) -> list[Chunk]:
    """Return chunk rows when ``payload`` is either a list or ``{"items": [...]}``."""
    rows = (
        payload
        if isinstance(payload, list)
        else payload.get("items", []) if isinstance(payload, Mapping) else []
    )
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise PipelineContractError("conversion output must be a list of mapping rows")
    return cast(list[Chunk], rows)


def _match(text: str, needle: str) -> bool:
    """Case-insensitive substring test."""
    return needle.lower() in text.lower()


def _filter_pages(pages: Iterable[Mapping[str, Any]], needle: str) -> list[dict[str, Any]]:
    """Return pages with blocks whose ``text`` contains ``needle``."""
    return [
        {**p, "blocks": bs}
        for p in pages
        if (bs := [b for b in p.get("blocks", []) if _match(b.get("text", ""), needle)])
    ]


def _trace_view(payload: Any, needle: str) -> Any:
    """Slice ``payload`` to elements matching ``needle``."""
    if isinstance(payload, Mapping):
        if "pages" in payload:
            return {**payload, "pages": _filter_pages(payload.get("pages", []), needle)}
        if "items" in payload:
            return [c for c in payload.get("items", []) if _match(c.get("text", ""), needle)]
    if isinstance(payload, list):
        return [c for c in payload if _match(c.get("text", ""), needle)]
    return payload


def convert(path: str, spec: PipelineSpec) -> list[Chunk]:
    """Convert document at ``path`` using ``spec`` and return emitted rows."""
    validate_options(spec.options)
    artifact = _input_artifact(path, spec)
    input_path = str((artifact.meta or {}).get("input", ""))
    steps = _enforce_invariants(spec, input_path=input_path)
    run_spec = PipelineSpec(pipeline=steps, options=spec.options)
    artifact, _ = _run_passes(run_spec, artifact)
    return _rows_from_payload(artifact.payload)


def _maybe_emit_jsonl(
    a: Artifact,
    spec: PipelineSpec,
    timings: Mapping[str, float],
) -> None:
    """Write JSONL output when pipeline requests ``emit_jsonl``."""
    if "emit_jsonl" in spec.pipeline:
        emit_jsonl.maybe_write(
            a,
            spec.options.get("emit_jsonl", {}),
            dict(timings),
        )


def _has_footnote(texts: Iterable[str]) -> bool:
    """Return True if any ``texts`` contain footnote markers."""
    pattern = re.compile(r"\bfootnote\b|\[\d+\]", re.IGNORECASE)
    return any(pattern.search(t) for t in texts)


def _page_exclusion_noop(spec: PipelineSpec, meta: Mapping[str, Any]) -> bool:
    """Warn when page exclusion was requested but no pages were excluded."""
    excl = spec.options.get("pdf_parse", {}).get("exclude_pages")
    if not excl:
        return False
    metrics = meta.get("metrics", {}).get("pdf_parse", {}) or {}
    return not metrics.get("excluded_pages")


def _has_metadata_gaps(chunks: Iterable[Mapping[str, Any]]) -> bool:
    """Detect missing ``chunk_id`` or ``source`` metadata fields."""
    required = {"chunk_id", "source"}
    return any(required - set(c.get("metadata", {})) for c in chunks)


def _underscore_loss(spec: PipelineSpec) -> bool:
    """Flag potential underscore loss when PyMuPDF4LLM engine is used."""
    engine = spec.options.get("pdf_parse", {}).get("engine")
    return engine == "pymupdf4llm"


def _generate_metadata_enabled(spec: PipelineSpec) -> bool:
    """Return True when chunk metadata generation is enabled."""
    return spec.options.get("split_semantic", {}).get("generate_metadata", True)


def _warning_checks(
    a: Artifact[object], spec: PipelineSpec, generate_metadata: bool
) -> Iterable[tuple[str, bool]]:
    """Yield pairs of warning names and their boolean status."""
    payload = a.payload if isinstance(a.payload, Iterable) else []
    chunks = [c for c in payload if isinstance(c, Mapping)]
    core = (
        (
            "footnote_anchors",
            _has_footnote(c.get("text", "") for c in chunks),
        ),
        (
            "page_exclusion_noop",
            _page_exclusion_noop(spec, a.meta or {}),
        ),
        (
            "underscore_loss",
            _underscore_loss(spec),
        ),
    )
    meta = (
        (
            "metadata_gaps",
            _has_metadata_gaps(chunks),
        ),
    )
    return chain(core, meta) if generate_metadata else core


def _collect_warnings(
    a: Artifact[object],
    spec: PipelineSpec,
    *,
    generate_metadata: bool,
) -> list[str]:
    """Return list of known-issue warnings derived from ``a`` and ``spec``."""
    return [name for name, flag in _warning_checks(a, spec, generate_metadata) if flag]


def _legacy_counts(metrics: Mapping[str, Any]) -> dict[str, int]:
    """Return aggregate page and chunk counts from pass metrics."""
    pages = (metrics.get("pdf_parse") or {}).get("pages")
    chunks = (metrics.get("split_semantic") or {}).get("chunks") or (
        metrics.get("emit_jsonl") or {}
    ).get("rows")
    pairs = (("page_count", pages), ("chunk_count", chunks))
    return {k: v for k, v in pairs if v is not None}


# --- run report helpers ----------------------------------------------------


def _dependency_versions() -> dict[str, Any]:
    names = ("fitz", "pypdf", "regex", "rapidfuzz")

    def version(n: str) -> Any:
        try:
            return getattr(import_module(n), "__version__", None)
        except (ImportError, AttributeError):
            return None

    return {n: version(n) for n in names}


def _pass_hash(p: Pass) -> tuple[str, str]:
    try:
        path = inspect.getsourcefile(p.__class__)
        data = Path(path).read_bytes() if path else b""
        digest = md5(data).hexdigest() if data else ""
    except (OSError, TypeError, ValueError):
        digest = ""
    return p.name, digest


def _pass_hashes(passes: Iterable[Pass]) -> dict[str, str]:
    return {name: h for name, h in (_pass_hash(p) for p in passes) if h}


def collect_environment(passes: Iterable[Pass]) -> dict[str, Any]:
    """Collect report environment data in the imperative runner boundary."""
    return {
        "sys_version": sys.version,
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "passes": _pass_hashes(passes),
    }


def assemble_report(
    timings: Mapping[str, float],
    meta: Mapping[str, Any],
    passes: Iterable[Pass],
    *,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble run report data from supplied values without performing I/O."""
    metrics = dict(meta.get("metrics") or {})
    counts = _legacy_counts(metrics)
    del passes  # retained for compatibility with callers of the old signature
    env = dict(environment or meta.get("environment") or {})
    return {
        "timings": dict(timings),
        "metrics": {**counts, **metrics, "env": env},
        "warnings": list(meta.get("warnings") or []),
    }


# replaced below


def write_run_report(spec: PipelineSpec, report: Mapping[str, Any]) -> None:
    """Write ``report`` to ``run_report.json`` honoring options path."""
    path = spec.options.get("run_report", {}).get("output_path", "run_report.json")
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _timed(p: Pass, a: Artifact[object], timings: dict[str, float]) -> Artifact[object]:
    """Run ``p`` while recording its execution duration."""
    t0 = time.time()
    try:
        return p(a)
    finally:
        timings[p.name] = time.time() - t0


def _meta_with_warnings(
    a: Artifact[object], spec: PipelineSpec, opts: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return meta merged with options and warning indicators."""
    gen_meta = _generate_metadata_enabled(spec)
    warnings = _collect_warnings(a, spec, generate_metadata=gen_meta)
    return {**(a.meta or {}), "options": opts, "warnings": warnings}


def run_convert(
    a: Artifact[object], spec: PipelineSpec, *, trace: str | None = None
) -> tuple[Artifact[object], dict[str, float]]:
    """Run declared passes, emit outputs, and persist a run report."""
    validate_options(spec.options)
    steps = _enforce_invariants(spec, input_path=str((a.meta or {}).get("input", "")))
    run_spec = PipelineSpec(pipeline=steps, options=spec.options)
    existing = cast(Mapping[str, Any], (a.meta or {}).get("options") or {})
    opts = {**existing, **spec.options}
    run_meta = {**(a.meta or {}), "options": opts}
    if trace:
        run_meta["trace_transformations"] = True
    a = Artifact(payload=a.payload, meta=run_meta)
    pass_configs = [
        _prepare_pass(registry()[name], spec.options.get(name, {})) for name in run_spec.pipeline
    ]
    passes = [configured for configured, _ in pass_configs]
    environment = collect_environment(passes)
    timings: dict[str, float] = {}
    try:
        for p, pass_options in pass_configs:
            seeded = Artifact(
                payload=a.payload,
                meta=_with_pass_options(a.meta, p.name, pass_options),
            )
            _validate_payload_shape(seeded.payload, p.name, phase="input")
            if trace:
                emit_trace.record_call(p.name)
            a = _timed(p, seeded, timings)
            _validate_payload_shape(a.payload, p.name, phase="output")
            if trace:
                emit_trace.write_snapshot(p.name, _trace_view(a.payload, trace))
                emit_trace.write_dups(p.name, a.payload)
    except Exception as exc:
        meta = _meta_with_warnings(a, spec, opts)
        report = assemble_report(
            timings,
            {**meta, "error": str(exc)},
            passes,
            environment=environment,
        )
        write_run_report(spec, report)
        raise
    _maybe_emit_jsonl(a, spec, timings)
    meta = _meta_with_warnings(a, spec, opts)
    report = assemble_report(timings, meta, passes, environment=environment)
    write_run_report(spec, report)
    return Artifact(payload=a.payload, meta=dict(meta)), timings


def run_inspect() -> dict[str, dict[str, str]]:
    """Return a lightweight view of the registry for CLI/tests."""
    return {
        name: {"input": str(p.input_type), "output": str(p.output_type)}
        for name, p in registry().items()
    }
