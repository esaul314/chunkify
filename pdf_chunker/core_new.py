from __future__ import annotations

import inspect
import json
import platform
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from functools import reduce
from hashlib import md5
from importlib import import_module
from itertools import chain
from pathlib import Path
from typing import Any, Final

import pdf_chunker.adapters.emit_trace as emit_trace
from pdf_chunker.adapters import emit_jsonl
from pdf_chunker.config import PipelineSpec
from pdf_chunker.framework import Artifact, Pass, registry

_PARSER_MAP: Final[Mapping[str, str]] = {
    ext: name
    for name, exts in {
        "pdf_parse": (".pdf",),
        "epub_parse": (".epub",),
    }.items()
    for ext in exts
}


def choose_parser(suffix: str) -> str:
    """Return parser step for ``suffix`` using a lookup table."""
    return _PARSER_MAP[suffix.lower()]


def _pass_steps(spec: PipelineSpec) -> list[str]:
    """Filter pipeline steps to registered passes; error on unknown ones."""
    regs = registry()
    unknown = [s for s in spec.pipeline if s not in regs and s != "emit_jsonl"]
    if unknown:
        raise KeyError(f"unknown steps: {unknown}")
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
    prefixes = {s.split("_", 1)[0] for s in steps if s.startswith(("pdf_", "epub_"))}
    if len(prefixes) > 1:
        raise ValueError(f"pipeline mixes media types: {sorted(prefixes)}")
    parser = choose_parser(ext)
    return [parser if s in {"pdf_parse", "epub_parse"} else s for s in steps]


def _enforce_invariants(spec: PipelineSpec, *, input_path: str) -> list[str]:
    """Return validated steps while enforcing order and media-type invariants."""
    ext = Path(input_path).suffix.lower()
    steps = _ensure_pdf_epub_separation(list(spec.pipeline), ext)
    _ensure_clean_precedes_split(steps)
    return _pass_steps(PipelineSpec(pipeline=steps, options=spec.options))


def _prepare_pass(pass_obj: Pass, overrides: Mapping[str, Any]) -> tuple[Pass, Mapping[str, Any]]:
    """Return a configured pass and sanitized overrides for meta propagation."""

    opts = dict(overrides)
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


def configure_pass(pass_obj: Pass, opts: Mapping[str, Any]) -> Pass:
    """Return a new pass with ``opts`` merged without mutating ``pass_obj``."""

    configured, _ = _prepare_pass(pass_obj, opts)
    return configured


def _time_step(
    acc: tuple[Artifact, dict[str, float]],
    p: Pass,
) -> tuple[Artifact, dict[str, float]]:
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


def _run_passes(spec: PipelineSpec, a: Artifact) -> tuple[Artifact, dict[str, float]]:
    """Run pipeline passes declared in ``spec`` capturing per-pass timings."""
    configs = [
        _prepare_pass(registry()[name], spec.options.get(name, {})) for name in spec.pipeline
    ]

    def _apply(
        acc: tuple[Artifact, dict[str, float]],
        config: tuple[Pass, Mapping[str, Any]],
    ) -> tuple[Artifact, dict[str, float]]:
        artifact, timings = acc
        p, overrides = config
        seeded = Artifact(
            payload=artifact.payload,
            meta=_with_pass_options(artifact.meta, p.name, overrides),
        )
        return _time_step((seeded, timings), p)

    return reduce(_apply, configs, (a, {}))


def _adapter_for(path: str):
    """Return IO adapter for ``path`` based on its extension."""
    ext = Path(path).suffix.lower()
    module = {
        ".epub": "pdf_chunker.adapters.io_epub",
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


def _input_artifact(path: str, spec: PipelineSpec | None = None) -> Artifact:
    """Load ``path`` honoring PDF exclusions from ``spec``."""
    opts = (spec or PipelineSpec()).options.get("pdf_parse", {})
    exclude = opts.get("exclude_pages")
    abs_path = str(Path(path).resolve())
    payload = (
        {"type": "page_blocks", "source_path": abs_path, "pages": []}
        if _excluded_all(abs_path, exclude)
        else _adapter_for(path).read(path, exclude_pages=exclude)
    )
    return Artifact(payload=payload, meta={"metrics": {}, "input": abs_path})


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Return chunk rows when ``payload`` is either a list or ``{"items": [...]}``."""
    return (
        payload
        if isinstance(payload, list)
        else payload.get("items", [])
        if isinstance(payload, Mapping)
        else []
    )


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


def convert(path: str, spec: PipelineSpec) -> list[dict[str, Any]]:
    """Convert document at ``path`` using ``spec`` and return emitted rows."""
    artifact = _input_artifact(path, spec)
    input_path = (artifact.meta or {}).get("input", "")
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
    a: Artifact, spec: PipelineSpec, generate_metadata: bool
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
    a: Artifact,
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
        except Exception:
            return None

    return {n: version(n) for n in names}


def _pass_hash(p: Pass) -> tuple[str, str]:
    try:
        path = inspect.getsourcefile(p.__class__)
        data = Path(path).read_bytes() if path else b""
        digest = md5(data).hexdigest() if data else ""
    except Exception:
        digest = ""
    return p.name, digest


def _pass_hashes(passes: Iterable[Pass]) -> dict[str, str]:
    return {name: h for name, h in (_pass_hash(p) for p in passes) if h}


def _env_snapshot(passes: Iterable[Pass]) -> dict[str, Any]:
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
) -> dict[str, Any]:
    """Purely assemble run report data without performing IO."""
    metrics = dict(meta.get("metrics") or {})
    counts = _legacy_counts(metrics)
    env = _env_snapshot(passes)
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


def _timed(p: Pass, a: Artifact, timings: dict[str, float]) -> Artifact:
    """Run ``p`` while recording its execution duration."""
    t0 = time.time()
    try:
        return p(a)
    finally:
        timings[p.name] = time.time() - t0


def _meta_with_warnings(
    a: Artifact, spec: PipelineSpec, opts: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return meta merged with options and warning indicators."""
    gen_meta = _generate_metadata_enabled(spec)
    warnings = _collect_warnings(a, spec, generate_metadata=gen_meta)
    return {**(a.meta or {}), "options": opts, "warnings": warnings}


def run_convert(
    a: Artifact, spec: PipelineSpec, *, trace: str | None = None
) -> tuple[Artifact, dict[str, float]]:
    """Run declared passes, emit outputs, and persist a run report."""
    steps = _enforce_invariants(spec, input_path=str((a.meta or {}).get("input", "")))
    run_spec = PipelineSpec(pipeline=steps, options=spec.options)
    existing = (a.meta or {}).get("options") or {}
    opts = {**existing, **spec.options}
    a = Artifact(payload=a.payload, meta={**(a.meta or {}), "options": opts})
    passes = [configure_pass(registry()[s], spec.options.get(s, {})) for s in run_spec.pipeline]
    timings: dict[str, float] = {}
    try:
        for p in passes:
            if trace:
                emit_trace.record_call(p.name)
            a = _timed(p, a, timings)
            if trace:
                emit_trace.write_snapshot(p.name, _trace_view(a.payload, trace))
                emit_trace.write_dups(p.name, a.payload)
    except Exception as exc:
        meta = _meta_with_warnings(a, spec, opts)
        report = assemble_report(timings, {**meta, "error": str(exc)}, passes)
        write_run_report(spec, report)
        raise
    _maybe_emit_jsonl(a, spec, timings)
    meta = _meta_with_warnings(a, spec, opts)
    report = assemble_report(timings, meta, passes)
    write_run_report(spec, report)
    return Artifact(payload=a.payload, meta=dict(meta)), timings


def run_inspect() -> dict[str, dict[str, str]]:
    """Return a lightweight view of the registry for CLI/tests."""
    return {
        name: {"input": str(p.input_type), "output": str(p.output_type)}
        for name, p in registry().items()
    }
