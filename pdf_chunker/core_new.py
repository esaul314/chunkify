from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from functools import reduce
from importlib import import_module
from itertools import chain
from pathlib import Path
from typing import Any, Callable

from pdf_chunker.adapters import emit_jsonl
from pdf_chunker.config import PipelineSpec
from pdf_chunker.framework import Artifact, registry


def _pass_steps(spec: PipelineSpec) -> list[str]:
    """Filter pipeline steps to registered passes; error on unknown ones."""
    regs = registry()
    unknown = [s for s in spec.pipeline if s not in regs and s != "emit_jsonl"]
    if unknown:
        raise KeyError(f"unknown steps: {unknown}")
    return [s for s in spec.pipeline if s in regs]


def _ensure_clean_precedes_split(steps: Sequence[str]) -> None:
    """Raise descriptive error if a split pass precedes ``text_clean``."""
    first_split = next(((s, i) for i, s in enumerate(steps) if s.startswith("split")), None)
    if not first_split:
        return
    clean_index = next((i for i, s in enumerate(steps) if s == "text_clean"), None)
    split_name, split_index = first_split
    if clean_index is None or clean_index > split_index:
        raise ValueError(f"{split_name} requires text_clean to run beforehand")


def _ensure_pdf_epub_separation(steps: Sequence[str], ext: str) -> None:
    """Validate PDF/EPUB pass separation and match to ``ext``."""
    prefixes = {s.split("_", 1)[0] for s in steps if s.startswith(("pdf_", "epub_"))}
    if len(prefixes) > 1:
        raise ValueError(f"pipeline mixes media types: {sorted(prefixes)}")
    ext_map = {".pdf": "pdf", ".epub": "epub"}
    expected = ext_map.get(ext)
    if prefixes and expected and prefixes != {expected}:
        found = next(iter(prefixes))
        raise ValueError(f"{ext} input cannot use {found} passes")


def _enforce_invariants(spec: PipelineSpec, *, input_path: str) -> list[str]:
    """Return validated steps while enforcing order and media-type invariants."""
    steps = list(spec.pipeline)
    ext = Path(input_path).suffix.lower()
    validators = (
        _ensure_clean_precedes_split,
        lambda s: _ensure_pdf_epub_separation(s, ext),
    )
    tuple(v(steps) for v in validators)
    return _pass_steps(spec)


def _time_step(
    opts: Mapping[str, Mapping[str, Any]],
) -> Callable[[tuple[Artifact, dict[str, float]], str], tuple[Artifact, dict[str, float]]]:
    """Return reducer that runs ``step`` with optional ``opts`` and records timing."""

    def runner(
        acc: tuple[Artifact, dict[str, float]], step: str
    ) -> tuple[Artifact, dict[str, float]]:
        a, timings = acc
        p = registry()[step]
        step_opts = opts.get(step, {})
        if step_opts:
            try:
                p = p.__class__(**step_opts)
            except TypeError:
                p = p
        t0 = time.time()
        a = p(a)
        return a, {**timings, step: time.time() - t0}

    return runner


def _run_passes(
    steps: Iterable[str],
    a: Artifact,
    opts: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Artifact, dict[str, float]]:
    """Run ``steps`` sequentially while capturing per-step timings."""
    a, timings = reduce(_time_step(opts or {}), steps, (a, {}))
    # Timings previously nested under ``meta['metrics']['_timings']``.  The
    # report helper now owns them, so we return the artifact unchanged and let
    # ``assemble_report`` incorporate timings alongside metrics.  Behaviour is
    # equivalent: callers still receive step timings, but artifact metadata stays
    # focused on pass-emitted metrics.
    return a, timings


def _adapter_for(path: str):
    """Return IO adapter for ``path`` based on its extension."""
    ext = Path(path).suffix.lower()
    module = {".epub": "pdf_chunker.adapters.io_epub"}.get(ext, "pdf_chunker.adapters.io_pdf")
    return import_module(module)


def _excluded_all(path: str, exclude: str | None) -> bool:
    """Return True when ``exclude`` removes every page of ``path``."""
    if not exclude:
        return False
    from pdf_chunker.page_utils import parse_page_ranges
    import fitz

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
        else payload.get("items", []) if isinstance(payload, Mapping) else []
    )


def convert(path: str, spec: PipelineSpec) -> list[dict[str, Any]]:
    """Convert document at ``path`` using ``spec`` and return emitted rows."""
    artifact = _input_artifact(path, spec)
    steps = _enforce_invariants(spec, input_path=artifact.meta["input"])
    artifact, _ = _run_passes(steps, artifact, spec.options)
    return _rows_from_payload(artifact.payload)


def _maybe_emit_jsonl(a: Artifact, spec: PipelineSpec, timings: Mapping[str, float]) -> None:
    """Write JSONL output when pipeline requests ``emit_jsonl``."""
    if "emit_jsonl" in spec.pipeline:
        emit_jsonl.maybe_write(a, spec.options.get("emit_jsonl", {}), timings)


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


def _collect_warnings(a: Artifact, spec: PipelineSpec, *, generate_metadata: bool) -> list[str]:
    """Return list of known-issue warnings derived from ``a`` and ``spec``."""
    return [name for name, flag in _warning_checks(a, spec, generate_metadata) if flag]


def _legacy_counts(metrics: Mapping[str, Any]) -> dict[str, int]:
    """Return aggregate page and chunk counts from pass metrics."""
    pages = (metrics.get("pdf_parse") or {}).get("pages")
    chunks = (metrics.get("split_semantic") or {}).get("chunks") or (
        metrics.get("emit_jsonl") or {}
    ).get("rows")
    return {k: v for k, v in (("page_count", pages), ("chunk_count", chunks)) if v is not None}


def assemble_report(timings: Mapping[str, float], meta: Mapping[str, Any]) -> dict[str, Any]:
    """Purely assemble run report data without performing IO."""
    metrics = dict(meta.get("metrics") or {})
    counts = _legacy_counts(metrics)
    return {
        "timings": dict(timings),
        "metrics": {**counts, **metrics},
        "warnings": list(meta.get("warnings") or []),
    }


def write_run_report(spec: PipelineSpec, report: Mapping[str, Any]) -> None:
    """Write ``report`` to ``run_report.json`` honoring options path."""
    path = spec.options.get("run_report", {}).get("output_path", "run_report.json")
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_convert(a: Artifact, spec: PipelineSpec) -> tuple[Artifact, dict[str, float]]:
    """Run declared passes, emit outputs, and persist a run report."""
    steps = _enforce_invariants(spec, input_path=str((a.meta or {}).get("input", "")))
    a, timings = _run_passes(steps, a, spec.options)
    _maybe_emit_jsonl(a, spec, timings)
    gen_meta = _generate_metadata_enabled(spec)
    warnings = _collect_warnings(a, spec, generate_metadata=gen_meta)
    meta = {**(a.meta or {}), "warnings": warnings}
    report = assemble_report(timings, meta)
    write_run_report(spec, report)
    return Artifact(payload=a.payload, meta=meta), timings


def run_inspect() -> dict[str, dict[str, str]]:
    """Return a lightweight view of the registry for CLI/tests."""
    return {
        name: {"input": str(p.input_type), "output": str(p.output_type)}
        for name, p in registry().items()
    }
