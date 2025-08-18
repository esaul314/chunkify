from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from functools import reduce
from pathlib import Path
from typing import Any

from pdf_chunker.adapters import emit_jsonl
from pdf_chunker.config import PipelineSpec
from pdf_chunker.framework import Artifact, registry, run_pipeline


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
    _ensure_clean_precedes_split(steps)
    _ensure_pdf_epub_separation(steps, Path(input_path).suffix.lower())
    return _pass_steps(spec)


def _time_step(acc: tuple[Artifact, dict[str, float]], step: str) -> tuple[Artifact, dict[str, float]]:
    """Run a single step and record its timing."""
    a, timings = acc
    t0 = time.time()
    a = run_pipeline([step], a)
    return a, {**timings, step: time.time() - t0}


def _run_passes(steps: Iterable[str], a: Artifact) -> tuple[Artifact, dict[str, float]]:
    """Run ``steps`` sequentially while capturing per-step timings."""
    a, timings = reduce(_time_step, steps, (a, {}))
    # Timings previously nested under ``meta['metrics']['_timings']``.  The
    # report helper now owns them, so we return the artifact unchanged and let
    # ``assemble_report`` incorporate timings alongside metrics.  Behaviour is
    # equivalent: callers still receive step timings, but artifact metadata stays
    # focused on pass-emitted metrics.
    return a, timings


def _maybe_emit_jsonl(
    a: Artifact, spec: PipelineSpec, timings: Mapping[str, float]
) -> None:
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


def _warning_checks(a: Artifact, spec: PipelineSpec) -> Iterable[tuple[str, bool]]:
    """Yield pairs of warning names and their boolean status."""
    payload = a.payload if isinstance(a.payload, Iterable) else []
    chunks = [c for c in payload if isinstance(c, Mapping)]
    return (
        (
            "footnote_anchors",
            _has_footnote(c.get("text", "") for c in chunks),
        ),
        (
            "page_exclusion_noop",
            _page_exclusion_noop(spec, a.meta or {}),
        ),
        (
            "metadata_gaps",
            _has_metadata_gaps(chunks),
        ),
        (
            "underscore_loss",
            _underscore_loss(spec),
        ),
    )


def _collect_warnings(a: Artifact, spec: PipelineSpec) -> list[str]:
    """Return list of known-issue warnings derived from ``a`` and ``spec``."""
    return [name for name, flag in _warning_checks(a, spec) if flag]


def assemble_report(timings: Mapping[str, float], meta: Mapping[str, Any]) -> dict[str, Any]:
    """Purely assemble run report data without performing IO."""
    metrics = dict(meta.get("metrics") or {})
    return {
        "timings": dict(timings),
        "metrics": metrics,
        "warnings": list(meta.get("warnings") or []),
    }


def write_run_report(spec: PipelineSpec, report: Mapping[str, Any]) -> None:
    """Write ``report`` to ``run_report.json`` honoring options path."""
    path = spec.options.get("run_report", {}).get("output_path", "run_report.json")
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_convert(a: Artifact, spec: PipelineSpec) -> tuple[Artifact, dict[str, float]]:
    """Run declared passes on ``a`` and return new artifact plus timings."""
    steps = _enforce_invariants(spec, input_path=str((a.meta or {}).get("input", "")))
    a, timings = _run_passes(steps, a)
    _maybe_emit_jsonl(a, spec, timings)
    warnings = _collect_warnings(a, spec)
    a = Artifact(payload=a.payload, meta={**(a.meta or {}), "warnings": warnings})
    return a, timings


def run_inspect() -> dict[str, dict[str, str]]:
    """Return a lightweight view of the registry for CLI/tests."""
    return {
        name: {"input": str(p.input_type), "output": str(p.output_type)}
        for name, p in registry().items()
    }
