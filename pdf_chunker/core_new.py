from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from pdf_chunker.adapters import emit_jsonl, io_pdf
from pdf_chunker.config import PipelineSpec
from pdf_chunker.framework import Artifact, registry, run_pipeline


def _adapter_for(path: str):
    """Return the IO adapter based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".epub":
        from pdf_chunker.adapters import io_epub

        return io_epub
    return io_pdf


def _initial_artifact(path: str) -> Artifact:
    """Load document via chosen adapter into an Artifact."""
    adapter = _adapter_for(path)
    payload = adapter.read(path)
    return Artifact(payload=payload, meta={"metrics": {}, "input": path})


def _pass_steps(spec: PipelineSpec) -> list[str]:
    """Filter pipeline steps to registered passes; error on unknown ones."""
    regs = registry()
    unknown = [s for s in spec.pipeline if s not in regs and s != "emit_jsonl"]
    if unknown:
        raise KeyError(f"unknown steps: {unknown}")
    return [s for s in spec.pipeline if s in regs]


def _run_passes(steps: Iterable[str], a: Artifact) -> tuple[Artifact, dict[str, float]]:
    """Run pipeline steps sequentially while recording timings."""
    timings: dict[str, float] = {}
    for s in steps:
        t0 = time.time()
        a = run_pipeline([s], a)
        timings[s] = time.time() - t0
    meta = dict(a.meta or {})
    meta.setdefault("metrics", {})["_timings"] = timings
    return Artifact(payload=a.payload, meta=meta), timings


def _emit(a: Artifact, spec: PipelineSpec, timings: dict[str, float]) -> None:
    """Emit the artifact using the JSONL adapter when configured."""
    emit_jsonl.maybe_write(a, spec.options.get("emit_jsonl", {}), timings)


def run_convert(input_path: str, spec: PipelineSpec) -> Artifact:
    """Load ``input_path``, run declared passes, and maybe write JSONL."""
    a = _initial_artifact(input_path)
    steps = _pass_steps(spec)
    a, timings = _run_passes(steps, a)
    _emit(a, spec, timings)
    return a


def run_inspect() -> dict[str, dict[str, str]]:
    """Return a lightweight view of the registry for CLI/tests."""
    return {
        name: {"input": str(p.input_type), "output": str(p.output_type)}
        for name, p in registry().items()
    }
