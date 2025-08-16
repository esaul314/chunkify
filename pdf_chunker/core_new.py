from __future__ import annotations

import time
from collections.abc import Iterable, Sequence, Callable
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


def _indices(pred: Callable[[str], bool], steps: Sequence[str]) -> list[int]:
    """Return indices of ``steps`` matching ``pred`` using comprehension."""
    return [i for i, s in enumerate(steps) if pred(s)]


def _ensure_clean_precedes_split(steps: Sequence[str]) -> None:
    """Raise if any split step occurs before ``text_clean``."""
    split_idx = _indices(lambda s: s.startswith("split"), steps)
    if not split_idx:
        return
    clean_idx = _indices(lambda s: s == "text_clean", steps)
    if not clean_idx or clean_idx[0] > min(split_idx):
        raise ValueError("text_clean must precede split passes")


def _ensure_pdf_epub_separation(steps: Sequence[str], ext: str) -> None:
    """Ensure pipeline contains only PDF or only EPUB passes and matches ``ext``."""
    prefixes = {s.split("_", 1)[0] for s in steps if s.startswith(("pdf_", "epub_"))}
    if len(prefixes) > 1:
        raise ValueError("pipeline mixes PDF and EPUB passes")
    ext_map = {".pdf": "pdf", ".epub": "epub"}
    expected = ext_map.get(ext)
    if prefixes and expected and prefixes != {expected}:
        raise ValueError(f"{ext} input incompatible with {next(iter(prefixes))} passes")


def _enforce_invariants(spec: PipelineSpec, *, input_path: str) -> None:
    """Validate pipeline order and media-type separation."""
    steps = list(spec.pipeline)
    _ensure_clean_precedes_split(steps)
    _ensure_pdf_epub_separation(steps, Path(input_path).suffix.lower())


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
    _enforce_invariants(spec, input_path=input_path)
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
