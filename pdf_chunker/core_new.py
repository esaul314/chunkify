from __future__ import annotations

import time

from pdf_chunker.config import PipelineSpec
from pdf_chunker.framework import Artifact, registry, run_pipeline


def run_convert(spec: PipelineSpec, initial: Artifact | None = None) -> Artifact:
    """Sequentially run registered passes from the spec; record simple timings."""
    a = initial or Artifact(payload=None, meta={"metrics": {}})
    timings: dict[str, float] = {}
    for s in spec.pipeline:
        t0 = time.time()
        a = run_pipeline([s], a)
        timings[s] = time.time() - t0
    meta = dict(a.meta or {})
    meta.setdefault("metrics", {})["_timings"] = timings
    return Artifact(payload=a.payload, meta=meta)


def run_inspect() -> dict[str, dict[str, str]]:
    """Return a lightweight view of the registry for CLI/tests."""
    return {
        name: {"input": str(p.input_type), "output": str(p.output_type)}
        for name, p in registry().items()
    }
