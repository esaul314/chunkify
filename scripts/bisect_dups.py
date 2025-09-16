#!/usr/bin/env python3
"""Identify the first pipeline step producing duplicate rows.

Given a directory containing JSON snapshots for intermediate passes, this script
replays the remaining pipeline passes for each snapshot (from ``pdf_parse``
through ``split_semantic``) and runs the duplicate detector on the resulting
rows. It exits with code 1 once duplicates appear, reporting the pass responsible.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import reduce
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pdf_chunker.config import PipelineSpec, load_spec
from pdf_chunker.core_new import configure_pass
from pdf_chunker.diagnostics.dups import find_dups_chunks, find_dups_pageblocks
from pdf_chunker.framework import Artifact, registry


STEPS = (
    "pdf_parse",
    "text_clean",
    "heading_detect",
    "list_detect",
    "split_semantic",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> Artifact:
    payload = _read_json(path)
    return Artifact(payload=payload, meta={"input": "<snapshot>", "metrics": {}})


def _rows(payload: Any) -> Iterable[Mapping[str, Any]]:
    return (
        payload
        if isinstance(payload, list)
        else payload.get("items", [])
        if isinstance(payload, Mapping)
        else []
    )


def _passes_after(spec: PipelineSpec, start: str) -> list[str]:
    try:
        idx = spec.pipeline.index(start) + 1
    except ValueError as e:  # pragma: no cover - defensive
        raise KeyError(f"{start} not in pipeline") from e
    return spec.pipeline[idx:]


def _run_passes(spec: PipelineSpec, art: Artifact, names: Sequence[str]) -> Artifact:
    regs = registry()
    configured = (configure_pass(regs[n], spec.options.get(n, {})) for n in names if n in regs)
    return reduce(lambda acc, p: p(acc), configured, art)


def _dup_count(rows: Sequence[Mapping[str, Any]]) -> int:
    finder = find_dups_pageblocks if rows and "bbox" in rows[0] else find_dups_chunks
    return len(finder(rows))


def check_step(spec: PipelineSpec, folder: Path, step: str) -> int:
    snap = folder / f"{step}.json"
    if not snap.exists():
        return 0
    art = _artifact(snap)
    result = _run_passes(spec, art, _passes_after(spec, step))
    rows = list(_rows(result.payload))
    return _dup_count(rows)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bisect duplicate introduction")
    p.add_argument("--dir", required=True, help="Directory containing snapshots")
    p.add_argument("--spec", default="pipeline.yaml")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(format="[%(levelname)s] %(name)s:%(funcName)s – %(message)s")
    spec = load_spec(args.spec)
    folder = Path(args.dir)
    for step in STEPS:
        dups = check_step(spec, folder, step)
        if dups:
            print(f"duplicates after {step}: {dups}")
            return 1
    print("no duplicates detected")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
