from __future__ import annotations

import argparse
import json
import logging
from functools import reduce
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pdf_chunker.adapters import emit_jsonl
from pdf_chunker.diagnostics.dups import (
    find_dups_chunks,
    find_dups_pageblocks,
)
from pdf_chunker.config import PipelineSpec, load_spec
from pdf_chunker.core_new import configure_pass
from pdf_chunker.framework import Artifact, registry


def _read_json(path: str) -> Any:
    """Load JSON from ``path`` using UTF-8 encoding."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_from_snapshot(path: str) -> Artifact:
    """Return Artifact seeded from a snapshot file."""
    payload = _read_json(path)
    return Artifact(payload=payload, meta={"input": "<from-snapshot>", "metrics": {}})


def passes_after(spec: PipelineSpec, start: str) -> list[str]:
    """Pass names in ``spec`` occurring after ``start``; error if missing."""
    try:
        idx = spec.pipeline.index(start) + 1
    except ValueError as e:  # pragma: no cover - defensive
        raise KeyError(f"{start} not in pipeline") from e
    return spec.pipeline[idx:]


def _configured(spec: PipelineSpec, names: Sequence[str]):
    regs = registry()
    return [
        configure_pass(regs[n], spec.options.get(n, {}))
        for n in names
        if n in regs
    ]


def run_passes(spec: PipelineSpec, a: Artifact, names: Sequence[str]) -> Artifact:
    """Apply ``names`` passes from ``spec`` to ``a`` sequentially."""
    return reduce(lambda acc, p: p(acc), _configured(spec, names), a)


def _rows(payload: Any) -> Iterable[Mapping[str, Any]]:
    """Extract chunk rows from ``payload`` when present."""
    return (
        payload
        if isinstance(payload, list)
        else payload.get("items", []) if isinstance(payload, Mapping) else []
    )


def replay(
    snapshot: str,
    start: str,
    spec: PipelineSpec,
    out: str | None,
    check_dups: bool,
) -> Artifact:
    """Replay downstream passes from ``snapshot`` starting after ``start``."""
    artifact = artifact_from_snapshot(snapshot)
    steps = passes_after(spec, start)
    logging.info("passes=%s", ",".join(steps))
    result = run_passes(spec, artifact, steps)
    rows = list(_rows(result.payload))
    emit_jsonl.write(rows, out)
    if check_dups:
        finder = (
            find_dups_pageblocks if rows and "bbox" in rows[0] else find_dups_chunks
        )
        dups = finder(rows)
        dups_path = Path(snapshot).parent / f"{steps[-1]}_dups.json"
        dups_path.write_text(
            json.dumps({"total": len(rows), "dups": dups}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("duplicate_rows=%d", len(dups))
    return result


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay remaining pipeline passes")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--from", dest="start", required=True)
    p.add_argument("--spec", default="pipeline.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--check-dups", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(format="[%(levelname)s] %(name)s:%(funcName)s – %(message)s")
    spec = load_spec(args.spec)
    replay(args.snapshot, args.start, spec, args.out, args.check_dups)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
