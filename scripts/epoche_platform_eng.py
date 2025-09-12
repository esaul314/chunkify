#!/usr/bin/env python3
"""Run platform-eng excerpt through the pipeline and check for duplicates.

This script runs ``pdf_chunker convert`` on ``platform-eng-excerpt.pdf`` with
tracing enabled for the phrase "Most engineers". After the run, it inspects the
trace directory and exits non-zero if any pass produced duplicate text or if a
pass executed more than once.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

TRACE_BASE = Path("artifacts/trace")


def _latest_run_dir(base: Path) -> Path:
    return max((d for d in base.iterdir() if d.is_dir()), key=lambda p: p.stat().st_mtime)


def _bad_calls(run_dir: Path) -> list[str]:
    data = json.loads((run_dir / "calls.json").read_text())
    return [step for step, count in data.get("counts", {}).items() if count != 1]


def _top_dups(run_dir: Path) -> list[tuple[str, Sequence[Mapping[str, object]]]]:
    def load(p: Path) -> tuple[str, list[Mapping[str, object]]]:
        d = json.loads(p.read_text())
        return p.name, d.get("dups", [])

    return [
        (name, dups[:5])
        for name, dups in (load(p) for p in run_dir.glob("*_dups.json"))
        if dups
    ]


def _run_pipeline(out: Path) -> subprocess.CompletedProcess[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "pdf_chunker.cli",
        "convert",
        "platform-eng-excerpt.pdf",
        "--spec",
        "pipeline.yaml",
        "--no-enrich",
        "--out",
        str(out),
        "--trace",
        "Most engineers",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    result = _run_pipeline(Path("/tmp/epoche.jsonl"))
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    run_dir = _latest_run_dir(TRACE_BASE)
    bad_calls = _bad_calls(run_dir)
    bad_dups = _top_dups(run_dir)
    if bad_calls or bad_dups:
        print("Issues detected:")
        if bad_calls:
            print("  Passes executed multiple times:", ", ".join(bad_calls))
        for name, dups in bad_dups:
            print(f"  Duplicates in {name}:")
            for d in dups:
                print(f"    {d['fp']}: {d['text']!r}")
        return 1
    print("No duplicates detected and passes executed once.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
