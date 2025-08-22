from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from subprocess import run
from typing import Any, Callable, Mapping, Sequence
from unittest.mock import patch
import difflib

from scripts import chunk_pdf


LEGACY_FLAG_MAP = {"--chunk-size": "--chunk_size"}


def _legacy_flags(flags: Sequence[str]) -> Sequence[str]:
    return [LEGACY_FLAG_MAP.get(flag, flag) for flag in flags]


def _run_legacy(pdf: Path, out_path: Path, flags: Sequence[str] = ()) -> Path:
    argv = ["chunk_pdf.py", str(pdf), *_legacy_flags(flags)]
    with (
        out_path.open("w", encoding="utf-8") as f,
        redirect_stdout(f),
        patch.object(sys, "argv", argv),
    ):
        chunk_pdf.main()
    return out_path


def _run_new(pdf: Path, out_path: Path, flags: Sequence[str] = ()) -> Path:
    run(
        [
            "pdf_chunker",
            "convert",
            str(pdf),
            *flags,
            "--out",
            str(out_path),
        ],
        check=True,
    )
    return out_path


VOLATILE_FIELDS = frozenset({"timings"})


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: v.strip() if isinstance(v, str) else v
        for k, v in sorted(row.items())
        if k not in VOLATILE_FIELDS
    }


def _canonical_lines(path: Path) -> list[str]:
    return [
        json.dumps(_normalize(r), sort_keys=True)
        for r in _load_rows(path)
    ]


def _write_diff(name: str, legacy: Path, new: Path, diffdir: Path) -> None:
    diffdir.mkdir(parents=True, exist_ok=True)
    diff = difflib.unified_diff(
        _canonical_lines(legacy),
        _canonical_lines(new),
        fromfile="legacy",
        tofile="new",
        lineterm="",
    )
    (diffdir / f"{name}.diff").write_text("\n".join(diff), encoding="utf-8")


def run_parity(
    pdf: Path,
    tmpdir: Path,
    flags: Sequence[str] = (),
    diffdir: Path | None = None,
) -> tuple[Path, Path]:
    tmpdir.mkdir(parents=True, exist_ok=True)
    runners: tuple[Callable[[], Path], ...] = (
        partial(_run_legacy, pdf, tmpdir / "legacy.jsonl", flags),
        partial(_run_new, pdf, tmpdir / "new.jsonl", flags),
    )
    legacy, new = tuple(fn() for fn in runners)
    if diffdir:
        _write_diff(pdf.stem, legacy, new, diffdir)
    return legacy, new


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legacy and new pipelines")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("tmpdir", type=Path)
    parser.add_argument("--diffdir", type=Path)
    args = parser.parse_args()
    legacy, new = run_parity(args.pdf, args.tmpdir, diffdir=args.diffdir)
    for p in (legacy, new):
        print(p)


if __name__ == "__main__":
    main()
