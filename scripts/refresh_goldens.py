"""Regenerate golden JSONL outputs and optionally approve updates."""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_chunker.core import process_document

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "expected"
SAMPLES_DIR = ROOT / "tests" / "golden" / "samples"


def _decode(src: Path, dst: Path) -> Path:
    dst.write_bytes(base64.b64decode(src.read_text()))
    return dst


def _identity(src: Path, dst: Path) -> Path:
    dst.write_bytes(src.read_bytes())
    return dst


_SPEC: dict[str, tuple[Path, str, Callable[[Path, Path], Path]]] = {
    "pdf": (SAMPLES_DIR / "sample.pdf.b64", "pdf", _decode),
    "tiny": (SAMPLES_DIR / "tiny.pdf", "pdf", _identity),
    "epub": (SAMPLES_DIR / "sample.epub.b64", "epub", _decode),
}


def _chunks(path: Path) -> Iterable[dict[str, object]]:
    return process_document(
        str(path), chunk_size=1000, overlap=0, generate_metadata=True, ai_enrichment=False
    )


def _jsonl(chunks: Iterable[dict[str, object]]) -> str:
    return "\n".join(json.dumps(c, sort_keys=True) for c in chunks)


def _diff(old: Path, new: Path) -> str:
    return "\n".join(
        difflib.unified_diff(
            old.read_text(encoding="utf-8").splitlines(),
            new.read_text(encoding="utf-8").splitlines(),
            fromfile=str(old),
            tofile=str(new),
        )
    )


def _refresh(kind: str, approve: bool, tmp: Path) -> str | None:
    src, ext, materialize = _SPEC[kind]
    try:
        inp = materialize(src, tmp / f"sample.{ext}")
    except Exception as exc:  # pragma: no cover
        return f"{kind}: materialization failed: {exc}"
    try:
        new_path = tmp / f"{kind}.jsonl"
        new_path.write_text(_jsonl(_chunks(inp)), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        return f"{kind}: generation failed: {exc}"
    target = GOLDEN_DIR / f"{kind}.jsonl"
    if not target.exists():
        return f"{kind}: no golden at {target}"
    diff = _diff(target, new_path)
    if diff:
        if approve:
            target.write_text(new_path.read_text(encoding="utf-8"), encoding="utf-8")
        return diff
    return f"{kind}: up-to-date"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approve", action="store_true", help="overwrite goldens")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        messages = (_refresh(k, args.approve, tmp) for k in _SPEC if k != "epub" or _have_epub())
        for msg in filter(None, messages):
            print(msg)
    return 0


def _have_epub() -> bool:
    try:  # pragma: no cover - optional dependency
        import ebooklib  # noqa: F401

        return True
    except Exception:
        print("epub: skipping (ebooklib missing)")
        return False


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
