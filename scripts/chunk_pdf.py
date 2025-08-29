from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from pdf_chunker.adapters import emit_jsonl
from pdf_chunker.cli import _cli_overrides, _resolve_spec_path
from pdf_chunker.config import load_spec
from pdf_chunker.core_new import convert as run_convert


def _to_row(row: dict[str, Any]) -> dict[str, Any]:
    base = {"text": row.get("text", "")}
    meta = {"metadata": row["meta"]} if "meta" in row else {}
    return base | meta


def _print_jsonl(rows: Iterable[dict[str, Any]]) -> None:
    """Emit ``rows`` as legacy-style JSON lines to stdout."""

    print("\n".join(json.dumps(_to_row(r), ensure_ascii=False) for r in rows))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chunk_pdf")
    parser.add_argument("document_file", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--exclude-pages")
    parser.add_argument("--no-metadata", action="store_true")
    args = parser.parse_args(argv)

    overrides = _cli_overrides(
        args.out,
        args.chunk_size,
        args.overlap,
        False,
        args.exclude_pages,
        args.no_metadata,
    )
    emit_opts = overrides.setdefault("emit_jsonl", {})
    emit_path = str(args.out) if args.out else None
    emit_opts["output_path"] = emit_path

    spec = load_spec(_resolve_spec_path("pipeline.yaml"), overrides=overrides)
    rows = run_convert(str(args.document_file), spec)
    if emit_path:
        emit_jsonl.write(rows, emit_path)
    _print_jsonl(rows)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
