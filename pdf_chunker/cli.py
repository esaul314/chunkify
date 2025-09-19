"""Command-line interface for the pdf_chunker package."""

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain
from typing import Any

from pdf_chunker.core import process_document

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Chunk a document into structured JSONL.")
    parser.add_argument("document_file", help="Path to the document file (PDF or EPUB)")
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--exclude-pages",
        type=str,
        help="Page ranges to exclude from processing (e.g., '1,3,5-10,15-20').",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Set this flag to exclude metadata from the output.",
    )
    parser.add_argument(
        "--list-spines",
        action="store_true",
        help="List EPUB spine items with their indices and filenames (EPUB only).",
    )
    return parser


def _list_spines(document_file: str) -> int:
    if not document_file.lower().endswith(".epub"):
        print("Error: --list-spines can only be used with EPUB files.", file=sys.stderr)
        return 1

    try:
        from pdf_chunker.epub_parsing import list_epub_spines

        spine_items = list_epub_spines(document_file)
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Error listing spine items: {exc}", file=sys.stderr)
        return 1

    header = f"EPUB Spine Structure ({len(spine_items)} items):"
    lines = (
        f"{item['index']:3d}. {item['filename']} - {item['content_preview']}"
        for item in spine_items
    )
    print("\n".join(chain((header,), lines)))
    return 0


def _serialize_chunks(chunks: Iterable[dict[str, Any]]) -> Iterator[str]:
    for chunk in chunks:
        if not chunk:
            continue
        try:
            yield json.dumps(chunk, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"Error serializing chunk: {exc}", file=sys.stderr)


def run_cli(args: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(args)

    logger.debug("Processing document: %s", namespace.document_file)
    logger.debug(
        "Arguments: chunk_size=%s, overlap=%s, no_metadata=%s, exclude_pages=%s",
        namespace.chunk_size,
        namespace.overlap,
        namespace.no_metadata,
        namespace.exclude_pages,
    )

    if namespace.list_spines:
        return _list_spines(namespace.document_file)

    chunks = process_document(
        namespace.document_file,
        namespace.chunk_size,
        namespace.overlap,
        generate_metadata=not namespace.no_metadata,
        exclude_pages=namespace.exclude_pages,
    )

    for payload in _serialize_chunks(chunks):
        print(payload)

    return 0


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
    )
    raise SystemExit(run_cli(args))


__all__ = ["run_cli", "main"]


if __name__ == "__main__":
    main()
