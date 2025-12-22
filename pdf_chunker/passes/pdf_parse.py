from __future__ import annotations
from collections.abc import Iterable, Mapping
from itertools import groupby
from typing import Any

from pdf_chunker.framework import Artifact, register

Blocks = Iterable[Mapping[str, Any]]


def _page(block: Mapping[str, Any]) -> int | None:
    page = block.get("source", {}).get("page")
    return page if isinstance(page, int) else None


def _group_blocks_by_page(blocks: Blocks) -> list[dict[str, Any]]:
    key = _page
    seq = sorted(blocks, key=lambda b: (key(b) is None, key(b) or 0))
    return [{"page": page, "blocks": list(group)} for page, group in groupby(seq, key=key)]


def _to_page_blocks(blocks: Blocks, source_path: str) -> dict:
    return {
        "type": "page_blocks",
        "source_path": source_path,
        "pages": _group_blocks_by_page(blocks),
    }


def _meta(meta: dict[str, Any] | None, pages: int) -> dict[str, Any]:
    base = dict(meta or {})
    metrics = base.setdefault("metrics", {}).setdefault("pdf_parse", {})
    metrics["pages"] = pages
    return base


class _PdfParsePass:
    """Normalize blocks into ``PageBlocks`` without performing IO."""

    name = "pdf_parse"
    input_type = object
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        payload = a.payload
        if isinstance(payload, Mapping) and payload.get("type") == "page_blocks":
            doc = payload
        else:
            blocks = (
                payload
                if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, Mapping))
                else []
            )
            source = (a.meta or {}).get("input") or "<memory>"
            doc = _to_page_blocks(blocks, str(source))
        meta = _meta(a.meta, len(doc.get("pages", [])))
        return Artifact(payload=doc, meta=meta)


pdf_parse = register(_PdfParsePass())
