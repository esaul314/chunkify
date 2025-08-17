from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import groupby
from typing import Any

from pdf_chunker.framework import Artifact, register


def to_page_blocks(
    blocks: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    source_path: str = "<memory>",
) -> dict[str, Any]:
    """Wrap a legacy ``list[block]`` payload into a ``page_blocks`` document."""

    def _page(block: Mapping[str, Any]) -> int:
        return int(block.get("page", 1))

    grouped = groupby(sorted(blocks, key=_page), _page)
    pages = ({"page": page, "blocks": [dict(b) for b in group]} for page, group in grouped)
    return {
        "type": "page_blocks",
        "source_path": source_path,
        "pages": list(pages),
    }


def _is_page_blocks(doc: Any) -> bool:
    return isinstance(doc, Mapping) and doc.get("type") == "page_blocks"


def _update_meta(meta: dict[str, Any] | None, pages: int) -> dict[str, Any]:
    base = dict(meta or {})
    metrics = base.setdefault("metrics", {})
    metrics.setdefault("pdf_parse", {})["pages"] = pages
    return base


class _PdfParsePass:
    """Coerce block payloads into the canonical ``page_blocks`` shape."""

    name = "pdf_parse"
    input_type = dict
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        payload = a.payload
        source = a.meta.get("input", "<memory>") if isinstance(a.meta, Mapping) else "<memory>"
        doc = (
            payload
            if _is_page_blocks(payload)
            else (
                to_page_blocks(payload, source_path=source)
                if isinstance(payload, list)
                else to_page_blocks([], source_path=source)
            )
        )
        meta = _update_meta(a.meta, len(doc.get("pages", [])))
        return Artifact(payload=doc, meta=meta)


pdf_parse = register(_PdfParsePass())
