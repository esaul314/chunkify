from __future__ import annotations

from collections.abc import Mapping
from pdf_chunker.framework import Artifact, register


class _PdfParsePass:
    """Extract PDF blocks while remaining side-effect free."""

    name = "pdf_parse"
    input_type = str
    output_type = list

    def __call__(self, a: Artifact) -> Artifact:
        from pdf_chunker import pdf_parsing as _pdf_parsing

        path = str(a.payload)
        exclude = (a.meta or {}).get("exclude_pages")
        blocks = _pdf_parsing._legacy_extract_text_blocks_from_pdf(path, exclude)
        pages = {b.get("source", {}).get("page") for b in blocks if isinstance(b, Mapping)}
        metrics = {**((a.meta or {}).get("metrics", {})), "pdf_parse": {"pages": len(pages)}}
        meta = {**(a.meta or {}), "metrics": metrics}
        return Artifact(payload=blocks, meta=meta)


pdf_parse = register(_PdfParsePass())
