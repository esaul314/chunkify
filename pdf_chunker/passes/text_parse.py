from pdf_chunker.framework import register
from pdf_chunker.passes.pdf_parse import _PdfParsePass


class _TextParsePass(_PdfParsePass):
    """Normalize transcript or verified Markdown blocks without reading text."""

    name = "text_parse"


text_parse = register(_TextParsePass())
