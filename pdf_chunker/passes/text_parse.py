from pdf_chunker.framework import register
from pdf_chunker.passes.pdf_parse import _PdfParsePass


class _TextParsePass(_PdfParsePass):
    """Plain-text analogue of ``pdf_parse`` reusing the same normalization."""

    name = "text_parse"


text_parse = register(_TextParsePass())
