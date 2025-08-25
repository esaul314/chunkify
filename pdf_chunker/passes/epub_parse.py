from pdf_chunker.framework import register
from pdf_chunker.passes.pdf_parse import _PdfParsePass


class _EpubParsePass(_PdfParsePass):
    """EPUB analogue of ``pdf_parse`` reusing the same normalization."""

    name = "epub_parse"


epub_parse = register(_EpubParsePass())
