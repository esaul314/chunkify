# parsing.py
#
# Default extraction approach: Simplified PyMuPDF4LLM integration
# - Uses traditional font-based extraction for all structural analysis
#   (headings, blocks, metadata)
# - Applies PyMuPDF4LLM text cleaning for superior text quality
#   (ligatures, word joining, whitespace)
# - Maintains proven reliability while leveraging PyMuPDF4LLM's evolving capabilities
# - Reduces complexity compared to complex hybrid approaches

from collections.abc import Callable
from pathlib import Path
from typing import Optional, cast

from .epub_parsing import TextBlock, extract_text_blocks_from_epub
from .pdf_parsing import extract_text_blocks_from_pdf

Extractor = Callable[[Path, Optional[str]], list[TextBlock]]


def _pdf_extractor(path: Path, exclude: Optional[str]) -> list[TextBlock]:
    return cast(
        list[TextBlock],
        extract_text_blocks_from_pdf(str(path), exclude_pages=exclude),
    )


def _epub_extractor(path: Path, exclude: Optional[str]) -> list[TextBlock]:
    return extract_text_blocks_from_epub(str(path), exclude_spines=exclude)


DISPATCH: dict[str, Extractor] = {
    ".pdf": _pdf_extractor,
    ".epub": _epub_extractor,
}


def extract_structured_text(
    path: Path | str, exclude_pages: Optional[str] = None
) -> list[TextBlock]:
    """Return text blocks from a PDF or EPUB file.

    The extractor is chosen solely by the file's suffix, keeping the function
    free of side effects and external state.
    """

    p = Path(path)
    try:
        extractor = DISPATCH[p.suffix.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported file type: '{p.suffix}'") from exc
    return extractor(p, exclude_pages)
