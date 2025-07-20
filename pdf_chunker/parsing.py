# parsing.py
#
# Default extraction approach: Simplified PyMuPDF4LLM integration
# - Uses traditional font-based extraction for all structural analysis (headings, blocks, metadata)
# - Applies PyMuPDF4LLM text cleaning for superior text quality (ligatures, word joining, whitespace)
# - Maintains proven reliability while leveraging PyMuPDF4LLM's evolving capabilities
# - Reduces complexity compared to complex hybrid approaches

import os
from .pdf_parsing import extract_text_blocks_from_pdf
from .epub_parsing import extract_text_blocks_from_epub


def extract_structured_text(filepath: str, exclude_pages: str = None) -> list[dict]:
    """
    Facade function to extract structured text from PDF or EPUB files.

    Args:
        filepath: Path to the input file
        exclude_pages: Pages to exclude (PDF only) or spine indices to exclude (EPUB only)

    Returns:
        List of structured text blocks

    Raises:
        ValueError: if the file type is unsupported
    """
    extension = os.path.splitext(filepath)[1].lower()

    match extension:
        case ".pdf":
            return extract_text_blocks_from_pdf(filepath, exclude_pages)
        case ".epub":
            return extract_text_blocks_from_epub(filepath, exclude_spines=exclude_pages)
        case _:
            raise ValueError(f"Unsupported file type: '{extension}'")
