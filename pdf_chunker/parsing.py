# parsing.py

import os
from .pdf_parsing import extract_text_blocks_from_pdf
from .epub_parsing import extract_text_blocks_from_epub


def extract_structured_text(filepath: str, exclude_pages: str = None) -> list[dict]:
    """
    Facade function to extract structured text from PDF or EPUB files.

    Args:
        filepath: Path to the input file
        exclude_pages: Pages to exclude (PDF only)

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
            return extract_text_blocks_from_epub(filepath)
        case _:
            raise ValueError(f"Unsupported file type: '{extension}'")
