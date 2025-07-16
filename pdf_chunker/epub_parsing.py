# epub_parsing.py

import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from .text_cleaning import clean_paragraph
from .heading_detection import _detect_heading_fallback
from .extraction_fallbacks import _detect_language


def get_element_text_content(element) -> str:
    """Extract text from BeautifulSoup element without extra separators."""
    return ' '.join(
        ' '.join(child.stripped_strings) if hasattr(child, 'stripped_strings') else child
        for child in element.contents
    )


def process_epub_item(item, filename):
    soup = BeautifulSoup(item.get_content(), 'html.parser')
    body = soup.find('body')
    if not body:
        return []

    blocks = []
    for element in body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        raw_text = get_element_text_content(element)
        block_text = clean_paragraph(raw_text)
        if not block_text:
            continue

        block_type = (
            "heading"
            if element.name.startswith('h') or _detect_heading_fallback(block_text)
            else "paragraph"
        )

        blocks.append({
            "type": block_type,
            "text": block_text,
            "language": _detect_language(block_text),
            "source": {"filename": filename, "location": item.get_name()}
        })

    return blocks


def extract_text_blocks_from_epub(filepath: str) -> list[dict]:
    """
    Extracts structured text blocks from an EPUB file.

    Uses ebooklib.ITEM_DOCUMENT to enumerate document items.
    """
    book = epub.read_epub(filepath)
    filename = os.path.basename(filepath)

    # ITEM_DOCUMENT constant represents the content documents
    return [
        block
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        for block in process_epub_item(item, filename)
    ]
