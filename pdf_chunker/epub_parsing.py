# epub_parsing.py

import os
import sys
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from .text_cleaning import clean_paragraph
from .heading_detection import _detect_heading_fallback
from .extraction_fallbacks import _detect_language


def get_element_text_content(element) -> str:
    """Extract text from BeautifulSoup element without extra separators."""
    return " ".join(
        (
            " ".join(child.stripped_strings)
            if hasattr(child, "stripped_strings")
            else child
        )
        for child in element.contents
    )


def process_epub_item(item, filename):
    soup = BeautifulSoup(item.get_content(), "html.parser")
    body = soup.find("body")
    if not body:
        return []

    blocks = []
    for element in body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
        raw_text = get_element_text_content(element)
        block_text = clean_paragraph(raw_text)
        if not block_text:
            continue

        block_type = (
            "heading"
            if element.name.startswith("h") or _detect_heading_fallback(block_text)
            else "paragraph"
        )

        blocks.append(
            {
                "type": block_type,
                "text": block_text,
                "language": _detect_language(block_text),
                "source": {"filename": filename, "location": item.get_name()},
            }
        )

    return blocks


def extract_text_blocks_from_epub(
    filepath: str, exclude_spines: str = None
) -> list[dict]:
    """
    Extracts structured text blocks from an EPUB file.

    Uses ebooklib.ITEM_DOCUMENT to enumerate document items.

    Args:
        filepath: Path to the EPUB file
        exclude_spines: Spine ranges to exclude (e.g., "1,3,5-10,15-20")
    """
    book = epub.read_epub(filepath)
    filename = os.path.basename(filepath)

    # Get spine items (ordered content documents)
    spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    print(f"EPUB has {len(spine_items)} spine items", file=sys.stderr)

    # Parse and validate spine exclusions
    excluded_spines = set()
    if exclude_spines:
        try:
            from .page_utils import parse_page_ranges, validate_page_exclusions

            excluded_spines = parse_page_ranges(exclude_spines)
            excluded_spines = validate_page_exclusions(
                excluded_spines, len(spine_items), filename
            )
        except ValueError as e:
            print(f"Error parsing spine exclusions: {e}", file=sys.stderr)
            print("Continuing without spine exclusions", file=sys.stderr)
            excluded_spines = set()

    # Process spine items with exclusion filtering
    all_blocks = []
    for spine_index, item in enumerate(
        spine_items, 1
    ):  # 1-based indexing like PDF pages
        if spine_index in excluded_spines:
            print(
                f"Skipping excluded spine item {spine_index}: {item.get_name()}",
                file=sys.stderr,
            )
            continue

        print(
            f"Processing spine item {spine_index}: {item.get_name()}", file=sys.stderr
        )
        blocks = process_epub_item(item, filename)
        all_blocks.extend(blocks)

    return all_blocks


def list_epub_spines(filepath: str) -> list[dict]:
    """
    Lists spine items from an EPUB file with their indices, filenames, and content previews.

    Args:
        filepath: Path to the EPUB file

    Returns:
        List of dictionaries containing spine information:
        [{"index": 1, "filename": "cover.xhtml", "content_preview": "Cover Page..."}, ...]
    """
    book = epub.read_epub(filepath)
    spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    def extract_content_preview(item) -> str:
        """Extract first 50-100 characters of text content from a spine item."""
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            body = soup.find("body")
            if not body:
                return "No readable content"

            # Extract text from all elements, similar to process_epub_item
            text_parts = []
            for element in body.find_all(
                ["p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "span"]
            ):
                raw_text = get_element_text_content(element)
                if raw_text and raw_text.strip():
                    text_parts.append(raw_text.strip())

            if not text_parts:
                return "No readable content"

            # Join text parts and clean up
            full_text = " ".join(text_parts)
            cleaned_text = clean_paragraph(full_text)

            # Truncate to 80 characters with ellipsis
            if len(cleaned_text) > 80:
                return cleaned_text[:77] + "..."
            return cleaned_text

        except Exception:
            return "Content extraction failed"

    return [
        {
            "index": index,
            "filename": item.get_name(),
            "content_preview": extract_content_preview(item),
        }
        for index, item in enumerate(spine_items, 1)
    ]
