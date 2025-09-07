# epub_parsing.py

import os
import sys
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, Tag
from typing import Dict, List, TypedDict, cast
from .text_cleaning import clean_paragraph
from .heading_detection import _detect_heading_fallback
from .fallbacks import default_language


class TextBlock(TypedDict):
    """Structured representation of extracted text blocks."""

    type: str
    text: str
    language: str
    source: Dict[str, str]


def get_element_text_content(element) -> str:
    """Extract text from BeautifulSoup element without extra separators."""

    def _to_text(child) -> str:
        if hasattr(child, "stripped_strings"):
            return " ".join(child.stripped_strings)
        return child

    return " ".join(_to_text(child) for child in element.contents)


def _prefixed_list_text(element, text: str) -> str:
    """Prefix ordered list items with their index."""
    parent = element.find_parent(["ol", "ul"])
    if parent and parent.name == "ol":
        siblings = [li for li in parent.find_all("li", recursive=False)]
        index = next((i for i, li in enumerate(siblings, 1) if li is element), 0)
        return f"{index}. {text}" if index else text
    return text


def _element_to_block(element, filename: str, item_name: str) -> TextBlock | None:
    """Convert a BeautifulSoup element into a TextBlock."""
    raw_text = get_element_text_content(element)
    block_text = clean_paragraph(raw_text)
    if not block_text:
        return None

    if element.name == "li":
        block_text = _prefixed_list_text(element, block_text)

    block_type = (
        "heading"
        if element.name.startswith("h") or _detect_heading_fallback(block_text)
        else "paragraph"
    )

    return {
        "type": block_type,
        "text": block_text,
        "language": default_language(),
        "source": {"filename": filename, "location": item_name},
    }


def process_epub_item(item: epub.EpubHtml, filename: str) -> List[TextBlock]:
    """Convert a spine item into structured text blocks."""
    soup = BeautifulSoup(item.get_content(), "html.parser")
    body = soup.find("body")
    if not isinstance(body, Tag):
        return []

    body_tag = cast(Tag, body)

    item_name = item.get_name()
    elements = body_tag.find_all(
        [
            "p",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ]
    )

    blocks = (_element_to_block(element, filename, item_name) for element in elements)
    return [
        block
        for block in blocks
        if block
        and not (
            item_name == "nav.xhtml"
            and block["type"] == "heading"
            and block["text"] == "Table of Contents"
        )
    ]


def extract_text_blocks_from_epub(
    filepath: str, exclude_spines: str | None = None
) -> List[TextBlock]:
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
                excluded_spines,
                len(spine_items),
                filename,
            )
        except ValueError as e:
            print(f"Error parsing spine exclusions: {e}", file=sys.stderr)
            print("Continuing without spine exclusions", file=sys.stderr)
            excluded_spines = set()

    # Process spine items with exclusion filtering
    all_blocks = []
    for spine_index, item in enumerate(spine_items, 1):
        if spine_index in excluded_spines:
            print(
                f"Skipping excluded spine item {spine_index}: {item.get_name()}",
                file=sys.stderr,
            )
            continue

        print(
            f"Processing spine item {spine_index}: {item.get_name()}",
            file=sys.stderr,
        )
        blocks = process_epub_item(item, filename)
        all_blocks.extend(blocks)

    return all_blocks


def list_epub_spines(filepath: str) -> list[dict]:
    """
    Lists spine items from an EPUB file with their indices,
    filenames, and content previews.

    Args:
        filepath: Path to the EPUB file

    Returns:
        List of dictionaries containing spine information:
        [{"index": 1, "filename": "cover.xhtml",
          "content_preview": "Cover Page..."}, ...]
    """
    book = epub.read_epub(filepath)
    spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    def extract_content_preview(item) -> str:
        """Extract first 50-100 characters of text content from a spine item."""
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            body = soup.find("body")
            if not isinstance(body, Tag):
                return "No readable content"

            body_tag = cast(Tag, body)
            text_parts = [
                raw_text.strip()
                for raw_text in (
                    get_element_text_content(element)
                    for element in body_tag.find_all(
                        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "span"]
                    )
                )
                if raw_text and raw_text.strip()
            ]

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
