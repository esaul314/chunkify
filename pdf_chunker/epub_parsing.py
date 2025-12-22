# epub_parsing.py

import os
import sys
from functools import reduce
from typing import Dict, Iterable, List, Tuple, TypedDict, cast

import ebooklib
from bs4 import BeautifulSoup, Tag
from ebooklib import epub
from .text_cleaning import clean_paragraph
from .heading_detection import _detect_heading_fallback
from .language import default_language


class TextBlock(TypedDict):
    """Structured representation of extracted text blocks."""

    type: str
    text: str
    language: str
    source: Dict[str, str | int]


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


def _element_to_block(
    element, filename: str, item_name: str, spine_index: int
) -> TextBlock | None:
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
        "source": {
            "filename": filename,
            "location": item_name,
            "page": spine_index,
        },
    }


def process_epub_item(
    item: epub.EpubHtml, filename: str, spine_index: int
) -> List[TextBlock]:
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

    blocks = (
        _element_to_block(element, filename, item_name, spine_index)
        for element in elements
    )
    filtered = [
        block
        for block in blocks
        if block
        and not (
            item_name == "nav.xhtml"
            and block["type"] == "heading"
            and block["text"] == "Table of Contents"
        )
    ]
    return _coalesce_body_blocks(filtered)


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
    all_blocks: List[TextBlock] = []
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
        blocks = process_epub_item(item, filename, spine_index)
        all_blocks.extend(blocks)

    return _coalesce_body_blocks(
        all_blocks,
        max_chars=EPUB_DOCUMENT_CHAR_LIMIT,
        min_anchor=EPUB_DOCUMENT_MIN_TARGET,
    )


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
EPUB_GROUP_CHAR_LIMIT = 650
EPUB_GROUP_MIN_TARGET = 450
EPUB_DOCUMENT_CHAR_LIMIT = 1100
EPUB_DOCUMENT_MIN_TARGET = 0
_DOUBLE_NEWLINE = "\n\n"


def _join_block_text(first: str, second: str) -> str:
    """Join block texts while preserving paragraph spacing."""

    if not first:
        return second
    if not second:
        return first
    return f"{first.rstrip()}{_DOUBLE_NEWLINE}{second.lstrip()}"


def _is_heading(block: TextBlock) -> bool:
    return block.get("type") == "heading"


def _should_merge_blocks(
    previous: TextBlock, current: TextBlock, *, max_chars: int, min_anchor: int
) -> bool:
    """Return True when paragraph/list blocks should coalesce."""

    if _is_heading(previous) or _is_heading(current):
        return False

    prev_text = previous["text"].rstrip()
    curr_text = current["text"].lstrip()

    if curr_text.endswith(":"):
        return False

    merged = _join_block_text(previous["text"], current["text"])
    merged_length = len(merged)
    if merged_length <= max_chars:
        return True

    previous_length = len(previous["text"])
    return previous_length < min_anchor and merged_length <= (
        max_chars + min_anchor
    )


def _attach_heading_prefixes(blocks: Iterable[TextBlock]) -> Tuple[List[TextBlock], List[TextBlock]]:
    """Attach collected heading text to the following paragraph or list block."""

    def step(
        state: Tuple[List[TextBlock], List[TextBlock]], block: TextBlock
    ) -> Tuple[List[TextBlock], List[TextBlock]]:
        pending, acc = state
        if _is_heading(block):
            return [*pending, block], acc

        if pending:
            heading_text = _DOUBLE_NEWLINE.join(h["text"] for h in pending)
            merged_source = dict(block.get("source", {}))
            page_candidates = [
                merged_source.get("page")
            ] + [h.get("source", {}).get("page") for h in pending]
            pages = [p for p in page_candidates if isinstance(p, int)]
            if pages:
                merged_source["page"] = min(pages)
            merged = {
                **block,
                "text": _join_block_text(heading_text, block["text"]),
                "source": merged_source,
            }
            return [], [*acc, merged]

        return [], [*acc, block]

    return reduce(step, blocks, ([], []))


def _coalesce_body_blocks(
    blocks: Iterable[TextBlock],
    *,
    max_chars: int = EPUB_GROUP_CHAR_LIMIT,
    min_anchor: int = EPUB_GROUP_MIN_TARGET,
) -> List[TextBlock]:
    """Group adjacent paragraph/list blocks while keeping headings separate."""

    pending_headings, prepared = _attach_heading_prefixes(blocks)

    def step(acc: List[TextBlock], block: TextBlock) -> List[TextBlock]:
        if not acc:
            return [block]

        previous = acc[-1]
        if not _should_merge_blocks(
            previous, block, max_chars=max_chars, min_anchor=min_anchor
        ):
            return [*acc, block]

        merged_block = {**previous, "text": _join_block_text(previous["text"], block["text"])}
        return [*acc[:-1], merged_block]

    merged_blocks = reduce(step, prepared, [])
    return [*merged_blocks, *pending_headings]

