# epub_parsing.py

import os
import sys
from collections.abc import Iterable
from functools import reduce
from typing import NotRequired, TypedDict, cast

import ebooklib
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from ebooklib import epub

from .heading_detection import _detect_heading_fallback
from .language import default_language
from .text_cleaning import clean_paragraph


class TextBlock(TypedDict):
    """Structured representation of extracted text blocks."""

    type: str
    text: str
    language: str
    source: dict[str, str | int]
    list_kind: NotRequired[str]


def get_element_text_content(element, *, omit_nested_lists: bool = False) -> str:
    """Extract text from a BeautifulSoup element without extra separators.

    When extracting a list item, nested lists are emitted as their own items and
    must not be copied into the parent item as well.
    """

    def _to_text(child) -> str:
        if hasattr(child, "stripped_strings"):
            return " ".join(child.stripped_strings)
        return child

    contents = (
        child
        for child in element.contents
        if not (omit_nested_lists and isinstance(child, Tag) and child.name in {"ol", "ul"})
    )
    return " ".join(_to_text(child) for child in contents)


def _prefixed_list_text(element, text: str) -> str:
    """Prefix ordered list items with their index."""
    parent = element.find_parent(["ol", "ul"])
    if parent and parent.name == "ol":
        siblings = [li for li in parent.find_all("li", recursive=False)]
        index = next((i for i, li in enumerate(siblings, 1) if li is element), 0)
        return f"{index}. {text}" if index else text
    return text


def _element_to_block(element, filename: str, item_name: str, spine_index: int) -> TextBlock | None:
    """Convert a BeautifulSoup element into a TextBlock."""
    raw_text = get_element_text_content(element, omit_nested_lists=element.name == "li")
    block_text = clean_paragraph(raw_text)
    if not block_text:
        return None

    # Determine block type and list metadata
    if element.name == "li":
        block_text = _prefixed_list_text(element, block_text)
        block_type = "list_item"
        # Detect list kind from parent element
        parent = element.find_parent(["ol", "ul"])
        list_kind = "numbered" if parent and parent.name == "ol" else "bullet"
    elif element.name.startswith("h") or _detect_heading_fallback(block_text):
        block_type = "heading"
        list_kind = None
    else:
        block_type = "paragraph"
        list_kind = None

    result: TextBlock = {
        "type": block_type,
        "text": block_text,
        "language": default_language(),
        "source": {
            "filename": filename,
            "location": item_name,
            "page": spine_index,
        },
    }
    if list_kind:
        result["list_kind"] = list_kind
    return result


_SEMANTIC_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
_CONTAINER_TAGS = {
    "div",
    "section",
    "article",
    "main",
    "blockquote",
    "ol",
    "ul",
    "table",
    "tr",
    "td",
}
_NONREADING_TAGS = {"script", "style", "nav", "noscript"}


def _inline_body_text(node: Tag | NavigableString) -> str:
    """Preserve inline word fragments; line breaks separate adjacent words."""
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in _NONREADING_TAGS:
        return ""
    if node.name == "br":
        return "\n"
    return "".join(
        _inline_body_text(child)
        for child in node.children
        if isinstance(child, (Tag, NavigableString))
    )


def _text_element(parts: list[str]) -> Tag | None:
    """Make an independent paragraph; never reparent or mutate the input DOM."""
    text = "".join(parts)
    if not text.strip():
        return None
    element = BeautifulSoup("", "html.parser").new_tag("p")
    element.string = text
    return element


def _body_elements(container: Tag) -> Iterable[Tag]:
    """Yield semantic blocks and uncovered body text once, in document order.

    Container wrappers are traversed, not emitted wholesale. A semantic block
    owns its inline descendants; nested list items retain separate list metadata.
    """
    pending: list[str] = []
    for child in container.children:
        if isinstance(child, Tag) and child.name in _NONREADING_TAGS:
            if child.name == "nav":
                # Keep the existing semantic TOC/list contract, but never
                # recover new arbitrary navigation text as body paragraphs.
                paragraph = _text_element(pending)
                if paragraph is not None:
                    yield paragraph
                pending = []
                yield from child.find_all(list(_SEMANTIC_TAGS))
            continue
        if isinstance(child, Tag) and (
            child.name in _SEMANTIC_TAGS | _CONTAINER_TAGS
            or child.find(list(_SEMANTIC_TAGS | _CONTAINER_TAGS))
        ):
            paragraph = _text_element(pending)
            if paragraph is not None:
                yield paragraph
            pending = []
            if child.name in _SEMANTIC_TAGS:
                yield child
                if child.name == "li":
                    yield from child.find_all("li")
            else:
                yield from _body_elements(child)
        elif isinstance(child, (Tag, NavigableString)):
            pending.append(_inline_body_text(child))
    paragraph = _text_element(pending)
    if paragraph is not None:
        yield paragraph


def process_epub_item(item: epub.EpubHtml, filename: str, spine_index: int) -> list[TextBlock]:
    """Convert a spine item into structured text blocks."""
    soup = BeautifulSoup(item.get_content(), "html.parser")
    body = soup.find("body")
    if not isinstance(body, Tag):
        return []

    body_tag = cast(Tag, body)

    item_name = item.get_name()
    elements = _body_elements(body_tag)

    blocks = (_element_to_block(element, filename, item_name, spine_index) for element in elements)
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
) -> list[TextBlock]:
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
    all_blocks: list[TextBlock] = []
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


def _merged_source(previous: TextBlock, current: TextBlock) -> dict[str, str | int]:
    """Keep navigation text from becoming the provenance of chapter content."""

    source = dict(previous.get("source", {}))
    current_source = current.get("source", {})
    if source.get("location") == "nav.xhtml":
        location = current_source.get("location")
        if isinstance(location, str) and location != "nav.xhtml":
            source["location"] = location
    return source


def _is_heading(block: TextBlock) -> bool:
    return block.get("type") == "heading"


def _should_merge_blocks(
    previous: TextBlock, current: TextBlock, *, max_chars: int, min_anchor: int
) -> bool:
    """Return True when paragraph/list blocks should coalesce."""

    if _is_heading(previous) or _is_heading(current):
        return False

    curr_text = current["text"].lstrip()

    if curr_text.endswith(":"):
        return False

    merged = _join_block_text(previous["text"], current["text"])
    merged_length = len(merged)
    if merged_length <= max_chars:
        return True

    previous_length = len(previous["text"])
    return previous_length < min_anchor and merged_length <= (max_chars + min_anchor)


def _attach_heading_prefixes(
    blocks: Iterable[TextBlock],
) -> tuple[list[TextBlock], list[TextBlock]]:
    """Attach collected heading text to the following paragraph or list block."""

    def step(
        state: tuple[list[TextBlock], list[TextBlock]], block: TextBlock
    ) -> tuple[list[TextBlock], list[TextBlock]]:
        pending, acc = state
        if _is_heading(block):
            return [*pending, block], acc

        if pending:
            heading_text = _DOUBLE_NEWLINE.join(h["text"] for h in pending)
            merged_source = dict(block.get("source", {}))
            page_candidates = [merged_source.get("page")] + [
                h.get("source", {}).get("page") for h in pending
            ]
            pages = [p for p in page_candidates if isinstance(p, int)]
            if pages:
                merged_source["page"] = min(pages)
            merged: TextBlock = {
                **block,
                "text": _join_block_text(heading_text, block["text"]),
                "source": merged_source,
            }
            return [], [*acc, merged]

        return [], [*acc, block]

    initial: tuple[list[TextBlock], list[TextBlock]] = ([], [])
    return reduce(step, blocks, initial)


def _coalesce_body_blocks(
    blocks: Iterable[TextBlock],
    *,
    max_chars: int = EPUB_GROUP_CHAR_LIMIT,
    min_anchor: int = EPUB_GROUP_MIN_TARGET,
) -> list[TextBlock]:
    """Group adjacent paragraph/list blocks while keeping headings separate."""

    pending_headings, prepared = _attach_heading_prefixes(blocks)

    def step(acc: list[TextBlock], block: TextBlock) -> list[TextBlock]:
        if not acc:
            return [block]

        previous = acc[-1]
        if not _should_merge_blocks(previous, block, max_chars=max_chars, min_anchor=min_anchor):
            return [*acc, block]

        merged_block: TextBlock = {
            **previous,
            "text": _join_block_text(previous["text"], block["text"]),
            "source": _merged_source(previous, block),
        }
        return [*acc[:-1], merged_block]

    initial: list[TextBlock] = []
    merged_blocks = reduce(step, prepared, initial)
    return [*merged_blocks, *pending_headings]
