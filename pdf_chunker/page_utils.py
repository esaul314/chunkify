import re
import sys
from typing import Iterable, Tuple


def _parse_single_page(part: str) -> int:
    """Convert a single page specifier into an integer."""
    try:
        page_num = int(part)
    except ValueError:
        raise ValueError(
            f"Invalid page number: '{part}'. Page numbers must be integers"
        )

    if page_num < 1:
        raise ValueError(f"Invalid page number: '{part}'. Page numbers must be >= 1")

    print(f"DEBUG: Individual page '{part}' parsed as: {page_num}", file=sys.stderr)
    return page_num


def _parse_page_range(part: str) -> Tuple[int, int]:
    """Return start and end integers for a range specifier like '5-10'."""
    range_parts = part.split("-")
    if len(range_parts) != 2:
        raise ValueError(f"Invalid page range format: '{part}'. Use format like '5-10'")

    try:
        start = int(range_parts[0].strip())
        end = int(range_parts[1].strip())
    except ValueError:
        raise ValueError(
            f"Invalid page numbers in range: '{part}'. Page numbers must be integers"
        )

    print(f"DEBUG: Range '{part}' parsed as start={start}, end={end}", file=sys.stderr)

    if start > end:
        raise ValueError(
            f"Invalid page range: '{part}'. Start page must be <= end page"
        )

    if start < 1 or end < 1:
        raise ValueError(f"Invalid page range: '{part}'. Page numbers must be >= 1")

    return start, end


def _expand_part(part: str) -> Iterable[int]:
    """Yield page numbers represented by a single specification part."""
    if "-" in part:
        start, end = _parse_page_range(part)
        range_pages = range(start, end + 1)
        print(
            f"DEBUG: Range '{part}' expands to pages: {list(range_pages)}",
            file=sys.stderr,
        )
        return range_pages

    page = _parse_single_page(part)
    return (page,)


def parse_page_ranges(page_spec: str) -> set:
    """Parse a comma-delimited page specification into a set of numbers.

    Supports formats like:
      - "1,3,5" (individual pages)
      - "1-5" (page ranges)
      - "1,3,5-10,15-20" (mixed individual and ranges)

    Args:
        page_spec: String specification of pages to exclude

    Returns:
        Set of page numbers to exclude

    Raises:
        ValueError: If the page specification is invalid
    """
    if not page_spec or not page_spec.strip():
        return set()

    parts = [p.strip() for p in page_spec.split(",") if p.strip()]

    print(f"DEBUG: Parsing page specification: '{page_spec}'", file=sys.stderr)
    print(f"DEBUG: Split into parts: {parts}", file=sys.stderr)

    excluded_pages = {page for part in parts for page in _expand_part(part)}

    final_pages = sorted(excluded_pages)
    print(f"DEBUG: Final excluded pages set: {final_pages}", file=sys.stderr)

    return excluded_pages


def validate_page_exclusions(
    excluded_pages: set, total_pages: int, filename: str
) -> set:
    """
    Validate page exclusions against the actual document and log warnings.

    Args:
        excluded_pages: Set of page numbers to exclude
        total_pages: Total number of pages in the document
        filename: Name of the file being processed (for logging)

    Returns:
        Set of valid page numbers to exclude (filtered to document bounds)
    """
    if not excluded_pages:
        return set()

    # Filter out pages beyond document bounds
    valid_exclusions = {page for page in excluded_pages if 1 <= page <= total_pages}

    # Log warnings for out-of-bounds pages
    invalid_pages = excluded_pages - valid_exclusions
    if invalid_pages:
        invalid_list = sorted(invalid_pages)
        print(
            f"Warning: Excluding pages beyond document bounds in '{filename}': {invalid_list} (document has {total_pages} pages)",
            file=sys.stderr,
        )

    # Log what pages are being excluded
    if valid_exclusions:
        excluded_list = sorted(valid_exclusions)
        print(
            f"Excluding {len(excluded_list)} pages from '{filename}': {excluded_list}",
            file=sys.stderr,
        )

    # Check if all pages would be excluded
    if len(valid_exclusions) >= total_pages:
        print(
            f"Warning: Page exclusions would exclude all pages in '{filename}'. Processing will continue with no exclusions.",
            file=sys.stderr,
        )
        return set()

    return valid_exclusions
