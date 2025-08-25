import sys
from typing import Iterable


def _to_int(part: str) -> int:
    """Return a validated 1-based page number from ``part``."""
    try:
        number = int(part)
    except ValueError as exc:  # pragma: no cover - rethrown with context
        raise ValueError(f"Invalid page number: {part}") from exc
    if number < 1:
        raise ValueError(f"Invalid page number: {part}")
    return number


def _expand(part: str) -> Iterable[int]:
    """Yield page numbers for a single comma-delimited ``part``."""
    head, *tail = part.split("-", 1)
    start = _to_int(head)
    if not tail:
        return (start,)
    end = _to_int(tail[0])
    if start > end:
        raise ValueError(f"Invalid range: {part}")
    return range(start, end + 1)


def parse_page_ranges(page_spec: str) -> set[int]:
    """Convert a string spec into a set of 1-based page numbers."""
    if not page_spec or not page_spec.strip():
        return set()
    parts = (p.strip() for p in page_spec.split(",") if p.strip())
    return {n for part in parts for n in _expand(part)}


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
