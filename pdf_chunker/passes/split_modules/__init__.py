"""Split modules subpackage for modular chunk splitting.

This subpackage decomposes the monolithic split_semantic.py (1,888 lines)
into focused modules:

- footers.py: Footer detection and stripping
- lists.py: List boundary detection and splitting
- overlap.py: Boundary overlap management
- emission.py: Segment emission (planned)
- stitching.py: Block stitching and merging (planned)

The main split_semantic.py remains the entry point but can delegate to these
modules for specific functionality.

Phase 3 of the strategic refactoring plan.
"""

from pdf_chunker.passes.split_modules.footers import (
    is_footer_artifact_record,
    record_is_footer_candidate,
    record_trailing_footer_lines,
    resolve_footer_suffix,
    strip_footer_suffix,
    strip_footer_suffixes,
)
from pdf_chunker.passes.split_modules.lists import (
    colon_bullet_boundary,
    first_list_number,
    last_list_number,
    list_tail_split_index,
    record_is_list_like,
    should_emit_list_boundary,
    starts_list_like,
)
from pdf_chunker.passes.split_modules.overlap import (
    is_overlap_token,
    missing_overlap_prefix,
    overlap_text,
    overlap_window,
    prepend_words,
    restore_chunk_overlap,
    restore_overlap_words,
    should_trim_overlap,
    split_words,
    trim_boundary_overlap,
    trim_sentence_prefix,
    trim_tokens,
)

__all__ = [
    # Footer detection
    "resolve_footer_suffix",
    "record_is_footer_candidate",
    "record_trailing_footer_lines",
    "strip_footer_suffix",
    "strip_footer_suffixes",
    "is_footer_artifact_record",
    # List boundary detection
    "first_list_number",
    "last_list_number",
    "starts_list_like",
    "record_is_list_like",
    "list_tail_split_index",
    "should_emit_list_boundary",
    "colon_bullet_boundary",
    # Overlap management
    "split_words",
    "overlap_window",
    "overlap_text",
    "is_overlap_token",
    "missing_overlap_prefix",
    "prepend_words",
    "restore_chunk_overlap",
    "restore_overlap_words",
    "trim_sentence_prefix",
    "should_trim_overlap",
    "trim_tokens",
    "trim_boundary_overlap",
]
