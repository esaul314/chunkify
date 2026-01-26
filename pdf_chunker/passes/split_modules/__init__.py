"""Split modules subpackage for modular chunk splitting.

This subpackage decomposes the monolithic split_semantic.py (1,888 lines)
into focused modules:

- footers.py: Footer detection and stripping
- lists.py: List boundary detection and splitting
- overlap.py: Boundary overlap management
- stitching.py: Block stitching and merging
- segments.py: Segment emission and collapsing
- inline_headings.py: Inline heading detection and promotion

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
from pdf_chunker.passes.split_modules.inline_headings import (
    promote_inline_heading,
    split_inline_heading,
    split_inline_heading_records,
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
from pdf_chunker.passes.split_modules.segments import (
    _CollapseEmitter,
    _allow_colon_list_overflow,
    _allow_cross_page_list,
    _allow_list_overflow,
    _buffer_has_number_two,
    _buffer_last_list_number,
    _collapse_records,
    _collapse_step,
    _effective_counts,
    _emit_buffer_segments,
    _emit_individual_records,
    _emit_segment_records,
    _enumerate_segments,
    _hard_limit,
    _join_record_texts,
    _maybe_merge_dense_page,
    _merged_segment_record,
    _normalize_numbered_list_text,
    _overlap_value,
    _page_or_footer_boundary,
    _projected_overflow,
    _record_starts_numbered_item,
    _resolve_bullet_strategy,
    _resolved_limit,
    _segment_allows_list_overflow,
    _segment_is_colon_list,
    _segment_offsets,
    _segment_totals,
    _should_emit_list_boundary,
    _text_has_number_two,
    collapse_records,
)
from pdf_chunker.passes.split_modules.stitching import (
    is_heading,
    merge_record_block,
    stitch_block_continuations,
    with_chunk_index,
)

__all__ = [
    # Footer detection
    "resolve_footer_suffix",
    "record_is_footer_candidate",
    "record_trailing_footer_lines",
    "strip_footer_suffix",
    "strip_footer_suffixes",
    "is_footer_artifact_record",
    # Inline heading detection
    "split_inline_heading",
    "split_inline_heading_records",
    "promote_inline_heading",
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
    # Segment emission
    "collapse_records",
    "_collapse_records",
    "_CollapseEmitter",
    "_emit_buffer_segments",
    "_emit_segment_records",
    "_emit_individual_records",
    "_merged_segment_record",
    "_segment_totals",
    "_segment_offsets",
    "_enumerate_segments",
    "_segment_allows_list_overflow",
    "_segment_is_colon_list",
    "_resolved_limit",
    "_hard_limit",
    "_overlap_value",
    "_effective_counts",
    "_join_record_texts",
    "_normalize_numbered_list_text",
    "_page_or_footer_boundary",
    "_allow_cross_page_list",
    "_allow_list_overflow",
    "_allow_colon_list_overflow",
    "_should_emit_list_boundary",
    "_buffer_last_list_number",
    "_record_starts_numbered_item",
    "_text_has_number_two",
    "_buffer_has_number_two",
    "_projected_overflow",
    "_collapse_step",
    "_maybe_merge_dense_page",
    "_resolve_bullet_strategy",
    # Block stitching
    "stitch_block_continuations",
    "merge_record_block",
    "with_chunk_index",
    "is_heading",
]
