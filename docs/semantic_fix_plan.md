# Semantic Chunking Recovery Plan

## Objective
Restore the semantic chunking pipeline so that all currently failing unit, property-based, and end-to-end regression tests pass without introducing new regressions.

## Guiding Principles
- Keep modifications within the existing functional, pure-style helpers inside `pdf_chunker.passes.split_semantic` and `pdf_chunker.passes.text_clean`.
- Prefer refactoring toward single-responsibility helpers and generator-based flows instead of imperative loops.
- Avoid expanding IO boundaries; focus strictly on transformation logic and metadata coherence.

## Execution Steps
1. **Boundary Overlap Refinement**
   - Rework `_trim_boundary_overlap` to operate on partial window matches so footer lines no longer bleed into following chunks.
   - Split colon-prefixed buffer entries before merge emission in `_collapse_records` to keep list bullets intact.
   - Acceptance tests: `tests/footer_artifact_test.py::{test_footer_and_subfooter_removed,test_bullet_footer_removed}` and `tests/hyphen_bullet_list_test.py::test_hyphen_bullet_lists_preserved`.

2. **Heading Merge Spacing**
   - Adjust `_merge_heading_texts` to insert exactly one newline for single-line headings while preserving multi-line spacing.
   - Acceptance tests: `tests/heading_merge_rule_test.py::{test_heading_followed_by_paragraph,test_heading_followed_by_list_item}`.

3. **CLI Override Enforcement**
   - Ensure overflow sentinels trigger immediate emission in `_collapse_records` without collapsing subsequent records.
   - Acceptance tests: `tests/passes/test_split_semantic_options.py::{test_cli_flags_affect_split_semantic,test_dense_fragments_respect_override_limits}`.

4. **List Metadata Coherence**
   - Downgrade mixed list/paragraph merges to `paragraph` block types while retaining `list_kind` only when every contributor is a true list item via `_coalesce_block_type` and `_meta_is_list`.
   - Acceptance test: `tests/passes/test_split_semantic_parity.py::test_merge_record_block_preserves_list_kind_in_mixed_merge`.

5. **List Detection Guard Rails**
   - Verify `list_detection_edge_case_test.py` negatives remain strict after the above adjustments; expand heuristics only if regressions persist.
   - Acceptance tests: `tests/list_detection_edge_case_test.py::{test_is_bullet_list_pair_negative,test_is_numbered_list_pair_negative}`.

6. **Text Cleaning Idempotence**
   - Normalize NBSP and similar whitespace recursively inside `clean_paragraph` and ensure `_split_text_into_chunks` falls back to raw fragments when the cleaned output becomes empty.
   - Acceptance tests: `tests/property_based_text_test.py::{test_split_text_preserves_non_whitespace,test_split_roundtrip_cleaning}`.

7. **Golden & Readability Regression Sweep**
   - After semantic and cleaning fixes, re-run golden conversion, numbered-item preservation, parity sweeps, and readability grade test to confirm no downstream regressions.
   - Acceptance tests: `tests/golden/test_conversion.py`, `tests/golden/test_conversion_epub_cli.py`, `tests/golden/test_golden_pdf.py`, `tests/numbered_item_preservation_test.py::test_numbered_item_preserved`, `tests/parity/test_e2e_parity.py`, `tests/semantic_chunking_test.py::test_no_chunk_starts_mid_sentence`, and `tests/test_readability.py::test_readability_matches_expected_grade`.

## Next Steps
Execute steps sequentially, validating each acceptance group before proceeding to the next to isolate potential regressions early.
