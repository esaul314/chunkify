# Test Stabilization Plan

## Completed Work
* Restored whitespace-preserving chunk slices by recording word spans, slicing windows, and applying footer pruning without mutating the splitter, keeping `_split_text_into_chunks` pure and reusable.【F:pdf_chunker/splitter.py†L501-L606】
* Isolated footer bullets and dense segments through `_record_trailing_footer_lines`, `_record_is_footer_candidate`, and the reworked `_collapse_records`, so merge decisions flow through functional helpers instead of inline mutation.【F:pdf_chunker/passes/split_semantic.py†L760-L1040】
* Tightened list heuristics with whitespace-aware bullet and number detectors, reducing spurious matches while keeping list metadata propagation intact.【F:pdf_chunker/passes/list_detect.py†L17-L183】
* Centralized sentence-fusion budgets in `_BudgetView` so soft-limit overflow and dense fragment handling rely on deterministic, testable dataclasses.【F:pdf_chunker/passes/sentence_fusion.py†L63-L199】
* Updated readability utilities to treat acronyms deterministically, feed syllable counts through CMU/Pyphen fallbacks, and return structured difficulty metadata.【F:pdf_chunker/utils.py†L100-L178】

## Current Status (2025-09-29)
* `nox -s lint` fails because `black --check` wants to reformat `pdf_chunker/passes/split_semantic.py` even though the logic is pure.【61c927†L1-L8】
* `nox -s typecheck` succeeds with no mypy issues.【428704†L1-L4】
* `nox -s tests` fails on eight assertions covering footer joins, numbered-list formatting, CLI overrides, sentence-boundary guarantees, parity, and stale goldens.【0c3fcc†L1-L420】

## Next Steps to Reach Green
1. **Restore lint compliance**  
   • **Why**: `black --check` blocks the lint session on `pdf_chunker/passes/split_semantic.py`, preventing CI sign-off.【61c927†L1-L8】  
   • **What**: Reformat the module (or refactor long expressions into smaller helpers) so the file satisfies Black without altering functional semantics.  
   • **How**: Extract nested comprehensions inside `_collapse_records` and `_emit_segment_records` into named pure helpers before running `black --check` again to confirm.【F:pdf_chunker/passes/split_semantic.py†L819-L1040】

2. **Rejoin footer sentences without reintroducing bullets**  
   • **Why**: `test_footer_newlines_joined` now finds the footer bullet marker instead of the expected joined prose, signalling that `_record_trailing_footer_lines` is flushing too aggressively.【4b2fc6†L9-L23】【0c3fcc†L1-L120】  
   • **What**: Refine the footer candidate predicate so we only split when every trailing line is an artifact, preserving legitimate paragraph joins.  
   • **How**: Compute footer suffixes via `_record_trailing_footer_lines`, then merge adjacent narrative segments through a pure filter before `emit()` executes, keeping the helper composable and test-covered.【F:pdf_chunker/passes/split_semantic.py†L760-L1040】

3. **Collapse blank lines in numbered list merges**  
   • **Why**: `test_numbered_list_merge_collapses_blank_lines` expects `"1. first\n2. second"`, but the current splitter preserves the empty line introduced by `_split_list_record`.【ee137d†L28-L33】【0c3fcc†L226-L320】  
   • **What**: Normalize double newlines when we stitch numbered list fragments so adjacent items render contiguously.  
   • **How**: Add a pure newline-collapsing helper inside the semantic chunker merge path (likely `_merge_record_block` or `_apply_overlap_within_segment`) and assert the behavior with the existing regression test.【F:pdf_chunker/passes/split_semantic.py†L819-L1040】

4. **Reassert CLI overrides and sentence boundaries**  
   • **Why**: The CLI override harness only returns a single chunk and sentences still start mid-stream, failing `test_cli_flags_affect_split_semantic` and `test_no_chunk_starts_mid_sentence`.【b2a355†L26-L61】【746003†L58-L65】【0c3fcc†L226-L420】  
   • **What**: Ensure `_get_split_fn` respects override parameters when wrapping `semantic_chunker` and that `_merge_sentence_fragments` never yields a chunk beginning inside an unfinished sentence.  
   • **How**: Thread the override chunk size into `_soft_segments` and adjust `_merge_sentence_fragments`' overflow logic so punctuation gating forces a boundary before we emit the next window, validating with the targeted tests.【F:pdf_chunker/passes/split_semantic.py†L819-L1040】【F:pdf_chunker/passes/sentence_fusion.py†L142-L199】

5. **Recover platform-eng parity before refreshing goldens**  
   • **Why**: `test_platform_eng_parity` compares the refactored pipeline with the legacy splitter and currently reports text/meta drift, so downstream golden updates would lock in incorrect behavior.【3a8168†L1-L113】【0c3fcc†L311-L420】  
   • **What**: Diff the manual pipeline against `_SplitSemanticPass` to locate the first divergent chunk, then reconcile list merging and continuation stitching until texts, metadata, and metrics match.  
   • **How**: Use `_record_is_footer_candidate` instrumentation to trace buffer flushes, adjust `_split_colon_bullet_segments` or `_maybe_merge_dense_page` as needed, and rerun the parity test once the sequences align.【F:pdf_chunker/passes/split_semantic.py†L819-L1040】

6. **Re-align regression goldens once semantics hold**  
   • **Why**: The PDF and EPUB conversion tests differ because readability scores and chunk ordering changed; refreshing now would mask functional regressions.【2e9bed†L19-L42】【c32204†L13-L52】【3b0ae4†L12-L28】【0c3fcc†L1-L200】  
   • **What**: After stabilizing footer/list/override behavior, rerun the approved refresh script (`python scripts/refresh_goldens.py --approve`) to update `pdf.jsonl`, `epub.jsonl`, and `tiny.jsonl`.  
   • **How**: Capture before/after chunk diffs, document the command in the test log, and ensure the readability metadata produced by `_compute_readability` matches expectations before committing.【F:pdf_chunker/utils.py†L139-L178】

7. **Perform an end-to-end verification sweep**  
   • **Why**: Once the targeted fixes land, we must prove `nox -s lint`, `nox -s typecheck`, and `nox -s tests` all pass to close the stabilization effort.【61c927†L1-L8】【428704†L1-L4】【0c3fcc†L1-L420】  
   • **What**: Run the full nox matrix and spot-check CLI output on `platform-eng-excerpt.pdf` to ensure parity persists outside the unit suite.  
   • **How**: Execute the mandated nox sessions plus a CLI dry run, capturing logs for the next agent and leaving the pipeline ready for final approval per guardrails.【F:pdf_chunker/passes/split_semantic.py†L819-L1040】
