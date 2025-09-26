# FIX-SPLIT task status

## Summary
- ✅ All FIX-SPLIT acceptance tests now pass. Override parameters flow through the semantic splitter, dense fragments honour the tighter budgets, and chunks no longer begin mid-sentence.
- ✅ Sentence-fusion and continuation stitching share the override-aware limits, so whitespace-free fragments and continuation heads respect the configured ceilings without relying on imperative loops.
- ✅ Regression coverage guards dense fragment budgets and the no-mid-sentence contract, keeping the behaviour pinned to the override expectations.

## Evidence
- `pytest tests/passes/test_split_semantic_options.py::test_split_counts_change_with_overrides[overrides0-gt]`
- `pytest tests/passes/test_split_semantic_options.py::test_split_counts_change_with_overrides[overrides1-lt]`
- `pytest tests/semantic_chunking_test.py::test_no_chunk_starts_mid_sentence`

## Remediations
- `_SplitSemanticPass.__call__` threads the resolved `SplitOptions` into `_chunk_items`, which now collapses records using override-aware word and dense-fragment budgets. Continuation stitching receives the same limit so the final chunks remain aligned with the configured ceilings.
- `_merge_sentence_fragments` relies on functional merge budgeting to stop recombining dense fragments beyond the override allowance, and `_stitch_continuation_heads` redistributes sentence tails without imperative loops, ensuring chunk starts always fall on sentence boundaries.

## Status
- Step 1 — ✅ Propagate override parameters throughout semantic splitting and collapse helpers.
- Step 2 — ✅ Enforce sentence-boundary starts during continuation stitching.
- Step 3 — ✅ Maintain dense-fragment regression coverage.
- Task Stub: FIX-SPLIT — ✅ Completed.
