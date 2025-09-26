# FIX-SPLIT task status

## Summary
- Targeted acceptance tests currently fail. Adjusting semantic split options via overrides does not change the resulting chunk count, and chunk boundaries can still start mid-sentence.
- Root causes are concentrated in the semantic split pass: override-aware options are computed, but the downstream merge logic ignores their tighter budgets for dense fragments, and sentence-fusion heuristics do not guard against tails that lack terminal punctuation.

## Evidence
- `pytest tests/passes/test_split_semantic_options.py::test_split_counts_change_with_overrides[overrides0-gt]` fails because tightening `chunk_size` to 200 does not increase the number of produced chunks. 【262369†L1-L34】
- `pytest tests/semantic_chunking_test.py::test_no_chunk_starts_mid_sentence` fails when the second chunk is emitted without the previous chunk ending at a sentence boundary. 【efdf40†L1-L19】

## Contributing factors
- `_SplitSemanticPass.__call__` builds override-aware `SplitOptions`, but the fallback `_merge_sentence_fragments` logic re-joins dense fragments even when the override budget demands an additional split, so the downstream helpers never materialise more chunks. 【F:pdf_chunker/passes/split_semantic.py†L496-L515】【F:pdf_chunker/passes/sentence_fusion.py†L124-L168】
- `_merge_blocks` merges adjacent blocks whenever the next block looks like a continuation, but it does not re-check whether the preceding text already satisfied the sentence boundary requirement before yielding the chunk, leaving downstream chunks to start mid-sentence. 【F:pdf_chunker/passes/split_semantic.py†L320-L345】

## Next steps
1. Refactor `_SplitSemanticPass` so the resolved `SplitOptions` flow into `_chunk_items` and sentence-fusion helpers without reintroducing the default budgets.
2. Extend the sentence-fusion heuristics with a functional guard that forces a chunk break whenever the candidate tail lacks terminal punctuation and the continuation would exceed the override-aware merge budget.
3. Backfill regression coverage for dense fragments (no whitespace) to ensure overrides influence both the semantic splitter and the merge budget.
