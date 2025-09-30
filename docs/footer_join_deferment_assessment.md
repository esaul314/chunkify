# Footer Join Regression — Deferment Assessment

## Summary
`tests/footer_artifact_test.py::test_bullet_footer_removed` asserts that the bullet heavy sample should collapse into a single semantic chunk with the footer stripped. The test guards the guarantee that footer detection removes repeated Lincoln Woods bullets before downstream passes consume the document.【F:tests/footer_artifact_test.py†L57-L63】 Running the scenario today yields three emitted chunks, two of which still contain footer bullets, confirming the regression remains active.【db6e7b†L1-L27】

## Impact on Retrieval-Augmented Generation (RAG)
* **Recall pollution** — The stray footer chunk is short, highly structured, and repeats across pages, making it a high-probability match for lexical retrieval. Serving this artifact instead of the surrounding narrative dilutes recall precision and wastes context window budget during answer synthesis.【db6e7b†L1-L27】
* **Context drift** — Because `_strip_footer_suffixes` leaves the bullets behind whenever `_drop_trailing_bullet_footers` refuses to prune them, subsequent segments arrive detached from their narrative envelope.【F:pdf_chunker/passes/split_semantic.py†L769-L909】 Retrieval pipelines that rely on neighboring context lose the continuity needed to answer multi-sentence questions.【db6e7b†L1-L27】

## Impact on LoRA Fine-Tuning Corpora
* **Noise amplification** — LoRA adapters inherit whatever recurring patterns dominate the training batches. Feeding repeated footer bullets teaches the model to emit disclaimers and navigation fragments instead of semantically rich completions.【db6e7b†L1-L27】
* **Budget inefficiency** — Each surplus chunk increases token counts without adding supervision signal. For long-form corpora this inflates training cost and shifts optimizer updates toward boilerplate text, counteracting the plan’s goal of dense, information-bearing examples.【F:tests/footer_artifact_test.py†L57-L63】

## Secondary Consequences of Deferring
* Other guards, such as `_record_is_footer_candidate`, remain tuned for a world where suffix pruning succeeds. Leaving the regression in place means later passes (e.g., list balancing and readability estimators) continue to receive malformed buffers, complicating upcoming stabilization work.【F:pdf_chunker/passes/split_semantic.py†L769-L919】
* Documentation and diagnostics already codify this regression as an active blocker, so deferring it contradicts the stabilization roadmap captured in Task Stub 1.【F:docs/task_stub_1_footer_suffix_diagnostics.md†L1-L20】

## Recommendation
Do **not** defer this fix. The regression measurably harms RAG recall, corrupts LoRA training corpora with redundant boilerplate, and undermines adjacent stabilization tasks. Addressing the footer join immediately keeps the pipeline aligned with its core objective: delivering high-signal, low-noise chunks ready for downstream retrieval and adaptation.【F:tests/footer_artifact_test.py†L57-L63】【db6e7b†L1-L27】
