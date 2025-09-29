# Test Stabilization Plan

The regression suite highlights a cluster of semantic-splitting and snapshot
issues. This plan prioritizes fixes so functional regressions land before we
refresh any goldens.

## 1. Restore chunk splitting invariants
- **Why**: `_split_text_into_chunks` now strips leading/trailing whitespace and
  rewraps bullet cleanup directly inside the splitter, which erases spacing and
  breaks the property-based invariants that compare cleaned inputs with the
  splitter round-trip.【F:pdf_chunker/splitter.py†L528-L563】【F:tests/property_based_text_test.py†L43-L59】
- **What**: Re-introduce whitespace-preserving token joins and keep the splitter
  focused on windowing logic. Extract bullet-footer filtering into a helper that
  operates on the detokenized text while retaining exact inter-token spacing.
- **How**: Build a pure composable helper that trims only artifact lines,
  delegate it from the splitter via functional composition, and cover the fix
  with the failing Hypothesis tests plus `splitter_transform_test.py`.

## 2. Stabilize footer chunk boundaries
- **Why**: `_collapse_records` merges footer bullets into nearby prose because
  `_starts_list_like` flags any bullet/number marker, causing footer segments to
  share buffers even when previous text ends cleanly. This collapses footer
  chunks that should remain isolated, violating `footer_artifact_test` counts and
  footer scrubbing assertions.【F:pdf_chunker/passes/split_semantic.py†L931-L1018】【F:pdf_chunker/passes/split_semantic.py†L1101-L1118】【F:tests/footer_artifact_test.py†L14-L63】
- **What**: Filter footer-style bullet runs before merge emission so genuine
  footers flush the buffer while real list bodies continue to fuse.
- **How**: Add a pure footer-detection predicate (re-using
  `page_artifacts._drop_trailing_bullet_footers` heuristics) inside
  `_collapse_records` and guard it with the footer regression tests.

## 3. Tighten list detection negatives
- **Why**: `is_bullet_list_pair` and `is_numbered_list_pair` accept random text
  whenever a colon precedes a hyphen or numbers appear later in the string,
  leading to false positives in the property suite.【F:pdf_chunker/passes/list_detect.py†L17-L105】【F:tests/list_detection_edge_case_test.py†L31-L99】
- **What**: Demand stronger evidence (marker plus delimiter spacing or prior
  context) before classifying continuations.
- **How**: Introduce focused predicates for inline markers and require either a
  confirmed list item on the current line or a colon followed by an actual list
  marker. Validate with the property tests and targeted list metadata checks.

## 4. Finish sentence fusion override handling
- **Why**: `_merge_sentence_fragments` still rejects merges when small chunk
  overrides trigger strict budgets even though the trailing fragment completes a
  sentence, so mid-sentence guards fail the override tests.【F:pdf_chunker/passes/sentence_fusion.py†L124-L333】【F:tests/semantic_chunking_test.py†L82-L231】
- **What**: Allow limited overflow when punctuation is pending and the hard cap
  still permits the merge, while keeping tiny chunk overrides from merging
  endlessly.
- **How**: Reshape the budget decision helper into a dataclass-driven flow that
  evaluates overflow and dense-fragment constraints deterministically, then
  extend the failing override test to assert the new edge case.

## 5. Re-align golden JSONL outputs
- **Why**: The CLI and sample PDF goldens assume the old chunk counts and text
  scaffolding; once the semantic fixes land, the expectations will drift until
  snapshots update. Tests like `epub_cli_regression_test` already pin specific
  chunk IDs, lengths, and prose scaffolding.【F:tests/epub_cli_regression_test.py†L30-L121】
- **What**: Rerun the EPUB and PDF conversions using the approved `--approve`
  workflow to regenerate goldens if and only if the new outputs match the
  intended semantics.
- **How**: Drive the adapters/CLI commands, capture the regenerated JSONL, and
  update only the snapshot fixtures alongside a recorded command log.

## 6. Reconcile readability expectations
- **Why**: The readability test pins an exact Flesch–Kincaid grade and difficulty
  tier from the first PDF golden chunk, so any upstream text change must either
  keep `_compute_readability` aligned or refresh the expectation.【F:pdf_chunker/utils.py†L116-L132】【F:tests/test_readability.py†L11-L27】
- **What**: Compare the new first chunk after semantic fixes with the golden and
  adjust `_compute_readability` rounding/tier logic if necessary.
- **How**: Prefer deterministic adjustments inside `_compute_readability` and
  confirm with `test_readability.py`; only refresh the fixture if the new chunk
  text truly changes the grade.
