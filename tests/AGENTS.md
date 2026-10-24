## tests/AGENTS.md

```markdown
# AGENTS

Test modules validating behavior of parsing, chunking, and enrichment layers.

## Coverage
- `article_conversion_test.py`: Pure explicit-default resolution, derived sizing,
  idempotence, partial application and input preservation; unsupported/effectful
  settings refuse. Synthetic bound Markdown/EPUB exercise actual runner/JSONL
  parity, environment precedence, intervening-document reruns, detached snapshots,
  metadata/source refusal, runtime/source/dependency invalidation and no writes.
- `conversion_descriptor_test.py`: Pure descriptor/association identity, scoped
  content/settings/metadata invalidation, mapping-order and input-preservation
  laws, immutable snapshots, partial application, strict JSON/version contracts,
  verified Markdown/EPUB logical identity and intervening-document reruns, tampered
  evidence, and a two-note metadata update retaining both references in real rows.
- `verified_markdown_ingress_test.py`: Synthetic proof-bound `.md`/EPUB parity,
  exact UTF-8 reconstruction including cue-like content, unchanged JSONL bytes
  after an intervening document, actual CLI output, Unicode JSON/YAML metadata,
  source/evidence/approval tampering, generated-field precedence, partially
  applied pure byte validation and refusal before injected enrichment.
- `bootstrap/test_quality_gate_scope.py`: Declared lint/typecheck paths, explicit
  supported-Python sessions, missing-interpreter failure policy, CI agreement,
  and syntax parsing of all production modules on the session interpreter;
  missing/empty/blank targets fail before dependency or command effects, with
  ordered diagnostics, one-shot/partially applied target resolution, duplicate
  handling, input-preservation/idempotence laws, literal path spelling, and
  unchanged valid-session command order and dependency profiles
- `pdf_extraction_test.py`: Extractor accuracy and fallback thresholds
- `geometry_contract_test.py`: In-memory strict margin/selection boundaries,
  partial application, stable rounded-bin ranking and cutoff laws, exact bounded
  prompt formatting, input preservation, one-based detection exclusions versus
  zero-based content pages, confidence thresholds, partial extraction on errors,
  and unavailable-backend compatibility
- `ai_enrichment_test.py`: Classification correctness and tag injection
- `ai_enrich_request_budget_test.py`: Complete wire messages, output/margin
  boundaries, immutable caller contract, cache/hash/model mismatch, grounding,
  and explicit enrichment failure without false completion.
  Also covers lossless Unicode windows, nonmonotone counting, guarded repair,
  deterministic aggregation, single-request compatibility and partial failure.
- `ai_enrich_context_fixtures_test.py`: Optional offline hash-pinned privately configured source fixtures, complete request/tokenizer equivalence on82chunks,
  unchanged parent/emitter provenance; absent local fixtures skip, changed ones fail.
  Includes hash-pinned optional oversized-parent/window regression using exact
  cached wire-token counting and offline provider stubs; no real inference.
- `ai_enrich_pass_test.py`: Pass-level enrichment and metadata injection
- `serialized_admission_test.py`: Pre-enrichment serialized JSONL budget,
  lossless escaped-Unicode splitting, metadata refusal, source identity, and
  canonical pipeline reconstruction/emission with bounded record sizes.
- `convert_returns_rows_test.py`: `convert` yields rows for chunk dict payloads
- `semantic_chunking_test.py`: Boundary conditions and oversize protection
- `chunk_pipeline_test.py`: Immutable option/metric updates, provenance precedence,
  one-shot iterator order/laziness, heading attachment, stable chunk IDs, and
  partially applied stages and keyword-index builders
- `sentence_fusion_test.py`: Merge-budget boundary cases, completed/unfinished
  sentence-run flushing, and property-based content/order/input preservation
- `structure_pass_purity_test.py`: Input-artifact preservation across five structure
  passes; partial application and right-biased metric-update composition; explicit
  malformed-metric failures; footer length/count boundaries, document-end search
  precedence, one-shot heading annotation, and page-local list continuation
- `page_artifact_pass_test.py`: Table-before-pattern precedence, whole-block footer
  removal and provenance, one-shot page input, partially applied cleanup, ordered
  count/decision aggregation and page-concatenation laws, runtime option precedence,
  inert callback/cache compatibility, retained metrics, and explicit malformed
  option/metric and invalid-regex failures
- `text_clean_purity_test.py`: String/document nested-metric preservation, missing
  metrics and empty documents, text-only compatibility, retained conditional
  metrics, one-shot page/block ordering, partially applied cleanup, page-concatenation
  and whitespace-idempotence laws, surviving-block style telemetry, runtime pattern
  precedence, callback format/order policy, explicit malformed-container failures,
  and idempotent compatibility-wrapper installation with preserved markers
- `page_exclusion_test.py`: Page range and filter correctness
- `epub_spine_test.py`: Spine index parsing and exclusion logic
- `io_epub_adapter_test.py`: EPUB adapter extraction, stable spine grouping,
  unknown-location ordering, and navigation provenance
- `epub_packaging_recovery_test.py`: Opt-in declared-body recovery, preserved
  reading order/source bytes, strict absent/encrypted-body failures, and
  hash-pinned optional affected books without enrichment or publication.
- `io_pdf_ocr_test.py`: Opt-in text-empty-page OCR, unchanged native/default
  behavior, source provenance, margins/exclusions, explicit engine failures,
  and a generated raster-only PDF integration using installed Tesseract.
- `io_pdf_parallel_ocr_test.py`: Explicit CPU-pool bounds, serial/default parity,
  ordered original-source metadata, errors without partial success, inherited
  worker PGID/affinity and optional real-book exact-payload/timing comparison.
- `io_pdf_poppler_test.py`: Explicit native-only Poppler selection, MuPDF sentinel,
  actual subprocess runner argv/timeout, mocked Poppler command contract, strict
  raw page slots, quality/exclusion/provenance/error/refusal cases, generated PDF
  integration, and an optional real-source adapter test requiring explicit
  path+rawSHA256+expectedpagecount pins. No AI, network, artifact or index work.
- `epub_body_markup_test.py`: Container body recovery without duplicate text;
  optional hash-pinned private-control CPU-only in-memory rows.
- `pdf_parse_pass_test.py`: Shared PDF/EPUB/text parser normalization, one-shot
  input ordering, extraction-quality metrics, and input preservation
- `extraction_fallback_pass_test.py`: Selection from in-memory extraction candidates
  and reporting of the selected engine and reason
- `process_document_override_test.py`: Callable override injection
- `env_utils_test.py`: Environment flag toggles via `monkeypatch`
- `list_detection_edge_case_test.py`: Bullet list parsing edge cases
- `page_artifact_detection_test.py`: Header/footer removal accuracy
- `page_artifacts_edge_case_test.py`: Page artifact suffix handling
- `footer_artifact_test.py`: Regression for footer and sub-footer removal and multi-page preservation
- `artifact_block_test.py`: Numeric margin block detection conservatism
- `scripts_cli_test.py`: CLI invocation sanity checks
- `text_adapter_test.py`: Plain-text transcript ingestion, cue normalization,
  speaker labels, and timestamp provenance
- `test_conversion_epub_cli.py`: CLI EPUB conversion parity with golden output
- `splitter_transform_test.py`: Chunk splitting of cleaned text artifacts
- `text_cleaning_transform_test.py`: Ligature, underscore, and hyphenation normalization
- `utils_test.py`: Source tracing and readability-only sentence normalization,
  including punctuation, indentation, short headings, and blank boundaries
- `rag_pipeline_readiness_test.py`: RAG pipeline overlap, metadata completeness, and LiteLLM preflight
- `corpus_evaluation_test.py`: Corpus manifest loading and category-based quality metrics
- `documentation_consistency_test.py`: Active roadmap, dependency, pipeline, and pass-table alignment
- Duplicate detection thresholds (via `detect_duplicates.py`).
- `parity/test_page_count_regression.py`: Page count parity with sentinel phrase check and TOC dot-leader removal
- `truncation_regression_test.py`: Ensures split_semantic preserves text beyond 25k characters

## AI Agent Guidance
- For resolved article preparation, include `tests/article_conversion_test.py`
  with the descriptor/ingress tests. Test implementation edits through temporary
  source roots and synthetic distribution metadata; never edit real package
  source or installed dependencies just to simulate a changed fingerprint.
  Preparation is not completion or cache admission; resource-byte qualification
  remains separate from Python-source and distribution-version identity.
- For conversion identity changes, add `tests/conversion_descriptor_test.py` to
  the direct article ingress regression run below. Use only synthetic approved
  captures in temporary roots. Comparison does not establish cache completion,
  downstream idempotency or permission to delete a missing note association.
- For direct article ingress, run `pytest tests/verified_markdown_ingress_test.py
  tests/joplin_markdown_preservation_test.py tests/text_adapter_test.py`, then
  `nox -s lint typecheck tests`. Generate synthetic evidence in temporary roots;
  keep actual captures and publication-review findings outside tracked source.
- Use clear input/output examples
- Avoid asserting on intermediate state unless necessary
- Shell and Python tests must pass independently
- Use idempotent tests.
- For quality-gate workflow changes, run
  `pytest tests/bootstrap/test_quality_gate_scope.py tests/documentation_consistency_test.py`,
  followed by `nox -s lint typecheck tests`. Test missing declarations against a
  temporary root with a recording session; never delete real production targets
  to reproduce a scope failure. Target groups must be non-empty, contain no blank
  paths, and resolve completely before session dependency/command effects.
- For geometric-zone changes, run `pytest tests/geometry_contract_test.py
  tests/geometry_test.py tests/bootstrap/test_quality_gate_scope.py`. The contract
  module is lint/format checked and uses in-memory pages without local PDFs.
  Existing real-PDF smoke tests remain separate; no interactive prompt or file
  access is needed to verify selection, scoring, or presentation.
- Prefer functional assertions on outputs.
- For semantic-helper changes, run `pytest tests/chunk_pipeline_test.py
  tests/sentence_fusion_test.py tests/semantic_chunking_test.py
  tests/passes/test_split_semantic_parity.py`; the first two modules are also
  included in the nox lint/format scope.
- Leverage ```tests/utils/common.sh` for shared setup.
- For structure-pass changes, run `pytest tests/structure_pass_purity_test.py
  tests/heading_detect_pass_test.py tests/list_detect_pass_test.py
  tests/test_merge_footers.py tests/test_doc_end_detection.py`. The purity module
  is included in the nox lint/format scope; property tests also verify the metric
  updater's composition, empty-update identity, and input-preservation laws.
- For page-artifact cleanup, also run `pytest tests/page_artifact_pass_test.py
  tests/test_page_artifacts.py tests/text_clean_pass_test.py
  tests/interactive_footer_test.py`. The dedicated regression module is included
  in the nox lint/format scope. Do not add prompting behavior incidentally while
  changing this pure cleanup boundary.
- For text-cleaning boundary changes, run `pytest tests/text_clean_purity_test.py
  tests/text_clean_pass_test.py tests/test_inline_styles.py
  tests/interactive_footer_test.py tests/property/test_clean_idempotence.py`.
  The new purity module is lint/format checked. Use in-memory callback stubs and
  isolate compatibility-wrapper installation with `monkeypatch`; input-artifact
  preservation does not imply that injected callbacks or caches have no effects.

## Utilities & Fixtures
- `tests/utils/common.sh`: Shell helpers for CLI-oriented tests
- `tests/conftest.py`: Colored output fixtures for test logging
- `pytest.monkeypatch`: Controls environment variables in `env_utils_test.py`

## Formatting
Run repository formatters before committing test changes:

```bash
black tests/
ruff check --no-fix tests/
nox -s lint
nox -s typecheck
nox -s tests
```

## Known Issues
- Test suite may not fully reflect current pipeline logic
- Some modules (e.g. `text_cleaning`) are not tested against known real-world PDF defects
- Some custom PDF edge-case tests missing.
- Real-world PDF/EPUB defects may not be simulated.

Note: tests are not likely to get stuck, but they might take a long time to run. It's better to wait.
```

---
