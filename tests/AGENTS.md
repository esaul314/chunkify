## tests/AGENTS.md

```markdown
# AGENTS

Test modules validating behavior of parsing, chunking, and enrichment layers.

## Coverage
- `bootstrap/test_quality_gate_scope.py`: Declared lint/typecheck paths, explicit
  supported-Python sessions, missing-interpreter failure policy, CI agreement,
  and syntax parsing of all production modules on the session interpreter
- `pdf_extraction_test.py`: Extractor accuracy and fallback thresholds
- `ai_enrichment_test.py`: Classification correctness and tag injection
- `ai_enrich_pass_test.py`: Pass-level enrichment and metadata injection
- `convert_returns_rows_test.py`: `convert` yields rows for chunk dict payloads
- `semantic_chunking_test.py`: Boundary conditions and oversize protection
- `chunk_pipeline_test.py`: Immutable option/metric updates, provenance precedence,
  one-shot iterator order/laziness, heading attachment, stable chunk IDs, and
  partially applied stages and keyword-index builders
- `sentence_fusion_test.py`: Merge-budget boundary cases, completed/unfinished
  sentence-run flushing, and property-based content/order/input preservation
- `structure_pass_purity_test.py`: Input-artifact preservation across four structure
  passes; partial application and right-biased metric-update composition; explicit
  malformed-metric failures; footer length/count boundaries, document-end search
  precedence, one-shot heading annotation, and page-local list continuation
- `page_exclusion_test.py`: Page range and filter correctness
- `epub_spine_test.py`: Spine index parsing and exclusion logic
- `io_epub_adapter_test.py`: EPUB adapter extraction, stable spine grouping,
  unknown-location ordering, and navigation provenance
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
- Use clear input/output examples
- Avoid asserting on intermediate state unless necessary
- Shell and Python tests must pass independently
- Use idempotent tests.
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
