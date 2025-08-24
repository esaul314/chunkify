## tests/AGENTS.md

```markdown
# AGENTS

Test modules validating behavior of parsing, chunking, and enrichment layers.

## Coverage
- `pdf_extraction_test.py`: Extractor accuracy and fallback thresholds
- `ai_enrichment_test.py`: Classification correctness and tag injection
- `ai_enrich_pass_test.py`: Pass-level enrichment and metadata injection
- `convert_returns_rows_test.py`: `convert` yields rows for chunk dict payloads
- `semantic_chunking_test.py`: Boundary conditions and oversize protection
- `page_exclusion_test.py`: Page range and filter correctness
- `epub_spine_test.py`: Spine index parsing and exclusion logic
- `process_document_override_test.py`: Callable override injection
- `env_utils_test.py`: Environment flag toggles via `monkeypatch`
- `list_detection_edge_case_test.py`: Bullet list parsing edge cases
- `page_artifact_detection_test.py`: Header/footer removal accuracy
- `page_artifacts_edge_case_test.py`: Page artifact suffix handling
- `footer_artifact_test.py`: Regression for footer and sub-footer removal and multi-page preservation
- `artifact_block_test.py`: Numeric margin block detection conservatism
- `scripts_cli_test.py`: CLI invocation sanity checks
- `test_conversion_epub_cli.py`: CLI EPUB conversion parity with golden output
- `splitter_transform_test.py`: Chunk splitting of cleaned text artifacts
- `text_cleaning_transform_test.py`: Ligature, underscore, and hyphenation normalization
- Duplicate detection thresholds (via `detect_duplicates.py`).

## AI Agent Guidance
- Use clear input/output examples
- Avoid asserting on intermediate state unless necessary
- Shell and Python tests must pass independently
- Use idempotent tests.
- Prefer functional assertions on outputs.
- Leverage ```tests/utils/common.sh` for shared setup.

## Utilities & Fixtures
- `tests/utils/common.sh`: Shell helpers for CLI-oriented tests
- `tests/conftest.py`: Colored output fixtures for test logging
- `pytest.monkeypatch`: Controls environment variables in `env_utils_test.py`

## Formatting
Run repository formatters before committing test changes:

```bash
black tests/
flake8 tests/
mypy pdf_chunker/
```

## Known Issues
- Test suite may not fully reflect current pipeline logic
- Some modules (e.g. `text_cleaning`) are not tested against known real-world PDF defects
- Some custom PDF edge-case tests missing.
- Real-world PDF/EPUB defects may not be simulated.
```

---
