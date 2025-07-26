## tests/AGENTS.md

```markdown
# AGENTS

Test modules validating behavior of parsing, chunking, and enrichment layers.

## Coverage
- `pdf_extraction_test.py`: Extractor accuracy and fallback thresholds
- `ai_enrichment_test.py`: Classification correctness and tag injection
- `semantic_chunking_test.py`: Boundary conditions and oversize protection
- `page_exclusion_test.py`: Page range and filter correctness
- `epub_spine_test.py`: Spine index parsing and exclusion logic
- `run_all_tests.sh`: Orchestrates full suite
- Duplicate detection thresholds (via `detect_duplicates.py`).

## AI Agent Guidance
- Use clear input/output examples
- Avoid asserting on intermediate state unless necessary
- Shell and Python tests must pass independently
- Use idempotent tests.
- Prefer functional assertions on outputs.
- Leverage ```tests/utils/common.sh` for shared setup.

## Known Issues
- Test suite may not fully reflect current pipeline logic
- Some modules (e.g. `text_cleaning`) are not tested against known real-world PDF defects
- Some custom PDF edge-case tests missing.
- Real-world PDF/EPUB defects may not be simulated.
```

---
