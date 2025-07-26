## tests/AGENTS.md

```markdown
# AGENTS

Testing infrastructure for validating transformations and pipeline correctness.

## Coverage
- `pdf_extraction_test.py`: Structural parsing and fallback validation
- `ai_enrichment_test.py`: Tag application and vocabulary use
- `semantic_chunking_test.py`: Chunk size rules and force truncation
- `page_exclusion_test.py`: PDF page-level filtering logic
- `epub_spine_test.py`: EPUB range and spine exclusions
- `run_all_tests.sh`: Full validation orchestration

## AI Agent Guidance
- Write functional tests for every public module
- Tests must avoid disk writes unless explicitly tested
- Compose shared logic using `tests/utils/common.sh`
```

---

