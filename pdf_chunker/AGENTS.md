## pdf\_chunker/pdf\_chunker/AGENTS.md

```markdown
# AGENTS

Houses the core modules orchestrating the pipeline.

## Responsibilities
- `core.py`: Orchestrates the three-pass architecture
- `parsing.py`: Font-based structural parsing
- `text_cleaning.py`: Ligature cleanup and text normalization
- `heading_detection.py`: Heading detection heuristics
- `extraction_fallbacks.py`: Resilient fallback strategies
- `page_utils.py`: Page and range filtering
- `epub_parsing.py`: EPUB spine parsing and exclusion
- `splitter.py`: Chunk boundary enforcement
- `ai_enrichment.py`: Applies YAML-based classification
- `utils.py`: Metadata transformation glue layer

## AI Agent Guidance
- Maintain strict separation between passes
- Do not hardcode tag values — always use external YAMLs
- Prefer pure functions; log and test all exceptions
- Do not conflate EPUB and PDF logic
```

---

