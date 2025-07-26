## pdf\_chunker/pdf\_chunker/AGENTS.md

```markdown
# AGENTS

Contains all core modules implementing the document processing pipeline.

## Responsibilities
- `core.py`: Top-level orchestrator
- `parsing.py`: Structural block parsing from fonts/layouts
- `text_cleaning.py`: Cleans ligatures, quotes, and control characters
- `heading_detection.py`: Extracts heading hierarchy heuristically
- `extraction_fallbacks.py`: Quality-aware fallback logic
- `page_utils.py`: Range filtering and validation
- `epub_parsing.py`: EPUB-specific spine exclusion
- `splitter.py`: Breaks text into safe, bounded, coherent chunks
- `ai_enrichment.py`: Applies tags using YAML vocabularies
- `utils.py`: Metadata mapping, glue logic

## AI Agent Guidance
- Never blur pass boundaries: Structural, Semantic, Enrichment must remain distinct
- Fallback logic must log both failure and quality score
- Avoid embedding AI config logic (always use `config/tags/`)
- Do not mix PDF and EPUB logic in the same module

## Known Issues
- Word reconstruction during text cleaning may be broken — some logic may exist but not be applied
- Footnote content often breaks sentence continuity across JSONL lines
```

---

