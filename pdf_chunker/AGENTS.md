## pdf\_chunker/pdf\_chunker/AGENTS.md

`````markdown
# AGENTS

Core modules implementing document processing pipeline passes.
Please, keep this file up-to-date with the latest code structure. If you notice that the code structure has changed, please update this file accordingly.

## Responsibilities
- ```core.py`: Orchestrates all passes.
- `parsing.py`: Structural block extraction.
- `text_cleaning.py`: Ligature repair, quote normalization, control-character removal.
- `heading_detection.py`: Infers heading hierarchy.
- `extraction_fallbacks.py`: Fallback strategies with scoring.
- `page_utils.py`: Page range parsing/filtering.
- `epub_parsing.py`: Spine discovery and exclusion.
- `splitter.py`: Enforces chunk boundaries and semantic cohesion.
- `ai_enrichment.py`: Applies YAML-based tags.
- `utils.py`: Metadata mapping and helper functions.

## AI Agent Guidance
- Respect strict separation of passes.
- Log fallback reasons and quality metrics.
- Use `text_cleaning.py` transforms before splitting.
- Do not mix EPUB logic into PDF modules.

## Known Issues
- Footnote anchors may appear in the middle of chunks.
- Page exclusion feature is not working reliably.
- Metadata fields sometimes missing when fallback triggers.
- Hyphenated continuation lines starting with bullet markers were not rejoined; this is now fixed in `text_cleaning._join_broken_words`.
```

---

