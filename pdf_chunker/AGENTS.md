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
- `ai_enrichment.py`: Legacy shim delegating to pass/adapter.
- `utils.py`: Metadata mapping and helper functions.
- `env_utils.py`: Environment flag helpers.
- `page_artifacts.py`: Header/footer detection utilities, `strip_artifacts`.
- `pdf_parsing.py`: High-level PDF parsing entry point.
- `pdf_blocks.py`: Dataclasses and helpers for page/block extraction and merging.
- `fallbacks.py`: Quality assessment and block-level extraction fallbacks.
- `pymupdf4llm_integration.py`: Optional PyMuPDF4LLM extraction and cleanup.
- `text_processing.py`: Additional text-repair helpers.
- `source_matchers.py`: Matching strategies for locating original source blocks.
- `list_detection.py`: Bullet and numbered list detection helpers.

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
- Underscore emphasis is stripped during PyMuPDF4LLM cleanup.
- Split-word merging is restricted to newline or double-space boundaries with frequency checks to avoid collapsing valid phrases.
```

---

