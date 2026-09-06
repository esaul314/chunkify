## pdf\_chunker/pdf\_chunker/AGENTS.md

`````markdown
# AGENTS

Core modules implementing document processing pipeline passes.
Please, keep this file up-to-date with the latest code structure. If you notice that the code structure has changed, please update this file accordingly.

## Responsibilities
- `core_new.py`: Canonical typed runner used by the CLI and ``pdf_chunker.convert``.
- `cli.py`: Thin Typer/argparse frontends over one normalized conversion runner;
  validates specs before input processing and scopes temporary environment overrides.
- `config.py`: Pipeline-spec loader with explicit no-spec handling and per-pass option validation.
- `core.py`: Frozen legacy ``process_document`` compatibility API with injectable stages.
- `contracts.py`: Typed-in-practice page, block, chunk, and run-metric boundary contracts.
- `framework.py`: Generic artifact carrier, collision-safe pass registry, and legacy runner.
- `parsing.py`: Structural block extraction.
- `text_cleaning.py`: Ligature repair, quote normalization, control-character removal.
- `heading_detection.py`: Infers heading hierarchy.
- `fallbacks.py`: Quality assessment and block-level extraction fallbacks.
- `language.py`: Default language utilities.
- `page_utils.py`: Page range parsing/filtering.
- `epub_parsing.py`: Spine discovery and exclusion.
- `splitter.py`: Enforces chunk boundaries and semantic cohesion.
- `passes/split_semantic.py`: Canonical semantic owner for stitching, boundaries,
  overlap, deduplication, footnote cleanup, and chunk metadata.
- `passes/emit_jsonl.py`: Normalizes canonical chunks for JSONL and validates
  serialized size; unmarked direct callers use the legacy compatibility path.
- `ai_enrichment.py`: Legacy shim delegating to pass/adapter.
- `adapters/io_text.py`: Plain-text transcript ingestion into canonical `page_blocks`.
- `utils.py`: Metadata mapping and helper functions.
- `env_utils.py`: Environment flag helpers.
- `page_artifacts.py`: Header/footer detection utilities, `strip_artifacts`.
- `pdf_parsing.py`: High-level PDF parsing entry point offering streaming block extraction.
- `pdf_blocks.py`: Dataclasses and helpers for page/block extraction and merging.
- `fallbacks.py`: Quality assessment and block-level extraction fallbacks.
- `pymupdf4llm_integration.py`: Optional PyMuPDF4LLM extraction and cleanup.
- `text_processing.py`: Additional text-repair helpers.
- `source_matchers.py`: Matching strategies for locating original source blocks.
- `list_detection.py`: Bullet and numbered list detection helpers.
- `passes/text_parse.py`: Pass registration for plain-text transcript parsing.
- `learned_patterns.py`: Persistence and matching for interactive `--teach` decisions; the CLI
  owns loading/saving while callback wrappers remain in-memory during a run.

## AI Agent Guidance
- Respect strict separation of passes.
- Log fallback reasons and quality metrics.
- Use `text_cleaning.py` transforms before splitting.
- Do not mix EPUB logic into PDF modules.
- Keep transcript parsing light: adapter-side text normalization, pass-side pure chunking, and metadata passthrough into emitted rows.

## Known Issues
- Footnote anchors may appear in the middle of chunks.
- Page exclusion feature is not working reliably.
- Metadata fields sometimes missing when fallback triggers.
- Hyphenated continuation lines starting with bullet markers were not rejoined; this is now fixed in `text_cleaning._join_broken_words`.
- Underscore emphasis is stripped during PyMuPDF4LLM cleanup.
- Split-word merging is restricted to newline or double-space boundaries with frequency checks to avoid collapsing valid phrases.
```

---
