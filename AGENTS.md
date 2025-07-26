# AGENTS.md — Project Guidance for OpenAI Codex

This `AGENTS.md` suite provides comprehensive guidance to OpenAI Codex and other AI agents working across this modular Python library for processing large PDF and EPUB documents. The purpose is to generate high-quality, semantically coherent document chunks with rich metadata for use in local LLM workflows, especially Retrieval-Augmented Generation (RAG).

## Project Structure for OpenAI Codex Navigation

The codebase follows a modular structure rooted in Unix philosophy (single responsibility, composability via interface boundaries):

* `/pdf_chunker/pdf_chunker`: Core logic organized by pass (structural, semantic, enrichment)
* `/config/tags`: YAML-based domain vocabularies for AI enrichment
* `/scripts`: Entry points and CLI utilities
* `/tests`: Functional and transformation validation
* `/tests/utils`: Shared orchestration logic for shell-based testing

## Core Processing Architecture

The project uses a robust **Three-Pass Pipeline**:

1. **Structural Pass** (via `parsing.py`, `heading_detection.py`, `extraction_fallbacks.py`):

   * Extracts visual and typographic structure from PDF/EPUB
   * Uses a simplified hybrid approach: traditional font-based detection + PyMuPDF4LLM’s cleaning
   * Applies fallback logic with automated quality scoring (PyMuPDF → pdftotext → pdfminer.six)

2. **Semantic Pass** (via `splitter.py`):

   * Enforces strict chunk boundaries (8k soft, 25k hard)
   * Avoids mid-paragraph and mid-sentence splits
   * Emits metadata for each chunk (source, location, inferred heading, etc.)

3. **AI Enrichment Pass** (via `ai_enrichment.py`, YAML vocabularies):

   * Applies structured tags from external configuration to enrich chunk semantics
   * Tags are domain-aware (e.g. philosophy, psychology, technical)

## Programming Philosophy

* Use **pure functions** wherever possible
* Write **composable modules** that communicate via clearly defined interfaces
* Prefer **declarative transforms** over procedural logic
* Emphasize **testability**, **predictability**, and **explicit logging**

## Known Issues and Limitations

There are known issues and suspected design flaws and defects:

* Tests may not be fully covering all critical features or edge cases
* Some code descriptions in `AGENTS.md` may be outdated due to recent drift; files may need alignment with reality
* **Footnote defect**: Footnotes are sometimes inserted mid-sentence, splitting one sentence into two lines and harming semantic continuity
* **Hyphenation defect**: Hyphenated words broken across lines (e.g. `con‐ tainer`) are not reconstructed properly — the hyphen and space are retained
* A messy commit may have excluded or bypassed part of the text-cleaning logic. Some code appears to exist for proper text normalization but may not be invoked or applied correctly

## Testing Requirements for OpenAI Codex

Use `pytest` for Python and orchestrated shell tests for system validation:

```bash
pytest tests/
bash tests/run_all_tests.sh
```

Tests must:

* Be **stateless** and **idempotent**
* Validate **transformation results**, not implementation
* Reuse logic via `tests/utils/common.sh`

## Pull Request Guidelines for OpenAI Codex

1. Follow single-responsibility principles
2. Reference affected modules clearly
3. Include regression tests for any fixes
4. Never bundle style-only and logic changes together
5. Prefer clarity over brevity in comments for complex transforms

## Programmatic Checks for OpenAI Codex

```bash
black pdf_chunker/ tests/
flake8 pdf_chunker/
mypy pdf_chunker/
bash scripts/validate_chunks.sh
```

---

