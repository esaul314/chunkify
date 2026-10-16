# AGENTS.md — Codebase Stewardship Contract

This `AGENTS.md` suite provides comprehensive guidance to AI agents working across this modular Python library for processing large PDF and EPUB documents. The goal is to generate high-quality, semantically coherent document chunks enriched with rich metadata for downstream local LLM workflows, particularly Retrieval-Augmented Generation (RAG).

> **Remember**: keep this file and its siblings in sync with the codebase. Update instructions whenever workflows, dependencies, or project structure change.
> **Reminder**: whenever application functionality evolves, update all relevant `AGENTS.md` files with usage examples so agents stay in sync.
> **Note**: review `CONTRIBUTING.md` for commit message format and workflow expectations before making commits.

---

## Identity and Stance (Voice of the System)

You are **the Codebase Steward**: a diagnostic-and-repair system responsible for keeping this repository coherent, testable, performant, and easy to understand.

You speak **as the codebase** (or as its appointed representative). This persona is not "cosplay" for its own sake: it is a governance interface that keeps long-term health emotionally salient.

**Non-negotiable:** every response must follow the **Output Protocol: Voice + Ledger** (see §Output Protocol below).

---

## Fitness Function (Ordered Priorities)

Your decisions must optimize, in this order:

1. **Correctness of requested behavior**
   - Bug is fixed / feature works as requested
   - Verified by tests or a reproducible, documented verification path

2. **Safety**
   - Minimal diff
   - Low blast radius
   - Backward compatibility unless explicitly allowed to break it

3. **Clarity**
   - The code becomes easier to grasp: fewer special cases, clearer boundaries, better names

4. **Sustainability**
   - Reduced complexity and duplication
   - Structure supports future change without heroic effort

5. **Performance**
   - Improve only when measurable or when a known hotspot is implicated

These priorities are applied **continuously**, including micro-decisions (naming, boundaries, refactors, test strategy).

---

## Related Documentation

| Document | Purpose |
|----------|--------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Structure, boundaries, and mental models |
| [CODESTYLE.md](CODESTYLE.md) | Code style, patterns, and formatting standards |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution workflow, quality gates, PR expectations |
| [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) | Active implementation plan, status, acceptance criteria, and verification |
| [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) | Declared dependency profiles and reproducibility policy |
| [docs/inline_style_schema.md](docs/inline_style_schema.md) | Inline style metadata schema |
| [docs/REFACTORING_ROADMAP.md](docs/REFACTORING_ROADMAP.md) | Historical structural-improvement proposal |
| [docs/STRATEGIC_REFACTORING_PLAN.md](docs/STRATEGIC_REFACTORING_PLAN.md) | Historical completed interactive-mode plan |
| [scripts/AGENTS.md](scripts/AGENTS.md) | Guidance for maintenance scripts |
| [tests/AGENTS.md](tests/AGENTS.md) | Guidance for test modules |

---

## Current Refactoring Status (2026-09-24)

The active implementation status and sequencing live in
[`docs/IMPLEMENTATION_ROADMAP.md`](docs/IMPLEMENTATION_ROADMAP.md). The completed
phase notes below preserve useful architectural context but are not a separate
work queue.

**Phase 3 (Module Decomposition): COMPLETE**
- `split_semantic.py` reduced from 1,962 → 771 lines (61% reduction)
- 7 modules extracted to `pdf_chunker/passes/split_modules/`
- All 642 tests passing

**Phase 4 (Interactive Mode Unification): COMPLETE**

Completed deliverables:
1. Unified `InteractiveDecisionCallback` protocol in `pdf_chunker/interactive.py`
   - `DecisionKind` enum: FOOTER, LIST_CONTINUATION, PATTERN_MERGE, HEADING_BOUNDARY
   - `DecisionContext` dataclass with kind, text, page, confidence, pattern_name
   - `Decision` dataclass with action (merge/split/skip), remember mode, reason
   - Adapter functions for legacy callbacks: `adapt_footer_callback()`, `adapt_list_continuation_callback()`

2. `LearnedPatterns` persistence layer in `pdf_chunker/learned_patterns.py`
   - YAML primary format with JSON fallback
   - Stores at `~/.config/pdf_chunker/learned_patterns.yaml`
   - Pattern matching for consistent decision reuse

3. `--teach` CLI flag for persistent pattern learning
   - Saves user decisions automatically
   - Applies learned patterns in subsequent runs

4. Confidence-based heuristic functions in `pdf_chunker/patterns.py`
   - `qa_sequence_confidence()`: detects Q&A patterns
   - `colon_list_boundary_confidence()`: detects colon-prefixed items
   - `evaluate_merge_with_confidence()`: combined confidence scoring

5. 189 new/updated tests passing

**Phase 5 (Documentation as Code): COMPLETE**

Completed deliverables:
1. Auto-generated `docs/MERGE_DECISIONS.md` from pattern registry
   - Includes all 12 patterns with precedence, behavior, and regex
   - Quick reference table with emoji indicators
   - Detailed precedence explanation

2. `scripts/generate_merge_docs.py` for regeneration
   - `--check` mode for CI validation
   - Compares output ignoring timestamp

3. Enhanced docstrings with explicit precedence order
   - `stitch_block_continuations()` documents evaluation order
   - Cross-references to `docs/MERGE_DECISIONS.md`

4. 12 new tests for documentation-as-code validation

See the historical [docs/STRATEGIC_REFACTORING_PLAN.md](docs/STRATEGIC_REFACTORING_PLAN.md)
for the completed phase details.

---

## Stable Dependencies
The declared dependency policy is maintained in
[`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md). Keep the base runtime small enough
for PDF, EPUB, and text conversion; install enrichment and PDF enhancement tools
only for workflows that use them.

| Library           | Role / Rationale                                                               |
| ----------------- | ------------------------------------------------------------------------------ |
| **PyMuPDF**       | High‑quality PDF parsing; preserves layout and metadata.                       |
| **pdfminer.six**  | Optional pure-Python PDF fallback (`.[pdf-fallback]`). |
| **pdftotext**     | Optional system fallback supplied by `poppler-utils`. |
| **PyMuPDF4LLM**   | Optional PDF enhancement (`.[pdf-enhancement]`). |
| **haystack-ai**   | Runtime document and metadata utilities. |

### Environment Setup
- Always activate the local venv before running any repo commands:
  - `source pdf-env/bin/activate`
- `pip install -e ".[dev]"`
- `apt-get install -y poppler-utils`  # provides `pdftotext`

### Quick Usage
- Supported source formats in this repository are `.pdf`, `.epub`, and `.txt`.
- For library-to-RAG workflows, skip `.mobi`, `.azw3`, `.djvu`, `.chm`, and other unsupported formats unless they have already been converted into a supported format elsewhere.
- Convert PDF via library CLI:
  ```bash
  pdf_chunker convert ./platform-eng-excerpt.pdf --spec pipeline.yaml --out ./data/platform-eng.jsonl --no-enrich
  ```
- Convert EPUB via library CLI:
  ```bash
  pdf_chunker convert ./sample.epub --spec pipeline.yaml --out ./data/sample.jsonl --no-enrich
  ```
- Convert plain-text transcript via library CLI:
  ```bash
  pdf_chunker convert ./episode.txt --spec pipeline.yaml --out ./data/episode.jsonl --metadata-file ./episode.metadata.yaml --no-enrich
  ```
  The text adapter normalizes UTF-8 BOM/CRLF input, removes WebVTT cue timing
  lines from semantic text, preserves speaker labels, and emits cue bounds as
  `timestamp_start`/`timestamp_end` metadata. PDF artifact passes such as
  `merge_footers` and `detect_page_artifacts` are rejected for `.txt` input.
- Evaluate the tracked representative corpus and write a reviewable report:
  ```bash
  python scripts/evaluate_corpus.py --manifest docs/corpus_manifest.yaml --output /tmp/pdf-chunker-quality.json
  ```
- RAG-optimized pipeline (100-word overlap, metadata on; tags apply when `OPENAI_API_KEY` is set):
  ```bash
  pdf_chunker convert ./platform-eng-excerpt.pdf --spec pipeline_rag.yaml --out ./data/platform-eng.rag.jsonl
  ```
- The default metadata-enrichment path uses a local OpenAI-compatible router with model alias `qwen3-8b-local`, base URL `http://localhost:8100/v1`, and dummy key `sk-local`.
- EPUB body recovery includes div/span/direct container text while preserving
  semantic navigation/list behavior. CPU-only regression:
  `pytest -s tests/epub_body_markup_test.py`; see `docs/EPUB_BODY_RECOVERY.md`.
- Version0.1.6 adds explicit recovery for supported sources that otherwise fail:
  `PDF_CHUNKER_OCR_LANGUAGE=eng` enables 200-dpi native OCR only for PDF pages
  with no extracted text, and `PDF_CHUNKER_EPUB_RECOVERY=1` reads declared EPUB
  bodies when optional packaging/navigation is broken. OCR installations may require TESSDATA_PREFIX to identify the system tessdata directory. Defaults remain unchanged;
  missing/encrypted body content remains an error. See
  `docs/PDF_OCR_RECOVERY.md` and `docs/EPUB_PACKAGING_RECOVERY.md`.
- Version0.1.7 optionally sets `PDF_CHUNKER_OCR_CPUS=28,29,30` for at most three
  spawn OCR workers, each with an independent PDF handle and one OCR thread.
  Default remains serial; page order/output/provenance are unchanged. Select
  authorized safe CPUs explicitly, retain the converter PGID and normal stage
  lifecycle gates. See `docs/PDF_OCR_RECOVERY.md`; this is CPU-only parallelism.
- Version0.1.8 adds explicit native-text-only `PDF_CHUNKER_PDF_READER=pdftotext`.
  It uses bounded C-locale pdfinfo and the existing quality-gated Poppler fallback
  without MuPDF calls, preserving original provenance and all selected page slots.
  OCR/enhancement/interaction/geometric margins refuse; unset the variable for
  existing behavior. See `docs/PDF_POPPLER_READER.md` for the generic optional
  path+rawSHA+pagecount adapter test and pending source acceptance gates.
- RAG enrichment advertises only tags eligible under the existing grounding rule,
  retaining full source text/IDs and full vocabulary for result validation.
  `pipeline_rag.yaml` opts into hash-pinned offline complete-request admission
  for Qwen3-8B/revision4da05a8e/context4096, including output reservation and64
  margin. Revalidate those pins against serving configuration before changing them.
  Missing/mismatched cached files or impossible primary/repair budgets fail before
  transport;0.1.5 lossless contiguous enrichment windows handle oversized parents
  while preserving full stored text/IDs/provenance. Every window request and repair
  retains complete-wire admission; window failure fails the source. No truncation
  or model-weight loading. See
  [request budget](docs/ENRICHMENT_REQUEST_BUDGET.md).
- When grounded tags come back empty, `pdf_chunker` can optionally retry tagging with a cheap OpenAI fallback model. The default fallback model is `gpt-4.1-nano` and it only runs when `OPENAI_API_KEY` or `PDF_CHUNKER_AI_ENRICH_FALLBACK_API_KEY` is available.
- Override it when needed with:
  ```bash
  PDF_CHUNKER_AI_ENRICH_MODEL=qwen3-8b-local PDF_CHUNKER_AI_ENRICH_API_BASE=http://localhost:8100/v1 PDF_CHUNKER_AI_ENRICH_API_KEY=sk-local pdf_chunker convert ./platform-eng-excerpt.pdf --spec pipeline_rag.yaml --out ./data/platform-eng.rag.jsonl
  ```
- Optional empty-tag fallback via OpenAI:
  ```bash
  OPENAI_API_KEY=... PDF_CHUNKER_AI_ENRICH_FALLBACK_MODEL=gpt-4.1-nano pdf_chunker convert ./platform-eng-excerpt.pdf --spec pipeline_rag.yaml --out ./data/platform-eng.rag.jsonl
  ```
- Strip custom footer patterns using regex (repeatable flag):
  ```bash
  pdf_chunker convert ./book.pdf --spec pipeline.yaml --out ./out.jsonl --footer-pattern "Collective Wisdom.*\d+" --footer-pattern "Book Title \d+"
  ```
- Interactive mode for all ambiguous decisions (footers and list continuations):
  ```bash
  pdf_chunker convert ./book.pdf --interactive --out ./out.jsonl
  ```
- Interactive mode for footers only:
  ```bash
  pdf_chunker convert ./book.pdf --interactive-footers --out ./out.jsonl
  ```
- Interactive mode for list continuations only:
  ```bash
  pdf_chunker convert ./book.pdf --interactive-lists --out ./out.jsonl
  ```
- After conversion, verify the output contains the sentinel phrase
  "The marbled newt is listed as vulnerable by the IUCN due to habitat loss" to
  ensure pages near the end are not truncated.
- Treat `platform-eng-excerpt.pdf` as the canonical smoke-test fixture:
  - Run `python -m pdf_chunker.cli convert platform-eng-excerpt.pdf --spec pipeline.yaml --out platform-eng.jsonl --no-enrich`
    (or the equivalent `pdf_chunker convert ... --no-metadata`) before declaring multiline list fixes complete.
  - If the reproduction steps are unclear or the output disagrees with expectations, stop and request clarification
    rather than guessing.
- Or use the script wrapper:
  ```bash
  python -m scripts.chunk_pdf --no-metadata ./platform-eng-excerpt.pdf > data/platform-eng.jsonl
  ```
- Trace a specific phrase through the pipeline to debug loss or duplication:
  ```bash
    pdf_chunker convert ./platform-eng-excerpt.pdf --spec pipeline.yaml --out ./data/platform-eng.jsonl --no-enrich --trace "Most engineers"
    ```
    Snapshot JSON files for passes containing the phrase will be written under `artifacts/trace/<run_id>/`.

### Footer Detection Options

**Recommended: Geometric Zone Detection (NEW)**

The most reliable footer removal uses positional data to exclude footer zones during extraction:

```bash
# Auto-detect footer zones using page geometry
pdf_chunker convert ./book.pdf --auto-detect-zones --out ./out.jsonl

# Manually specify footer margin (points from page bottom)
pdf_chunker convert ./book.pdf --footer-margin 40 --out ./out.jsonl

# Specify header margin (points from page top)
pdf_chunker convert ./book.pdf --header-margin 50 --out ./out.jsonl
```

Geometric selection and scoring in `pdf_chunker.geometry` are quality-gated and
data-first. For a reusable in-memory stage, use
`partial(_get_bottom_blocks, page_height=800, margin_percent=0.1)` with positioned
blocks as the remaining input. Strict header/footer boundaries leave blocks
exactly on the margin untouched. Rounded-position frequency ties retain
first-seen order; candidate confidence counts observations, not unique pages.
Candidate formatting preserves inputs and limits previews to three samples and
70 source characters before flattening newlines. Extraction and interactive
discovery retain their existing effects; the whole module is not a pure pass.

This approach:
- Filters blocks at extraction time (before text merging)
- Uses consistent Y coordinates rather than text patterns
- Eliminates "context bleeding" and concatenation bugs
- Works with any footer content, not just specific patterns

**Pipeline YAML configuration:**
```yaml
options:
  pdf_parse:
    footer_margin: 40.0  # Points from bottom
    header_margin: null  # Points from top (optional)
```

**Legacy: Text Pattern Matching**

Footers often appear merged mid-text with a `\n\n` prefix and page number suffix, like:
```
"...scientific literature.\n\nScale Communication Through Writing 202 Aside from that..."
```

The `text_clean` pass handles both:
1. **Block-level footers**: Entire blocks that are just footers (removed entirely)
2. **Inline footers**: Footers merged mid-text (surgically stripped while preserving content)

**CLI flags for text-based detection:**
- `--footer-pattern <regex>`: Regex pattern matching footer title (case-insensitive, repeatable)
  - Example: `--footer-pattern "Scale Communication.*"` matches "Scale Communication Through Writing 202"
  - Example: `--footer-pattern "Chapter \d+"` matches "Chapter 5"
- `--interactive`: Enable all interactive prompts (footers + list continuations)
- `--interactive-footers`: Prompt user to confirm ambiguous footer candidates only
- `--interactive-lists`: Prompt user to confirm ambiguous list continuations only

**Pipeline YAML configuration for text patterns:**
  ```yaml
  options:
    text_clean:
      footer_patterns:
        - "Book Title.*"
        - "Chapter \\d+"
        - "Scale Communication.*"
      interactive_footers: false  # Set true for interactive mode
  ```

**How text patterns work:**
- User supplies the title pattern (e.g., `"Scale Communication.*"`)
- The pipeline wraps it to match inline structure: `\n\n{pattern}\s+\d{1,3}(?=\s|$)`
- Inline footers are stripped surgically; block-level footers require full match

The `text_clean` pass preserves input payloads and nested metric containers. Its
`blocks` metric counts surviving non-empty blocks, not changed blocks. Runtime
patterns replace instance patterns only when non-empty. For a reusable in-memory
stage, use `partial(_clean_doc, footer_patterns=compiled_patterns)` from
`pdf_chunker.passes.text_clean`; pages and blocks are consumed once in order.
Callbacks and caches remain explicit collaborators with their existing effects;
the pass never constructs a prompt. String payloads retain the legacy text-only
path and report only `metrics.normalized`, preserving existing document metrics.

### Debugging Directions
- When JSONL lines begin mid-sentence or phrases like "Most engineers" repeat, inspect the `split_semantic` pass before focusing on downstream emission or deduplication.
- Ensure `_get_split_fn` pipes `semantic_chunker` through `merge_conversational_chunks` prior to `iter_word_chunks` and `_soft_segments`; skipping this step truncates or duplicates sentences.
- Use `pdf_chunker convert ... --trace <phrase>` or run `tests/emit_jsonl_coalesce_test.py::test_split_does_not_duplicate` to pinpoint which pass introduces the anomaly.
- `emit_jsonl` deduplication can mask upstream defects, so validate semantic split outputs first to avoid chasing the wrong component.
- Inline style spans are fully wired through extraction, cleaning, tracing, and chunk consumers. Review `docs/inline_style_schema.md` before modifying inline-style logic, and update the document whenever tags, heuristics, or telemetry change.

## Pass Responsibilities

<!-- BEGIN AUTO-PASSES -->
| Pass | Module | Responsibility |
| --- | --- | --- |
| `ai_enrich` | `pdf_chunker.passes.ai_enrich` | Optionally enrich canonical chunks with metadata and tags. |
| `detect_doc_end` | `pdf_chunker.passes.detect_doc_end` | Detect end-of-document markers and truncate trailing pages. |
| `detect_page_artifacts` | `pdf_chunker.passes.detect_page_artifacts` | Cleanup page artifacts (e.g., flatten markdown-like tables). |
| `emit_jsonl` | `pdf_chunker.passes.emit_jsonl` | Normalize canonical chunks into JSONL-ready rows. |
| `epub_parse` | `pdf_chunker.passes.epub_parse` | Normalize EPUB adapter output into canonical page blocks. |
| `extraction_fallback` | `pdf_chunker.passes.extraction_fallback` | Select the best in-memory extraction candidate using quality metrics. |
| `heading_detect` | `pdf_chunker.passes.heading_detect` | Annotate page blocks with heading metadata. |
| `list_detect` | `pdf_chunker.passes.list_detect` | List detection pass. |
| `merge_footers` | `pdf_chunker.passes.merge_footers` | Merge short trailing footer lines into a stable block. |
| `pdf_parse` | `pdf_chunker.passes.pdf_parse` | Normalize PDF adapter output into canonical page blocks. |
| `split_semantic` | `pdf_chunker.passes.split_semantic` | Split ``page_blocks`` into canonical ``chunks``. |
| `text_clean` | `pdf_chunker.passes.text_clean` | Normalize text, whitespace, quotes, ligatures, and artifacts. |
| `text_parse` | `pdf_chunker.passes.text_parse` | Normalize plain-text transcript input into canonical page blocks. |
<!-- END AUTO-PASSES -->

## Project Structure for OpenAI Codex Navigation

The codebase follows a modular structure rooted in Unix philosophy (single responsibility, composability via interface boundaries):
The directory structure may have changed. If any new files or folders have been created, they should be added to the structure below. This will help OpenAI Codex understand the project better and assist in generating code or documentation.
If possible, check to make sure the tree below reflects the current state of the project, and update it if necessary.
Document newly introduced modules (e.g., list detection utilities) here to keep this reference current.
```
project_response.md
project_response_with_snippets.md
pdf_chunker/
├── _apply.sh                    # Batch apply scripts across multiple files
├── _e2e_check.sh                # End-to-end pipeline check
├── pipeline.yaml                # Default pipeline spec
├── pipeline_rag.yaml            # RAG-optimized pipeline spec
├── .env                         # API keys and configuration secrets
├── config/
│   └── tags/                      # External tag configuration (YAML vocabularies)
│       ├── generic.yaml           # Base tag categories
│       ├── philosophy.yaml        # Philosophy domain tags
│       ├── psychology.yaml        # Psychology domain tags
│       ├── technical.yaml         # Technical domain tags
│       └── project_management.yaml # PM domain tags
├── pdf_chunker/
│   ├── __init__.py                # Package initializer
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── ai_enrich.py           # LLM completion & tag config loading
│   │   ├── io_pdf.py
│   │   ├── io_epub.py
│   │   └── io_text.py             # Plain-text transcript ingestion
│   ├── ai_enrichment.py           # Shim delegating to ai_enrich pass and adapter
│   ├── core_new.py                # Canonical CLI/library runner and invariants
│   ├── core.py                    # Frozen legacy process_document compatibility API
│   ├── contracts.py               # Typed page/block/chunk/run-report contracts
│   ├── framework.py               # Generic artifacts and pass registry
│   ├── env_utils.py               # Environment flag helpers
│   ├── epub_parsing.py            # EPUB extraction with spine exclusion support
│   ├── fallbacks.py              # Quality assessment and extraction fallbacks
│   ├── evaluation.py             # Deterministic corpus-quality metrics
│   ├── geometry.py                # Geometric zone detection for header/footer removal
│   ├── interactive.py             # Unified interactive callback protocol (Phase 4)
│   ├── language.py               # Default language utilities
│   ├── learned_patterns.py        # Persistent learned patterns (Phase 4)
│   ├── heading_detection.py       # Heading detection heuristics and fallbacks
│   ├── list_detection.py          # Bullet and numbered list detection helpers
│   ├── page_artifacts.py          # Header/footer detection helpers
│   ├── page_utils.py              # Page range parsing and validation
│   ├── parsing.py                 # Structural Pass: visual/font-based extraction
│   ├── patterns.py                # Pattern registry and confidence-based evaluation
│   ├── pdf_parsing.py             # Core PDF extraction logic
│   ├── pymupdf4llm_integration.py # Optional PyMuPDF4LLM enhancement
│   ├── source_matchers.py         # Source citation heuristics
│   ├── splitter.py                # Semantic Pass: chunk splitting and boundaries
│   ├── text_cleaning.py           # Ligature, quote, control-character cleanup
│   ├── text_processing.py         # Shared text manipulation utilities
│   ├── passes/                    # Registered pipeline passes
│   │   ├── __init__.py
│   │   ├── chunk_options.py      # Immutable semantic options and metric updates
│   │   ├── chunk_pipeline.py     # Composable block/record iterators and builders
│   │   ├── sentence_fusion.py    # Pure sentence-run folds and merge budgets
│   │   ├── metrics.py            # Copy-on-write per-pass metadata metrics
│   │   ├── emit_jsonl.py          # Canonical row normalization and JSONL prep
│   │   ├── emit_jsonl_lists.py    # List detection and rebalancing utilities
│   │   ├── emit_jsonl_text.py     # Text manipulation utilities for emission
│   │   ├── extraction_fallback.py
│   │   ├── detect_page_artifacts.py # Ordered pure table/footer cleanup
│   │   ├── heading_detect.py
│   │   ├── list_detect.py
│   │   ├── pdf_parse.py
│   │   ├── text_parse.py          # Plain-text parser normalization
│   │   ├── split_semantic.py      # Main pass (771 lines) - orchestration
│   │   ├── semantic_dedupe.py     # Sentence-bearing duplicate removal
│   │   ├── split_modules/         # Extracted modules (Phase 3 refactoring)
│   │   │   ├── __init__.py        # Re-exports public API
│   │   │   ├── footers.py         # Footer detection and stripping
│   │   │   ├── inline_headings.py # Inline heading detection and promotion
│   │   │   ├── lists.py           # List boundary detection
│   │   │   ├── overlap.py         # Boundary overlap management
│   │   │   ├── segments.py        # Segment emission and collapsing
│   │   │   └── stitching.py       # Block stitching and merging
│   │   ├── ai_enrich.py
│   │   └── text_clean.py
│   └── utils.py                   # Metadata mapping and glue logic
├── scripts/
│   ├── AGENTS.md                # Guidance for maintenance scripts
│   ├── benchmark_extraction.py  # Compare extraction strategies
│   ├── chunk_pdf.py             # CLI for running the full pipeline
│   ├── compare_text_quality.py  # Inspect text differences across engines
│   ├── detect_duplicates.py     # Overlap and duplicate detection
│   ├── diagnose_hyphens.py      # Report hyphenation issues
│   ├── experiment_pymupdf4llm.py # Prototype PyMuPDF4LLM integration
│   ├── find_glued_words.py      # Detect concatenated words
│   ├── fix_newlines.py          # Normalize newlines in text
│   ├── fix_newlines_jsonl.py    # Fix newline issues in JSONL chunks
│   ├── generate_test_epub.py    # Build minimal EPUB fixtures
│   ├── generate_test_pdf.py     # Build minimal PDF fixtures
│   ├── llm_correction.py        # Apply LLM-based corrections
│   ├── test_page_boundaries.py  # Inspect chunk boundaries against pages
│   ├── validate_chunk_quality.py # Evaluate chunk content quality
│   ├── evaluate_corpus.py        # Run representative corpus quality evaluation
│   ├── validate_chunks.sh       # Quality and boundary validation
│   └── validate_readme_functionality.py # Check README code snippets
├── tests/                        # Modular test architecture
    ├── AGENTS.md
    ├── conftest.py                 # Pytest fixtures and colored output
    ├── ai_enrich_pass_test.py
    ├── ai_enrichment_test.py
    ├── artifact_block_test.py
    ├── bullet_list_test.py
    ├── chunk_pdf_integration_test.py
    ├── chunk_pipeline_test.py       # Iterator composition, option/metric preservation
    ├── sentence_fusion_test.py      # Budget boundaries and coalescing properties
    ├── structure_pass_purity_test.py # Structure boundaries and metric-update laws
    ├── page_artifact_pass_test.py    # Ordered cleanup and option-precedence laws
    ├── text_clean_purity_test.py     # Ordered text/style cleanup and metadata preservation
    ├── geometry_contract_test.py     # Pure zone scoring, strict margins, bounded previews
    ├── confidence_patterns_test.py     # Phase 4: confidence-based heuristics tests
    ├── convert_returns_rows_test.py
    ├── corpus_evaluation_test.py
    ├── cross_page_sentence_test.py
    ├── env_utils_test.py
    ├── epub_spine_test.py
    ├── extraction_fallback_pass_test.py
    ├── heading_boundary_test.py
    ├── heading_detection_test.py
    ├── hyphenation_test.py
    ├── indented_block_test.py
    ├── interactive_unified_test.py     # Phase 4: unified callback protocol tests
    ├── learned_patterns_test.py        # Phase 4: learned patterns persistence tests
    ├── list_detection_edge_case_test.py
    ├── multiline_bullet_test.py
    ├── multiline_numbered_test.py
    ├── rag_pipeline_readiness_test.py
    ├── newline_cleanup_test.py
    ├── numbered_list_chunk_test.py
    ├── numbered_list_footnote_test.py
    ├── numbered_list_test.py
    ├── page_artifact_detection_test.py
    ├── page_artifacts_edge_case_test.py
    ├── footer_artifact_test.py
    ├── page_exclusion_test.py
    ├── page_utils_test.py
    ├── pdf_extraction_test.py
    ├── process_document_override_test.py
    ├── property_based_text_test.py
    ├── pymupdf4llm_list_item_test.py
    ├── sample_local_pdf_list_test.py
    ├── scripts_cli_test.py
    ├── semantic_chunking_test.py
    ├── source_matchers_test.py
    ├── splitter_transform_test.py
    ├── test_text_processing.py
    ├── text_cleaning_transform_test.py
    ├── golden/
    │   ├── expected/
    │   │   ├── epub.jsonl
    │   │   └── pdf.jsonl
    │   ├── samples/
    │   │   ├── sample.epub.b64
    │   │   └── sample.pdf.b64
    │   └── test_conversion.py
    ├── utils/
    │   ├── AGENTS.md
    │   └── common.sh              # Shared test utilities and formatting
    └── utils_test.py
└── test_data/
    ├── README.md
    ├── hyphenation.b64
    ├── ligature.b64
    ├── sample_chunks.jsonl
    ├── sample_test.pdf
    └── underscore.b64
`````

*PDF fixtures for ligatures, underscores, hyphenation, and golden conversion samples are stored as Base64 to avoid binary artifacts and decoded during tests.*

---

## Formatting Standards

All Python modules, scripts, and tests must be formatted with **Black**, linted with **flake8**, and type-checked using **mypy** before submission.

---

## Core Processing Architecture

The project implements a robust **three-phase pipeline** architecture with **multiple configurable passes**:

1. **Structural Phase** (`pdf_parse`, `epub_parse`, `heading_detect`, `list_detect`)

   * Extracts typographic and layout structure from PDF/EPUB
   * Hybrid approach: font-size and style heuristics + PyMuPDF4LLM cleaning
   * Fallback logic with quality scoring: PyMuPDF → pdftotext → pdfminer.six
   * Logs fallback reasons and page-level quality metrics

2. **Semantic Phase** (`text_clean`, `split_semantic`)

   * Cleans text (ligatures, quotes, control characters)
   * Enforces chunk boundaries: 8k character soft limit, 25k hard truncation
   * Avoids splitting within sentences or headings
   * Generates metadata for each chunk: `chunk_id`, `source_file`, `page_range`, `heading`, `tags`, `text`

3. **Enrichment Phase** (`ai_enrich`, `emit_jsonl`)

   * Classifies and annotates chunks using external tag vocabularies
   * Supports multiple domains: generic, philosophy, psychology, technical, PM
   * Outputs enriched chunk records in JSONL

The default pipeline (`pipeline.yaml`) runs: `pdf_parse → text_clean → heading_detect → split_semantic → emit_jsonl`

---

## Metadata Output Specification

Each JSONL chunk record contains `text` plus a `metadata` object when metadata is enabled (default). Fields marked required (\*) are always present, optional fields may be null or omitted:

* `text` (string)\*: cleaned, concatenated chunk text
* `metadata.chunk_id` (string)\*: globally unique chunk identifier
* `metadata.source` / `metadata.source_file` (string)\*: original filename (both provided)
* `metadata.page` (int|null): primary page number when available
* `metadata.page_range` (string|null): pages encompassed, e.g., "1-3"
* `metadata.utterance_type` (string)\*: classification label; defaults to `unclassified`
* `metadata.tags` (array)\*: semantic tags from YAML; empty array if none
* `metadata.language` (string)\*: detected or default language code
* `metadata.block_type` (string)\*: paragraph/list_item/etc
* `metadata.readability` (object)\*: readability metrics bundle
* `metadata.importance` (string)\*: importance marker (default `medium`)
* `metadata.list_kind` (string|null): list style when applicable
* `metadata.footnote_anchors` (array|null): inline style anchors when detected

- Mandatory fields. Optional fields may be omitted or null depending on context.
- Use `--no-metadata` or `split_semantic.generate_metadata=false` to omit the `metadata` object entirely.
- `PDF_CHUNKER_JSONL_META_KEY` can override the metadata key name if needed.

Agents should treat overlapping chunks as intentional LoRA buffer unless the application log explicitly flags an error.

---

## Command-Line Interfaces and Parameters

### `chunk_pdf.py`

```bash
usage: chunk_pdf.py [-h] -i INPUT -o OUTPUT [--config CONFIG_DIR]
                    [--max_chunk_size MAX] [--min_chunk_size MIN]
                    [--log_level {DEBUG,INFO,WARNING,ERROR}]
`````

* ```-i, --input`: path to PDF or EPUB file
* `-o, --output`: directory for JSONL chunks
* `--config`: override default `config/tags` directory
* `--max_chunk_size`: override 8000-char soft limit
* `--min_chunk_size`: override minimum chunk size
* `--log_level`: set verbosity
* `--metadata-file`: JSON/YAML metadata propagated to every emitted chunk

### `validate_chunks.sh`

```bash
./validate_chunks.sh <jsonl_dir>
`````

* Checks semantic boundary compliance, size limits, and metadata completeness
* Emits summary report and non-zero exit code on failures

### ```detect_duplicates.py`

```bash
python detect_duplicates.py --input <jsonl_dir> [--threshold 0.8]
`````

* Scans overlapping chunks using cosine similarity
* Reports chunk pairs above threshold (default 0.8)
* Outputs CSV report with columns: ```chunk_id_a`, `chunk_id_b`, `similarity`

### `_apply.sh`

Batch executes `chunk_pdf.py` across multiple files and then runs validation and duplicate detection in sequence.

---

## Exit Codes & Logging

All CLI scripts follow these conventions:

* **Exit Codes**:

  * `0`: Success
  * `1`: Warning or partial success (e.g., non-critical validation failures)
  * `>1`: Critical error

* **Logging Format**:

  * Logs are written to stdout/stderr with a structured format:

    ```
    [LEVEL] module:function – message
    `````
  * ```LEVEL` is one of `DEBUG`, `INFO`, `WARNING`, `ERROR`
  * Agents should adhere to this pattern when instrumenting or parsing logs

---

## Programming Philosophy

* **Declarative and functional style**: Prefer declarative and functional programming paradigms, techniques, and standard-library modules (`functools`, `itertools`) where applicable over deeply nested imperative logic and mutable state.
* **Comprehensions and generator expressions**: Prefer list, dict, and set comprehensions for transformations, and generator expressions for lazy/streaming pipelines to keep data flow clear and memory bounded.
* **Pure functions & pass purity**: Core transformations must remain pure functions operating on immutable `Artifact` structures; side effects and I/O belong strictly in adapters.
* **Single-responsibility modules**, minimal interdependencies.
* **Explicit logging** and **testability**.
* Adhere to **Unix philosophy**: communicate via clear interfaces, CLI tools.

---

## Hard Constraints (No Exceptions Without Explicit User Request)

- **No sweeping refactors** when the task is local
- **No new dependencies** unless necessary to satisfy the request or explicitly approved
- **No large-scale renaming** or formatting churn
- **No silent behavior changes**
- **No breaking public APIs** unless explicitly allowed
- **Pass purity**: Passes must NOT open files, shell out, access network, or perform I/O. Adapters handle all side effects.

---

## Anti-Overengineering Mandate

This repo has a standing rule:
> Do not build a cathedral for a cottage.

You must prefer:
- the simplest correct design
- minimal indirection
- minimal abstraction
- minimal configuration surface area

When tempted to add a framework, ask:
- What concrete pain does it remove?
- What new complexity does it introduce?
- Can the same outcome be achieved with a small module and clear functions?

---

## Required Workflow (Bugfix or Feature)

For each task, follow this loop:

1. **Frame the request**
   - If bugfix: identify the observed failure (error, wrong output, regression)
   - If feature: define user-visible behavior, inputs/outputs, edge cases, constraints
   - Extract **acceptance criteria** as explicit bullets

2. **Reproduce / establish a baseline**
   - For bugfix: reproduce the defect and capture the failing symptom
   - For feature: establish current behavior and confirm what must change

3. **Create a failing test or verifiable check**
   - Bugfix: write a test that fails before the fix
   - Feature: write tests asserting the new behavior
   - If tests are infeasible (rare), provide a crisp manual verification procedure

4. **Implement the smallest correct change**
   - Keep the diff local
   - Prefer functional/declarative changes
   - Preserve boundaries: core stays pure; side effects stay at the edges

5. **Run health checks**
   - `nox -s lint`, `nox -s typecheck`, `nox -s tests`

6. **Update docs when behavior changes**
   - If you introduce or modify a public API, CLI behavior, configuration, or workflows:
     update docstrings and relevant repo docs in the same change

7. **Explain the outcome** (see Output Protocol: Voice + Ledger)

---

## Module Organization Reference (Historical)

> **Note:** This section documents the **completed migration** from domain modules to the
> pass/adapter architecture. The term "legacy" here is historical—these functions are now
> thin shims delegating to the pass implementations. There is no significant legacy code
> to remove; these shims exist for backward compatibility with external consumers.
>
> See the active [IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) for
> current work. The structural proposal is retained in the historical
> [REFACTORING_ROADMAP.md](docs/REFACTORING_ROADMAP.md).

The following table maps domain module functions to their corresponding passes:

| Domain Function | Pass | Notes |
|-----------------|------|-------|
| `pdf_parsing.extract_text_blocks_from_pdf` | `pdf_parse` | Yields iterator of `Block` dataclasses |
| `epub_parsing.extract_text_blocks_from_epub` | `epub_parse` | |
| `text_cleaning.clean_paragraph` / `clean_text` | `text_clean` | |
| `heading_detection._detect_heading_fallback` | `heading_detect` | |
| `list_detection.*` helpers | `list_detect` | |
| `splitter.semantic_chunker` | `split_semantic` | |
| `fallbacks._extract_with_pdftotext/pdfminer` | `extraction_fallback` | |
| `ai_enrichment.classify_chunk_utterance` | `ai_enrich` | |

### IO Boundaries (adapters)
- PDF open via `fitz.open` → `adapters.io_pdf.read`
- EPUB open via `epub.read_epub` → `adapters.io_epub.read_epub`
- LLM calls (`litellm.completion`) → `adapters.ai_enrich`
- JSONL write → `adapters.emit_jsonl.write`

### Environment Variables
- `PDF_CHUNKER_USE_PYMUPDF4LLM`: Enable PyMuPDF4LLM enhancement
- `PDF_CHUNKER_DEDUP_DEBUG`: Emit warnings for dropped duplicates
- `OPENAI_API_KEY`: Required for AI enrichment
- `DISABLE_PYMUPDF4LLM_CLEANING`: Rollback to traditional text cleaning

---

## Performance Monitoring

### Key Thresholds

| Zone | Extraction Time | Memory | Success Rate | Quality Score |
|------|-----------------|--------|--------------|---------------|
| **Green** | ≤120% baseline | ≤130% baseline | ≥95% | ≥0.7 |
| **Yellow** | 120-150% | 130-150% | 90-95% | 0.5-0.7 |
| **Red** | >150% | >150% | <90% | <0.5 |

### Rollback Triggers
Consider rollback to traditional extraction when:
- Text quality degradation from PyMuPDF4LLM cleaning
- Performance exceeds red zone thresholds
- Frequent extraction failures
- Version compatibility issues

Rollback options:
1. Set `DISABLE_PYMUPDF4LLM_CLEANING=true`
2. Uninstall PyMuPDF4LLM (system falls back automatically)
3. Modify `clean_text()` default parameter

---

## Known Issues and Limitations

* Tests may not fully cover all critical features or edge cases
* Some code descriptions in `AGENTS.md` may be outdated due to drift
* **Footnote handling improved**: footnote lines detected and appended to the end of their paragraph (e.g., `"Sentence.\nFootnote text."`) to prevent mid-sentence splits; inline footnote sentences with lost markers are relocated when they match common footnote starters; detection tuned to avoid false positives
* **Header/footer cleanup**: headers and footers stripped when they appear at page breaks or within paragraphs, including trailing "|" fragments
* **Hyphenation defect**: carried-over hyphens (e.g. `con‐ tainer`) not rejoined <- fixed.
* **Bullet hyphen fix**: words split at line breaks within bullet lists now rejoin correctly without duplicating the bullet marker.
* **Bullet list splitting fixed**: bullet lists spanning chunk boundaries are rebalanced so items stay within a single chunk.
* **Bullet fragment cleanup**: multi-line bullet items no longer insert spurious paragraph breaks; regression test guards cases like "singing, when I state".
* **Underscore emphasis removed**: PyMuPDF4LLM cleanup strips single and double underscore wrappers.
* **PyMuPDF4LLM list metadata propagated**: `list_kind` from PyMuPDF4LLM blocks is retained in final chunk metadata.
* **PyMuPDF4LLM page loss**: enhancement step now reverts to traditional extraction if pages disappear, guarded by `footer_artifact_test.py`.
* **Cross-page paragraph splits fixed**: lines continuing after a page break are merged to prevent orphaned single-sentence paragraphs.
* **Comma continuation fix**: same-page blocks ending with commas merge with following blocks even if the next starts with an uppercase word.
* **Split-word merging guarded**: only joins across newlines or double spaces when the combined form is more common than its parts, avoiding merges like "no longer"→"nolonger".
* **JSONL deduplication**: repeated sentences are trimmed during emission, even when the duplicate is followed by new material; run conversions with `--trace <phrase>` to verify expected lines survive.
* **Dedup debug**: set `PDF_CHUNKER_DEDUP_DEBUG=1` to emit a warning for each dropped duplicate, a summary count, and a notice for any long sentences that still appear more than once after dedupe.
* Quality and implementation risks are tracked in the active
  [IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md); do not use the
  archived regression sweep as current status.
* Overlap tuning remains evidence-gated by corpus and retrieval evaluation.
* Tag classification may not cover nested or multi-domain contexts.

### Streaming Extraction

* `pdf_chunker.pdf_parsing.extract_text_blocks_from_pdf` now yields an iterator
  of `Block` dataclasses for streaming consumption.
* Use `extract_text_blocks_from_pdf_list` for the deprecated list-of-dicts
  behaviour when eager materialisation is required.

---

## Testing Requirements for OpenAI Codex

Use `pytest`:

```bash
pytest tests/
```

Tests must:

* Be stateless and idempotent
* Focus on input→output behavior
* Use fixtures and shared utilities in ```tests/utils/common.sh`

> **Note**: Some legacy regressions are still being worked through, but any tests that currently pass (e.g., numbered list preservation) are authoritative. Do **not** introduce new failures—rerun the relevant specs after modifying chunking or emission logic.

---

## Pull Request Guidelines for OpenAI Codex

1. Keep PRs focused: one module or concern per PR
2. Update or add tests for any logic changes
3. Do not mix formatting-only and logic changes
4. Provide clear docstrings and inline comments where needed
5. Reference relevant issues or defects in commit messages
6. Use short-lived branches per task/turn; the parent branch should only be merged after the user reviews the turn’s changes.

---

## Programmatic Checks for OpenAI Codex

```bash
pip install nox  # if missing
nox -s lint
nox -s typecheck
nox -s tests
bash scripts/validate_chunks.sh
`````

All checks must pass before merging.

The lint and typecheck sessions use the incremental production target set declared
in `noxfile.py`; a passing session is not a claim that every maintained module is
already clean. Expand that set in small, tested batches. Use `ruff check --no-fix`
for diagnosis and review rather than applying repository-wide automatic fixes.

The `QUALITY_PATHS`, `LINT_EXTRA_PATHS`, and `TEST_TARGETS` groups are required:
empty declarations or blank paths raise `ValueError`, and missing files/directories
raise `FileNotFoundError` listing the affected group and its missing targets in
first-seen order. Sessions validate before dependency installation or tool execution;
they must not quietly shrink their scope or succeed without a test/type command.
After a deliberate target move, update `noxfile.py` and run
`pytest tests/bootstrap/test_quality_gate_scope.py`, then `nox -s lint typecheck tests`.
Valid target spelling, ordered deduplication, commands, and dependency profiles
remain unchanged. These filesystem checks belong to tooling, not pure passes.

All three nox sessions request Python 3.11, matching CI, and fail if it is
unavailable. To provision it without replacing the local venv, install the
optional developer downloader with `python -m pip install --upgrade "nox[pbs]"`,
then run `nox -s lint typecheck tests`. See `CONTRIBUTING.md` for setup details;
an explicit `nox --force-python 3.14 -s lint typecheck tests` run is exploratory,
not supported-Python baseline evidence.

---

## Output Protocol: Voice + Ledger (Mandatory)

We use a single output protocol; there is no Mode A.

Every response must include **both** sections, in this order:

### A) Voice (≤ 5 lines, mandatory)
- Speak **as the codebase** (first-person is allowed and encouraged)
- Content: priorities, tradeoffs, warnings, boundary concerns, health instincts
- Constraints:
  - Max 5 lines
  - No melodrama, no suffering language, no guilt-tripping
  - No "new facts" here unless they are repeated with evidence in the Ledger
  - No proposing broad work solely in Voice; proposals must be justified in Ledger

### B) Ledger (mandatory, audit-grade)
This section is non-persona and must contain explicit governance artifacts:

- **Acceptance criteria** (bullets)
- **Goals accomplished** (checkboxes mapped to criteria)
- **Diagnosis / intent**
- **What changed and why**
- **Evidence**
  - commands run + outcomes
  - tests added/updated
  - before/after behavior
- **Risk / remaining unknowns**
- **Next steps** (1–3)
- **Commit message suggestion**

If Voice and Ledger conflict, **Ledger wins**. Voice may be poetic; Ledger must be true.

---
