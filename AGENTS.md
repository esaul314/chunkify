# AGENTS.md — Project Guidance for OpenAI Codex

This `AGENTS.md` suite provides comprehensive guidance to OpenAI Codex and other AI agents working across this modular Python library for processing large PDF and EPUB documents. The goal is to generate high-quality, semantically coherent document chunks enriched with rich metadata for downstream local LLM workflows, particularly Retrieval-Augmented Generation (RAG).

---

## Project Structure for OpenAI Codex Navigation

The codebase follows a modular structure rooted in Unix philosophy (single responsibility, composability via interface boundaries):
The directory structure may have changed. If any new files or folders have been created, they should be added to the structure below. This will help OpenAI Codex understand the project better and assist in generating code or documentation.
If possible, check to make sure the tree below reflects the current state of the project, and update it if necessary.
Document newly introduced modules (e.g., list detection utilities) here to keep this reference current.
```
pdf_chunker/
├── _apply.sh                    # Batch apply scripts across multiple files
├── _e2e_check.sh                # End-to-end pipeline check
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
│   ├── ai_enrichment.py           # AI Pass: classification and YAML-based tagging
│   ├── core.py                    # Orchestrates the three-pass pipeline
│   ├── env_utils.py               # Environment flag helpers
│   ├── epub_parsing.py            # EPUB extraction with spine exclusion support
│   ├── extraction_fallbacks.py    # Fallback strategies (pdftotext, pdfminer)
│   ├── heading_detection.py       # Heading detection heuristics and fallbacks
│   ├── list_detection.py          # Bullet and numbered list detection helpers
│   ├── page_artifacts.py          # Header/footer detection helpers
│   ├── page_utils.py              # Page range parsing and validation
│   ├── parsing.py                 # Structural Pass: visual/font-based extraction
│   ├── pdf_parsing.py             # Core PDF extraction logic
│   ├── pymupdf4llm_integration.py # Optional PyMuPDF4LLM enhancement
│   ├── source_matchers.py         # Source citation heuristics
│   ├── splitter.py                # Semantic Pass: chunk splitting and boundaries
│   ├── text_cleaning.py           # Ligature, quote, control-character cleanup
│   ├── text_processing.py         # Shared text manipulation utilities
│   ├── passes/                    # Registered pipeline passes
│   │   ├── __init__.py
│   │   ├── extraction_fallback.py
│   │   ├── heading_detect.py
│   │   ├── list_detect.py
│   │   ├── pdf_parse.py
│   │   ├── split_semantic.py
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
    ├── cross_page_sentence_test.py
    ├── env_utils_test.py
    ├── epub_spine_test.py
    ├── heading_boundary_test.py
    ├── heading_detection_test.py
    ├── hyphenation_test.py
    ├── indented_block_test.py
    ├── list_detection_edge_case_test.py
    ├── multiline_bullet_test.py
    ├── multiline_numbered_test.py
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
    ├── utils/
    │   ├── AGENTS.md
    │   └── common.sh              # Shared test utilities and formatting
    └── utils_test.py
└── test_data/
    ├── README.md
    ├── hyphenation.b64
    ├── ligature.b64
    ├── sample_test.pdf
    └── underscore.b64
`````

*PDF fixtures for ligatures, underscores, and hyphenation are stored as Base64 to avoid binary artifacts and decoded during tests.*

---

## Formatting Standards

All Python modules, scripts, and tests must be formatted with **Black**, linted with **flake8**, and type-checked using **mypy** before submission.

---

## Core Processing Architecture

The project implements a robust **Three-Pass Pipeline**:

1. **Structural Pass** (`parsing.py`, `heading_detection.py`, `extraction_fallbacks.py`)

   * Extracts typographic and layout structure from PDF/EPUB
   * Hybrid approach: font-size and style heuristics + PyMuPDF4LLM cleaning
   * Fallback logic with quality scoring: PyMuPDF → pdftotext → pdfminer.six
   * Logs fallback reasons and page-level quality metrics

2. **Semantic Pass** (`splitter.py`)

   * Enforces chunk boundaries: 8k character soft limit, 25k hard truncation
   * Avoids splitting within sentences or headings
   * Generates metadata for each chunk:

     * ```chunk_id`, `source_file`, `page_range`, `heading`, `tags`, `text`

3. **AI Enrichment Pass** (`ai_enrichment.py`, YAML configs)

   * Classifies and annotates chunks using external tag vocabularies
   * Supports multiple domains: generic, philosophy, psychology, technical, PM
   * Outputs enriched chunk records in JSONL

---

## Metadata Output Specification

Each JSONL chunk record contains the following fields; fields marked required (\*) are always present, optional fields may be null or omitted:

* `chunk_id` (string)\*: globally unique chunk identifier
* `source_file` (string)\*: original filename
* `page_range` (string)\*: pages encompassed, e.g., "1-3"
* `heading` (string|null): nearest inferred heading; null if none
* `tags` (array)\*: semantic tags from YAML; empty array if none
* `text` (string)\*: cleaned, concatenated chunk text
* `quality_score` (float|null): fallback extraction score (0–1); null if primary extraction succeeded

- Mandatory fields. Optional fields may be omitted or null depending on context.

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

* **Pure functions** and **declarative transforms** preferred
* Use **functional programming** techniques and libraries where applicable
* **Single-responsibility modules**, minimal interdependencies
* **Explicit logging** and **testability**
* Adhere to **Unix philosophy**: communicate via clear interfaces, CLI tools

---

## Known Issues and Limitations

* Tests may not fully cover all critical features or edge cases
* Some code descriptions in `AGENTS.md` may be outdated due to drift
* **Footnote handling improved**: footnote lines detected and removed to prevent mid-sentence splits; detection tuned to avoid false positives
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
* Possible regression where `text_cleaning.py` updated logic not applied
* Overlap detection threshold may need tuning
* Tag classification may not cover nested or multi-domain contexts

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

---

## Pull Request Guidelines for OpenAI Codex

1. Keep PRs focused: one module or concern per PR
2. Update or add tests for any logic changes
3. Do not mix formatting-only and logic changes
4. Provide clear docstrings and inline comments where needed
5. Reference relevant issues or defects in commit messages

---

## Programmatic Checks for OpenAI Codex

```bash
black pdf_chunker/ scripts/ tests/
flake8 pdf_chunker/ scripts/ tests/
mypy pdf_chunker/
bash scripts/validate_chunks.sh
`````

All checks must pass before merging.

---

