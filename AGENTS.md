# AGENTS.md — Project Guidance for OpenAI Codex

This `AGENTS.md` suite provides comprehensive guidance to OpenAI Codex and other AI agents working across this modular Python library for processing large PDF and EPUB documents. The goal is to generate high-quality, semantically coherent document chunks enriched with rich metadata for downstream local LLM workflows, particularly Retrieval-Augmented Generation (RAG).

---

## Project Structure for OpenAI Codex Navigation

The codebase follows a modular structure rooted in Unix philosophy (single responsibility, composability via interface boundaries):
The directory structure may have changed. If any new files or folders have been created, they should be added to the structure below. This will help OpenAI Codex understand the project better and assist in generating code or documentation.
If possible, check to make sure the tree below reflects the current state of the project, and update it if necessary.
```
pdf_chunker/
├── .env                           # API keys and configuration secrets
├── config/
│   └── tags/                      # External tag configuration (YAML vocabularies)
│       ├── generic.yaml           # Base tag categories
│       ├── philosophy.yaml        # Philosophy domain tags
│       ├── psychology.yaml        # Psychology domain tags
│       ├── technical.yaml         # Technical domain tags
│       └── project_management.yaml # PM domain tags
├── pdf_chunker/
│   ├── core.py                    # Orchestrates the three-pass pipeline
│   ├── parsing.py                 # Structural Pass: visual/font-based extraction
│   ├── text_cleaning.py           # Ligature, quote, control-character cleanup
│   ├── heading_detection.py       # Heading detection heuristics and fallbacks
│   ├── extraction_fallbacks.py    # Fallback strategies (pdftotext, pdfminer)
│   ├── page_utils.py              # Page range parsing and validation
│   ├── epub_parsing.py            # EPUB extraction with spine exclusion support
│   ├── splitter.py                # Semantic Pass: chunk splitting and boundaries
│   ├── utils.py                   # Metadata mapping and glue logic
│   └── ai_enrichment.py           # AI Pass: classification and YAML-based tagging
├── scripts/
│   ├── chunk_pdf.py               # CLI for running the full pipeline
│   ├── validate_chunks.sh         # Quality and boundary validation
│   ├── detect_duplicates.py       # Overlap and duplicate detection
│   └── _apply.sh                  # Batch apply scripts across multiple files
└── tests/                         # Modular test architecture
    ├── utils/
    │   └── common.sh              # Shared test utilities and formatting
    ├── pdf_extraction_test.py     # PDF extraction and fallback validation
    ├── ai_enrichment_test.py      # Tagging and classification tests
    ├── semantic_chunking_test.py  # Chunk size and semantic rules tests
    ├── page_exclusion_test.py     # PDF page exclusion tests
    ├── epub_spine_test.py         # EPUB spine exclusion tests
    └── run_all_tests.sh           # Orchestrates all test modules

`````

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
* **Single-responsibility modules**, minimal interdependencies
* **Explicit logging** and **testability**
* Adhere to **Unix philosophy**: communicate via clear interfaces, CLI tools

---

## Known Issues and Limitations

* Tests may not fully cover all critical features or edge cases
* Some code descriptions in `AGENTS.md` may be outdated due to drift
* **Footnote handling improved**: footnote lines detected and removed to prevent mid-sentence splits
* **Hyphenation defect**: carried-over hyphens (e.g. `con‐ tainer`) not rejoined <- this has been fixed. Make sure you update AGENTS.md whenever you make changes. This file must reflect the reality.
* Possible regression where `text_cleaning.py` updated logic not applied
* Overlap detection threshold may need tuning
* Tag classification may not cover nested or multi-domain contexts

---

## Testing Requirements for OpenAI Codex

Use `pytest` and shell scripts:

```bash
pytest tests/
bash tests/run_all_tests.sh
`````

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
flake8 pdf_chunker/
mypy pdf_chunker/
bash scripts/validate_chunks.sh
`````

All checks must pass before merging.

---

