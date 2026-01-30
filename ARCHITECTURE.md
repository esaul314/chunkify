# ARCHITECTURE.md — Structure, Boundaries, and Mental Models

This repository is designed to be **easy to grasp**. Structure exists to reduce cognitive load.

**pdf-chunker** is a modular Python library for processing large PDF and EPUB documents, generating semantically coherent chunks enriched with metadata for downstream LLM workflows (RAG, LoRA fine-tuning).

---

## Related Documentation

| Document | Purpose |
|----------|--------|
| [AGENTS.md](AGENTS.md) | Primary guidance for AI agents—domain map, constraints, workflow |
| [CODESTYLE.md](CODESTYLE.md) | Code style, patterns, and formatting standards |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution workflow, quality gates, PR expectations |
| [docs/inline_style_schema.md](docs/inline_style_schema.md) | Inline style metadata schema for downstream processing |

---

## 1) Repository layout

````
pdf_chunker/
├── pdf_chunker/
│   ├── __init__.py
│   ├── core.py              # Pipeline orchestration and invariant enforcement
│   ├── cli.py               # Typer CLI: convert / inspect commands
│   ├── config.py            # Pydantic Settings + YAML pipeline spec loader
│   ├── framework.py         # Artifact/Pass registry + pipeline runner
│   ├── interactive.py       # Unified interactive callback protocol
│   ├── learned_patterns.py  # Persistent learned patterns (~/.config/pdf_chunker/)
│   ├── patterns.py          # Pattern registry and confidence-based evaluation
│   ├── adapters/            # I/O boundaries (PDF/EPUB read, JSONL write, LLM calls)
│   │   ├── io_pdf.py
│   │   ├── io_epub.py
│   │   ├── emit_jsonl.py
│   │   └── ai_enrich.py
│   ├── passes/              # Pure transformation passes (no I/O)
│   │   ├── pdf_parse.py
│   │   ├── epub_parse.py
│   │   ├── text_clean.py
│   │   ├── heading_detect.py
│   │   ├── list_detect.py
│   │   ├── split_semantic.py
│   │   ├── split_modules/   # Decomposed modules (footers, lists, stitching, etc.)
│   │   ├── ai_enrich.py
│   │   └── emit_jsonl.py
│   ├── # Domain modules (shared pure logic)
│   ├── text_cleaning.py     # Ligature, quote, control-char cleanup
│   ├── heading_detection.py # Heading heuristics and fallbacks
│   ├── list_detection.py    # Bullet and numbered list detection
│   ├── splitter.py          # Chunk splitting and semantic boundaries
│   ├── fallbacks.py         # Quality scoring and extraction fallbacks
│   ├── inline_styles.py     # Inline style span normalization
│   ├── pdf_blocks.py        # Block extraction with inline style metadata
│   ├── page_artifacts.py    # Header/footer detection helpers
│   └── utils.py             # Metadata mapping and glue logic
├── tests/                   # Mirrors package structure; golden snapshots in tests/golden/
├── scripts/                 # Operational scripts (CLI wrappers, diagnostics)
├── config/tags/             # YAML tag vocabularies for AI enrichment
├── docs/                    # Additional documentation (inline_style_schema.md, etc.)
├── test_data/               # Base64-encoded test fixtures (ligature.b64, etc.)
└── pipeline.yaml            # Default pipeline configuration
````

If you only remember one rule:
> Keep side effects at the edges (adapters). Keep decisions in the core (passes).

## 2) The "functional core, imperative shell" rule

- **Core (passes)**:
  - pure functions transforming in-memory `Artifact` values,
  - deterministic behavior,
  - explicit inputs and outputs,
  - **no** filesystem, network, or subprocess calls.

- **Shell (adapters + CLI)**:
  - parse CLI args, read PDF/EPUB files, call external APIs (LLM),
  - convert to/from canonical data structures (`PageBlocks`, `Chunks`),
  - handle logging and error presentation.

This split makes testing cheap and refactoring safe. Passes can be unit-tested with in-memory fixtures; adapters are tested with integration/golden tests.

## 3) Three-phase pipeline architecture

The pipeline implements a robust **three-phase** transformation (multiple configurable passes grouped into logical phases):

1. **Structural Phase** (`pdf_parse`, `epub_parse`, `heading_detect`, `list_detect`)
   - Extracts typographic and layout structure from PDF/EPUB
   - Hybrid approach: font-size and style heuristics + PyMuPDF4LLM cleaning
   - Fallback logic with quality scoring: PyMuPDF → pdftotext → pdfminer.six

2. **Semantic Phase** (`text_clean`, `split_semantic`)
   - Enforces chunk boundaries: 8k character soft limit, 25k hard truncation
   - Avoids splitting within sentences or headings
   - Generates metadata for each chunk: `chunk_id`, `source_file`, `page_range`, `heading`, `tags`, `text`

3. **Enrichment Phase** (`ai_enrich`, `emit_jsonl`)
   - Classifies and annotates chunks using external tag vocabularies
   - Supports multiple domains: generic, philosophy, psychology, technical, PM
   - Outputs enriched chunk records in JSONL

## 4) Canonical artifact contracts

**After parse (PageBlocks):**
```python
{
  "type": "page_blocks",
  "source_path": "...",
  "pages": [
    {"page": 1, "blocks": [{"text": "...", "type": "paragraph", "inline_styles": [...], ...}]}
  ]
}
```

**After split (Chunks):**
```python
{
  "type": "chunks",
  "items": [
    {"id": "...", "text": "...", "meta": {"page": 1, "heading": "...", "list_kind": "...", ...}}
  ]
}
```

Legacy block fields (`type`, `text`, `language`, `source`, `bbox`) are preserved when lifting into artifacts.

## 5) Module boundaries and naming

A module should answer: "What is this *for*?"

Good module names in this project:
- `pdf_parsing.py`, `text_cleaning.py`, `heading_detection.py`, `splitter.py`, `inline_styles.py`

Avoid:
- `helpers.py` (unless genuinely tiny),
- `misc.py`,
- `utils.py` that becomes a landfill.

## 6) Patterns we endorse (when appropriate)

Use patterns to reduce confusion, not to signal sophistication.

- **Strategy**: multiple extraction engines behind one interface (PyMuPDF, pdftotext, pdfminer)
- **Adapter**: isolate third-party APIs and unstable boundaries (LLM calls, file I/O)
- **Factory**: controlled object construction via `pipeline.yaml` configuration
- **Pipeline**: data transformations as registered passes (our core pattern)
- **Protocol**: unified callback interfaces for interactive decisions (`InteractiveDecisionCallback`)

## 6.1) Interactive Callback System

The pipeline supports interactive decision-making for ambiguous cases (footers, list continuations, heading boundaries). This is implemented via a **unified callback protocol**:

```python
class InteractiveDecisionCallback(Protocol):
    def __call__(self, context: DecisionContext) -> Decision: ...
```

**Key components:**
- `DecisionKind`: Enum of decision types (FOOTER, LIST_CONTINUATION, PATTERN_MERGE, HEADING_BOUNDARY)
- `DecisionContext`: Frozen dataclass with kind, text, page, confidence, pattern_name, extra context
- `Decision`: Action (merge/split/skip) + remember mode (once/always/never) + reason

**Confidence-based evaluation:**
- High confidence (≥0.85): Automatic decision, no prompt needed
- Medium confidence (0.3-0.85): Interactive prompt shown
- Low confidence (<0.3): Default behavior applied

**Learned patterns:**
- `--teach` mode persists user decisions to `~/.config/pdf_chunker/learned_patterns.yaml`
- Patterns are matched on subsequent runs for consistent behavior
- Adapter functions maintain backward compatibility with legacy callbacks

## 7) SOLID, with taste

- **S**: keep pass responsibilities narrow and testable
- **O**: extend by adding new passes, not by rewriting stable ones
- **L**: treat subtypes honestly; don't lie via inheritance
- **I**: small interfaces; `Pass` protocol is minimal (`name`, `input_type`, `output_type`, `__call__`)
- **D**: depend on abstractions at boundaries; inject collaborators

Python note: prefer composition over inheritance unless you have a crisp polymorphic contract.

## 8) UNIX philosophy in a Python repo

- small pieces, composable,
- explicit interfaces,
- avoid "god objects,"
- prefer text-friendly I/O (JSONL, YAML),
- one tool per job, but tools can be chained.

CLI design principles:
- predictable exit codes (0=success, 1=warning/partial, >1=error),
- sane defaults,
- `--help` that tells the truth,
- stdout for data, stderr for logs/errors.

## 9) Enforced invariants

- **Order:** `text_clean` must run **before** any splitting step.
- **Separation:** EPUB-specific logic must never execute in a PDF pipeline (and vice versa).
- **Fallbacks:** reasons + quality metrics must be logged & summarized.
- **Pass purity:** Passes must NOT open files, shell out, access network, or perform I/O. Adapters handle all side effects.

## 10) Change policy

Changes must preserve readability:

- Keep diffs small and localized.
- If a file grows beyond ~300–500 lines, consider splitting.
- If a function exceeds ~40–60 lines, consider extracting helpers.
- If you add complexity, add tests and docs proportional to that complexity.

## 11) Testing philosophy

- Most tests should hit the **core** (pure passes with in-memory data).
- Boundary tests (golden snapshots) exist but remain fewer and more surgical.
- Prefer property-like thinking: invariants, edge cases, and failure modes.
- PDF fixtures are Base64-encoded in `test_data/` to avoid binary artifacts.

Target: fast tests, run often. Use `nox -s tests` or `pytest` directly.

---

## 12) Performance limits and defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| Soft text limit | 8k chars | Triggers chunking |
| Hard text limit | 25k chars | Truncation boundary |
| Target chunk size | 400 chars | Default for `--chunk_size` |
| Overlap | 50 chars | Default for `--overlap` |
| pdftotext timeout | 60s | Subprocess fallback |
| LLM max tokens | 100 | Completion limit |
| Min chunk size | max(8, chunk_size//10) | Avoids tiny fragments |

---

## 13) Known issues and design debt

Track these when working on related code:

| Issue | Status | Guidance |
|-------|--------|----------|
| Footnote handling | Improved | Footnotes appended to paragraph end; inline footnotes relocated |
| Header/footer cleanup | Active | Headers/footers stripped at page breaks, including trailing "\|" |
| Hyphenation | Fixed | Carried-over hyphens rejoined |
| Bullet hyphen fix | Fixed | Words split at line breaks within bullet lists rejoin correctly |
| Underscore emphasis | Fixed | PyMuPDF4LLM cleanup strips single/double underscore wrappers |
| Cross-page paragraph splits | Fixed | Lines continuing after page break merged |
| JSONL deduplication | Active | Repeated sentences trimmed during emission |

See [AGENTS.md](AGENTS.md) for the full list of known issues and debugging directions.
