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
- `geometry.py`: Position-based zone selection and scoring, plus existing PDF
  extraction and interactive-discovery compatibility boundaries.
- `epub_parsing.py`: Spine discovery/exclusion and ordered uncovered body-markup
  recovery; preserves semantic list/navigation behavior without duplicate wrappers.
  CPU regression procedure: `pytest -s tests/epub_body_markup_test.py`.
- `splitter.py`: Enforces chunk boundaries and semantic cohesion.
- `passes/split_semantic.py`: Canonical semantic owner for stitching, boundaries,
  overlap, deduplication, footnote cleanup, and chunk metadata.
- `passes/chunk_options.py`: Immutable semantic sizing options and copied metric updates.
- `passes/chunk_pipeline.py`: Composable block/record iterators with injected folds
  and builders; metadata builders receive `index` by keyword.
- `passes/sentence_fusion.py`: Pure sentence-run transitions and merge-budget decisions.
- `passes/metrics.py`: Data-first, copy-on-write per-pass metric updates.
- `passes/text_clean.py`: Ordered text/style cleanup and copied normalization
  metrics, with explicit injected footer callbacks and compatibility wrappers.
- `passes/heading_detect.py`, `passes/list_detect.py`, `passes/merge_footers.py`,
  `passes/detect_doc_end.py`, and `passes/detect_page_artifacts.py`: Structure
  transforms that preserve input payloads and nested metadata while reporting
  their own metrics.
- `passes/emit_jsonl.py`: Normalizes canonical chunks for JSONL and validates
  serialized size; unmarked direct callers use the legacy compatibility path.
- `ai_enrichment.py`: Legacy shim delegating to pass/adapter.
- `adapters/request_budget.py`: Optional cached-tokenizer/template admission
  for the explicit RAG Qwen8B serving contract; no downloads or model loading.
  The enrichment adapter counts the complete wire message plus output/margin
  before any primary or classification-repair transport. Pass prompt grounding
  preserves full text/IDs and full vocabulary for response validation.
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
- `evaluation.py`: Deterministic corpus-quality metrics for preservation, structure,
  metadata completeness, overlap, duplicates, and artifact-removal reporting.
- Respect strict separation of passes.
- Log fallback reasons and quality metrics.
- Use `text_cleaning.py` transforms before splitting.
- Do not mix EPUB logic into PDF modules.
- Keep transcript parsing light: adapter-side text normalization, pass-side pure chunking, and metadata passthrough into emitted rows.
- Transcript timing is provenance, not semantic text: `io_text` emits
  `timestamp_start`/`timestamp_end` source fields, and the shared splitter
  carries first/last available cue bounds into chunk metadata.
- `.txt` pipelines must not include PDF artifact passes; the canonical runner
  rejects `merge_footers` and `detect_page_artifacts` for text input.

## Current boundary notes
- Active implementation status and remaining risks are tracked in
  `docs/IMPLEMENTATION_ROADMAP.md`.
- Geometry is included in the incremental quality gate. Its in-memory helpers
  are data-first: `partial(_get_bottom_blocks, page_height=800,
  margin_percent=0.1)` selects positioned blocks without modifying them.
  Margins are strict, rounding uses Python's nearest-integer bins, frequency ties
  retain first-seen order, and confidence counts observations rather than unique
  pages. `partial(_analyze_all_zone_candidates, texts_by_position=samples,
  min_count=2)` builds a reusable scoring stage; returned sample lists are borrowed.
  `ZoneCandidate.format_for_prompt()` is pure and bounds previews to three samples
  and 70 source characters before newline flattening. Document extraction retains
  partial results on errors, and interactive discovery prints/prompts: do not
  describe the whole module as pure or change these compatibility effects here.
- Optional PDF enhancement and fallback dependencies are documented in
  `docs/DEPENDENCIES.md`; base conversion must not require them.
- Split-word merging is restricted to newline or double-space boundaries with
  frequency checks to avoid collapsing valid phrases.
- Semantic iterator stages take their input stream first and collaborators as
  keyword-only settings, so `partial(chunk_records, generate_metadata=True,
  build_plain=plain_builder, build_with_meta=metadata_builder)` is a reusable
  stage. A metadata builder may itself be partially applied but must accept
  `(text, block, page, *, index)`; inputs remain unmodified.
- `chunk_options`, `chunk_pipeline`, and `sentence_fusion` are included in the
  incremental Python 3.11 quality gate. Run `nox -s lint typecheck tests` when
  changing their types, transitions, or collaborator interfaces.
- Structure passes and `passes/metrics.py` are also quality-gated. Use
  `partial(with_metrics, pass_name="list_detect", updates=counts)` when a reusable
  metadata stage is useful; pass metadata as the remaining input. The updater
  preserves unrelated entries and copies the changed containers, then `Artifact`
  detaches nested metadata. Existing metric containers must be mappings;
  malformed containers raise an explicit `TypeError` rather than being discarded.
- Page-artifact cleanup maps each page once and aggregates counts and decisions
  in page/block order. Use `partial(_clean_doc, footer_config=config)` for a
  reusable data-first cleanup stage in focused tests. The pass flattens tables
  before matching known footer patterns; matching removes the entire block.
  Runtime known patterns or a callback replace the instance config, while empty
  patterns and never-pattern-only overrides retain it. Callback/cache/interactive
  fields remain accepted for compatibility, but this path does not classify
  ambiguous footers or invoke callbacks. Malformed option containers raise
  `TypeError`; prior decision metrics are retained when a run has no new decisions.
- Text cleanup is also quality-gated. Use `partial(_clean_doc,
  footer_patterns=compiled, inline_patterns=inline)` from `passes.text_clean`
  for a reusable data-first stage. Page/block iterators are consumed once in order;
  surviving blocks determine count and style coverage. Page numbers override
  block-local numbers, and nonempty runtime patterns replace instance patterns.
  String payloads use the legacy text-only path and only set `metrics.normalized`;
  document cleanup uses `with_metrics` and retains conditional metrics without a
  new value. Option/metric containers must be mappings (`TypeError` otherwise).
  Injected callback/cache effects and idempotent import-time compatibility wrappers
  remain unchanged; do not describe this whole module as strictly side-effect-free
  or remove those wrappers without separate compatibility coverage.
```

---
