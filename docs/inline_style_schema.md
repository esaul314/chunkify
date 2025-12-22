# Inline Style Metadata Schema & Feature Guide

## Why this matters
Inline style metadata is now part of the end-to-end PDF pipeline. Every `pdf_chunker.pdf_blocks.Block` can carry a normalized
`inline_styles` payload, and downstream passes (splitting, heading detection, telemetry, tracing) actively consume it. The schema
is therefore a first-class contract that future work must keep in sync. This document is written for AI agents who will extend,
troubleshoot, or redesign the feature.

### Current footprint
* **Extraction:** `pdf_chunker.pdf_blocks._extract_block_inline_styles` harvests PyMuPDF spans, remaps them to the cleaned block
  text, and normalizes them with helpers from `pdf_chunker.inline_styles`.
* **Cleaning alignment:** `pdf_chunker.passes.text_clean` remaps spans after text transforms, and exports roll-up metrics via
  `_inline_style_metrics`.
* **Consumers:**
  - `split_semantic` uses inline styles to isolate footnote anchors, detect captions, and preserve span metadata while merging
    blocks.
  - `heading_detect` boosts confidence when bold/caps inline spans dominate a block.
  - `emit_jsonl` strips superscript glyphs while retaining metadata traced from inline styles.
* **Trace + telemetry:** `pdf_chunker.adapters.emit_trace` emits `<step>_inline_styles.json` artifacts for every traced step, and
  the cleaning pass counts styled blocks per run.

## Data model
```python
@dataclass(frozen=True)
class InlineStyleSpan:
    start: int           # Unicode code point offset into Block.text
    end: int             # Exclusive offset
    style: str           # Normalized style token
    confidence: float | None = None
    attrs: Mapping[str, str] | None = None
```

Spans live in `Block.inline_styles: list[InlineStyleSpan] | None`. When the list is empty we coerce the attribute to `None` so
consumers can keep treating absence as "no style information".

### Style tags
The table below documents the tags we currently emit (✅), those that consumers expect but extraction does not yet provide (⚠️),
and placeholders for future expansion (🛠️).

| Tag | Status | Emitted by | Consumers | Notes |
| --- | --- | --- | --- | --- |
| `bold` | ✅ | PyMuPDF flags & font weight checks | `split_semantic`, `heading_detect` | Triggered when flag 16 is set or the font name suggests heavy weight. |
| `italic` | ✅ | PyMuPDF flags & font style checks | `split_semantic`, `heading_detect` | Flag 2 or `italic/oblique/slanted` in font name. |
| `underline` | ✅ | PyMuPDF flag 8 | (future) link emphasis | Captured so that later passes can recognize emphasized references. |
| `monospace` | ✅ | Font-family heuristics | (planned) code-fragment heuristics | Set when font name contains `mono`, `code`, `courier`, or `console`. |
| `superscript` | ✅ | Baseline delta vs. line median | `split_semantic`, `emit_jsonl` | Primary signal for footnote anchors. |
| `subscript` | ✅ | Baseline delta vs. line median | (future) math processing | Useful for chemical formulas. |
| `link` | ✅ | Span metadata or annotations | `emit_trace` (for debugging) | Includes `attrs['href']` and optional `title`. |
| `caps` | ⚠️ | Not yet emitted | `heading_detect`, `split_semantic` | Low-hanging fruit: detect all-caps tokens within mixed-case lines. |
| `small_caps` | ⚠️ | Not yet emitted | `heading_detect` | Should be derived from font metadata or uppercase ratio heuristics. |
| `drop_cap` | 🛠️ | Pending design | (future) decoration filters | Requires bbox-aware size comparison. |
| `em_dash_break` | 🛠️ | Pending design | (future) soft split hints | Detect span boundaries coinciding with em dash separators. |

Tags are extensible. Downstream code must ignore unknown tags.

### Span invariants
1. `0 <= start < end <= len(Block.text)` (Python `len`, Unicode-aware).
2. Spans are sorted and non-overlapping. Combined emphasis uses duplicate spans sharing bounds, not nesting.
3. Adjacent spans with identical `(style, confidence, attrs)` are merged during normalization.
4. Deterministic extractors emit `confidence=1.0`. Statistical emitters may use `[0.0, 1.0]`.
5. `attrs` holds auxiliary metadata (`{"href": "..."}` or future `{"note_id": "1"}`); keys use `snake_case`.
6. Consumers must gracefully handle `inline_styles is None`.

## Implementation reference
### Normalization helpers (`pdf_chunker.inline_styles`)
* `normalize_spans` composes coercion, optional bounds remapping, clamping, sorting, and adjacent-merge logic into a pure
  pipeline.
* `build_index_map` + `build_index_remapper` translate offsets from raw extractor text to cleaned block text. The remapper is a
  pure `Callable` so downstream code can remain functional and testable.

Unit tests live in `tests/test_inline_styles.py` and cover coercion, normalization, cleaning remaps, consumer behavior, and the
footnote anchor plumbing.

### Extraction (`pdf_chunker.pdf_blocks`)
1. `_iter_text_spans` walks PyMuPDF’s `page.get_text("dict")` payload to recover spans and their baselines.
2. `_collect_styles` converts PyMuPDF flags, font metadata, and link annotations into `(style, attrs)` tuples.
3. `_extract_block_inline_styles` expands those tuples into spans, remaps them to cleaned text, and returns a list or `None`.
4. The resulting list is stored on each `Block` returned by `_structured_block`.

The extraction phase currently emits the ✅ tags from the status table above. Adding new tags should happen here (or in the
future PyMuPDF4LLM bridge).

### Cleaning alignment (`pdf_chunker.passes.text_clean`)
* `_inline_style_metrics` calculates `inline_style_block_ratio` and `inline_style_tag_counts` for run-level telemetry.
* `_remap_inline_styles` rebuilds spans after ligature fixes, hyphen merges, or whitespace edits by delegating to the
  normalization helpers.

### Downstream consumers
* **`split_semantic`**
  - Uses `_inline_style_ratio` to recognize caption candidates when ≥60% of text is bold/italic.
  - Extracts superscript spans into `_footnote_spans` and copies any `attrs` into `footnote_anchors` metadata.
  - Carries spans forward when blocks are merged or split.
* **`heading_detect`**
  - Computes inline style ratios to boost heading confidence when bold/caps spans dominate the block.
* **`emit_jsonl`**
  - Drops literal superscript glyphs before serialization but preserves associated metadata so anchors stay intact.
* **`emit_trace`**
  - Writes `<step>_inline_styles.json` alongside every trace snapshot, summarizing styled ranges and attributes.

## Working effectively with inline styles
### Inspecting spans locally
```bash
pdf_chunker convert platform-eng-excerpt.pdf --spec pipeline.yaml --out /tmp/out.jsonl --no-enrich --trace "inline footnote"
```
This produces `artifacts/trace/<run_id>/<step>_inline_styles.json`. Each entry lists block indices, snippets, and style metadata.

### Adding new tags or heuristics
1. Extend `_collect_styles` (or a dedicated helper) to emit the new tag.
2. Update normalization/cleaning tests with representative spans.
3. Amend consumer logic if the new tag should influence chunking or headings.
4. Document the change in this file and regenerate any relevant trace fixtures.

### Debugging & troubleshooting checklist
* **Extraction sanity:** Use `tests/test_inline_styles.py::test_inline_style_extraction` as a template; feed a minimal PDF block
  through `_extract_block_inline_styles` and assert span locations.
* **Cleaning drift:** When spans misalign after hyphen or ligature fixes, inspect `_remap_inline_styles` outputs in trace files
  and verify `build_index_map` produces a sensible mapping.
* **Chunk anomalies:** If captions or footnotes regress, set `PDF_CHUNKER_DEBUG_INLINE=1` (env flag read by tests) to keep
  `_footnote_spans` in emitted chunks for inspection, or instrument `split_semantic._collect_superscripts` locally.
* **Telemetry gaps:** If `inline_style_block_ratio` unexpectedly drops, confirm the extractor is still populating spans by running
  `pytest tests/test_inline_styles.py` and inspecting recent trace artifacts.

## High-value roadmap (low-effort, high-impact)
The following items are prioritized because they unblock downstream heuristics with modest effort:

1. **Emit `caps` spans during extraction.**
   - Detect all-uppercase tokens within mixed-case lines and add a `caps` span covering the word (or contiguous range).
   - Update heading detection tests to assert the new signal is honored.

2. **Implement `small_caps` detection.**
   - When PyMuPDF exposes `smallcaps` font metadata or when a span’s glyphs are uppercase but rendered at reduced size, tag the
     range as `small_caps`.
   - Backfill fixtures in `tests/test_inline_styles.py` to assert remapping and consumption.

3. **Attach `note_id` attributes to superscript anchors.**
   - During extraction or cleaning, parse superscript spans that contain purely numeric text and set `attrs={"note_id": text}`.
   - Extend `split_semantic` tests to confirm metadata survives into `footnote_anchors`.

4. **Expose inline style ratios in trace metadata.**
   - Augment the trace payload with per-block style coverage (e.g., `% bold`, `% italic`) to simplify debugging of borderline
     headings or captions.

5. **Harden link attribute capture.**
   - Normalize additional annotation shapes (e.g., nested dictionaries) and include `attrs['title']` or `['dest']` where present.
   - Add regression tests exercising the coercion path in `inline_styles._coerce_span`.

Track these items in the engineering board and update this roadmap as work completes.

## Extending beyond PDFs
When EPUB or alternative extractors emit inline style hints, conform them to the same schema:
* Coerce upstream metadata to `InlineStyleSpan` using `normalize_spans`.
* Guarantee offsets refer to the cleaned text that downstream passes see.
* If a format cannot supply spans, set `inline_styles=None` and log an explicit capability gap so telemetry reflects the absence.

---
Maintain this document as the single source of truth for inline style behavior. When you ship enhancements, update the table and
roadmap so the next agent knows the current state of the world.
