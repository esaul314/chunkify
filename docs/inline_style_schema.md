# Inline Style Metadata Schema Proposal

## Context
- `pdf_chunker.pdf_blocks.Block` currently provides block-level metadata (`text`, `type`, `bbox`, etc.) but drops inline emphasis information coming from source extraction spans.
- Downstream passes (`split_semantic`, `heading_detect`, future caption and footnote classifiers) therefore cannot differentiate:
  - Emphasized fragments that signal captions or sidebars.
  - Superscript footnote markers needed for deduplication or metadata linkage.
  - Inline bullets/ordinal markers that survive cleaning but should influence chunk boundaries.
- Goal: extend the shared `Block` payload with a **normalized inline style layer** while keeping existing semantics intact.

## Proposed Data Model
Add an optional `inline_styles: list[InlineStyleSpan] | None` attribute to `Block`.

```python
@dataclass(frozen=True)
class InlineStyleSpan:
    start: int           # Unicode codepoint offset into Block.text
    end: int             # Exclusive offset
    style: StyleTag      # Enumerated token (see below)
    confidence: float | None = None
    attrs: Mapping[str, str] | None = None
```

### StyleTag enumeration
| Tag                | Description / extraction hint                             | Consumer use cases                                    |
| ------------------ | --------------------------------------------------------- | ----------------------------------------------------- |
| `bold`             | PyMuPDF flag 16 or span font weight > baseline             | Caption detection; emphasis heuristics                |
| `italic`           | PyMuPDF flag 2 or font style contains `Italic/Oblique`     | Emphasis heuristics; highlight quotes                 |
| `small_caps`       | Detected by span font metadata or uppercase ratio         | Identify subheadings; annotate acronyms               |
| `superscript`      | Span baseline offset < -2pt relative to line median       | Footnote marker detection                             |
| `subscript`        | Span baseline offset > +2pt relative to line median       | Chemical formulas; math notation                      |
| `underline`        | Span flag 8                                               | Link detection; special formatting cues               |
| `link`             | Extracted URI or link annotation                          | Preserve anchor metadata; emit clickable references   |
| `monospace`        | Font family flagged as mono                               | Code snippet recognition                              |
| `caps`             | All uppercase tokens within mixed-case context            | Abbreviation spotting; emphasis scoring               |
| `em_dash_break`    | Inline em dash from span boundary (for list heuristics)   | Soft-split heuristics; chunk boundary hints           |
| `drop_cap`         | Single leading glyph significantly taller than median     | Decorative detection; avoid misclassifying headings   |

*Tags can be extended; unknown tags must be ignored by consumers.*

### Span invariants
1. `0 <= start < end <= len(Block.text)` using Python `len` (Unicode codepoints).
2. Spans are sorted ascending by `start` and **non-overlapping**. Nested emphasis is represented by multiple spans with the same boundaries (e.g., bold+italic) rather than overlaps.
3. Adjacent spans with identical `(style, confidence, attrs)` must be merged upstream before emission.
4. `confidence` defaults to `1.0` when derived from deterministic engine flags; probabilistic classifiers may emit values in `[0.0, 1.0]`.
5. `attrs` encodes auxiliary data such as `{"href": "..."}` for links or `{"note_id": "1"}` for superscripts mapped to footnotes. Keys must be lower snake case.
6. When `inline_styles` is absent or empty, downstream logic must behave exactly as today.

## Extraction Responsibilities
- **pdf_parse (PyMuPDF native)**
  - Inspect `page.get_text("dict")` spans to derive baseline, font flags, and textual fragments.
  - Convert per-span formatting to normalized `InlineStyleSpan` objects relative to the cleaned block text.
  - Record hyperlink URIs from annotations into `link` spans with `attrs` containing `href` and optional `title`.
- **pymupdf4llm integration**
  - Translate existing PyMuPDF4LLM metadata (e.g., `list_kind`, `emphasis`) to the same schema.
  - Guarantee parity with native extractor; when PyMuPDF4LLM lacks data, emit empty spans but preserve attribute for compatibility.
- **text_clean pass**
  - When altering text (e.g., ligature repair, hyphen joins), apply identical transforms to span offsets using pure functions to avoid drift.
  - Drop spans that collapse to zero length after cleaning.
- **fallback extractors**
  - When upstream data lacks spans, set `inline_styles=None` and log capability gaps for monitoring.

## Consumer Requirements
1. **Split heuristics**: `split_semantic` may treat blocks with `superscript` near the end as footnotes and detach them into metadata instead of main text.
2. **Heading detection**: combine `bold`, `caps`, and `small_caps` spans to boost heading confidence without altering baseline heuristics.
3. **Caption scoring**: list detection and future caption classifiers should look for blocks where >60% of characters fall under `italic` or `bold` spans.
4. **Deduplication**: `emit_jsonl` should strip `superscript` spans while keeping `attrs.note_id` for cross-referencing footnote bodies.
5. **AI enrichment**: expose inline style summary (e.g., first bold phrase) in metadata to inform downstream prompt engineering.
6. **Trace/debug tooling**: extend trace artifacts to serialize inline span info for troubleshooting (JSON-friendly structure).

## Implementation Tasks (sequenced)
1. **Schema introduction**
   - Add `InlineStyleSpan` dataclass + `inline_styles` field to `Block` with default `None` to avoid changing serialized payloads immediately.
   - Provide helper functions for span normalization (merge, clamp) under a new `pdf_chunker.inline_styles` module.
2. **Extractor instrumentation**
   - Update PyMuPDF block assembly to emit spans based on `dict` output.
   - Ensure cleaning transforms re-map offsets via pure functional utilities.
3. **Consumer opt-in**
   - Modify passes to accept the new metadata while retaining fallback behaviors when spans are missing.
4. **Telemetry + validation**
   - Emit metrics on percentage of blocks with style metadata.
   - Add regression tests asserting span alignment after text cleaning.

## Open Questions & Follow-ups
- **Serialization format**: do we expose spans directly in JSONL output now or keep them internal? Recommendation: defer until consumers request; include feature flag when ready.
- **Nested emphasis**: current invariant forbids overlapping spans. If downstream needs nesting (e.g., bold+italic on same token), confirm whether duplicate spans are sufficient or if we should allow `styles: set[str]` per span.
- **Non-textual cues**: should inline images or equations embed pseudo-style spans (e.g., `{"style": "inline_image"}`)? Pending design input from math extraction workstream.
- **Performance**: additional span remapping during text cleaning may add overhead. Benchmark once implemented to ensure <5% impact on large PDFs.
- **Accessibility metadata**: linking `attrs` to PDF structure tree (e.g., actual alt text) may require future schema extension; note dependency on upcoming accessibility extraction tasks.

## Acceptance Checklist
- Schema documented here is approved by platform owners.
- Downstream teams sign off on invariants (non-overlap, offset semantics).
- Follow-up tickets filed for unresolved questions.
- No change in current JSONL payload until a separate rollout plan is in place.
