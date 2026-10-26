# Document-bound section review

Ordinary extraction and chunking defaults stay unchanged. Enable native PDF
structure handling by inserting `section_detect` after heading detection and before
semantic splitting:

~~~yaml
pipeline: [pdf_parse, text_clean, heading_detect, section_detect, split_semantic, emit_jsonl]
options:
  heading_detect:
    enabled: true
  section_detect:
    enabled: true
    questions_path: /absolute/new-pending.json
    # answers_path: /absolute/reviewed-answers.json
    # interactive: true
    # preservation_path: /absolute/new-private-preservation.json
~~~

The input boundary hashes the original PDF and captures physical text blocks
before footer filtering, fallback replacement or paragraph merging. Native
physical units retain page, block number, bounding box, raw text hash and original
character extent. The reader contract binds effective PyMuPDF/MuPDF versions,
text flags and the converter implementation fingerprint. Explicit alternative
reader selection refuses this native proof mode; it never silently substitutes
another reader. Existing ordinary reader and OCR behavior remains available.

The pure detector combines navigation, exact headings, typography, entry
patterns and section continuity. Roles are body, contents, acknowledgments,
permissions, references, index and unknown, independently of AI enrichment
categories. Metadata carries role, section identity, signals, review status and
decision origin. Conflicting evidence remains unknown and eligible for review.
An obsolete references outline cannot confidently override a supported new
introduction. Position or a prose mention of a role word alone establishes
nothing.

Exact internal heading spans split independently. Merging and overlap stay
inside a section. Native blocks already retain original text; recovery does not
replace a heading slice with its complete parent block. Historical extraction
without native identities can still request hash-bound original contents text
with `recover_structural_text: true`. A source coverage conflict preserves
extracted text and raises questions rather than inventing a boundary.

Pending questions carry original and adjacent text, detection signals, source,
reader/settings/detector identities and exact page/block/spans. An agent reviews
original pages first; unsupported decisions remain pending for a human. Answers
are an array copied from questions, plus origin (agent/user), a nonempty reason,
and role or boolean boundary. Terminal and machine paths share validation and
replay. Extra, duplicate or stale answers fail. A decision never deletes text,
changes exclusions or learns a global pattern.

Callbacks extend DecisionKind with SECTION_ROLE/SECTION_BOUNDARY. Return
Decision(action="classify", section_role=..., section_boundary=..., origin="agent",
reason=...) for a supported decision, or action="skip" to leave it pending;
remember remains once. Terminal mode uses interactive=true. Machine mode uses
questions_path/answers_path. New report files use exclusive creation and mode0600.

## Ordered source accounting

Structural PDF mode keeps full original and derived text in a private,
content-addressed ledger. It records every original occurrence, stage order,
declared NFKC/quote/whitespace/letter-only line-hyphen normalization, exact raw
and emitted spans, section order, and bounded authorized overlap. Repeated text
is counted by occurrence. Digit separation, missing tails, duplicate output,
reordered pages/sections, false page ranges and changed raw identities refuse
acceptance. Explicit page or geometric exclusions come only from actual caller
settings.

Rows keep the established block_provenance dictionary shape (page, block index,
bbox and text hash) with a compact occurrence ID. Fixed-size fields bind the raw
manifest, full unit-map hash and per-record witness. Complete identities and raw
texts stay in the private ledger. Serialized siblings retain parent page/range
metadata while their contributor references and emitted spans describe their own
text slices. Generated provenance is partitioned together with exact text before
AI enrichment; record, overlap, chunk and enrichment budgets are unchanged.
Arbitrary caller metadata that cannot fit still refuses.

The coverage scope is **native extracted glyph text**. PDF annotation objects are
captured with exact text/location but remain an explicit unresolved surface;
image-only pages also remain a gap. This is not a claim of complete visual PDF,
OCR, drawing or annotation interpretation. Pending preservation blocks enrichment.

The CLI runner derives a private ledger beside an existing output path
(`out.jsonl.preservation.json`) or beside the run report, unless an explicit
preservation_path is supplied. It reports unresolved preservation warnings. The
library convert() returns pending rows only when an explicit private
preservation_path retains the reasons; otherwise unresolved preservation raises.
Successful library calls without a retained ledger return status=preview.
Only a retained, independently verified ledger establishes replayable proof.

Accounting caps are 8 million raw characters, 50,000 units, 256,000 characters
per unit, 8,192 consumed context tokens and 64 source/context transitions per
logical row. It uses bounded token alignment rather than a whole-document diff.
Unsupported or oversized cases refuse; they do not receive inferred approval.

## CPU-only preparation and independent verification

Run from the converter root:

~~~bash
PYTHONPATH=. python scripts/prepare_section_pilots.py \
  --manifest local-section-reviews/sources.json \
  --output local-section-reviews/example-run
PYTHONPATH=. python scripts/verify_section_pilots.py \
  --directory local-section-reviews/example-run \
  --report local-section-reviews/example-verification.json
~~~

The external source manifest maps arbitrary keys to source, source_sha256 and
an artifacts list of historical path/sha256 pairs. Empty artifacts is valid.
Private inputs and outputs belong outside tracked source; do not commit real
document titles, paths, hashes, extracted passages or generated reports.

A pending first run can be followed by a separately reviewed run using
`--answers-manifest local-section-reviews/review-inputs.json`. This optional
manifest maps source keys to objects with path and sha256 for an externally
reviewed answer array. The helper verifies those hashes, freezes answers in its
new output directory, and invokes the same canonical validator. It supplies no
answers or origin on behalf of an agent or user.

Verification rereads the hash-verified original with the recorded effective
reader, compares raw units/layout, replays canonical detection and supplied
review, and recomputes each normalization, intermediate and serialized span
witness. Updating a producer receipt or acceptance boolean does not bypass this
comparison. Reader/implementation changes require new evidence and review.

CPU acceptance establishes only the declared source-preservation scope and
resolved section review. Enrichment remains pending. Compare purpose-specific
retrieval, citations and application-bound context separately before choosing
new artifacts or publishing. Retain originals and failed reports.
