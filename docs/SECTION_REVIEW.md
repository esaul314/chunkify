# Document-bound section review

Normal pipeline defaults remain unchanged. Insert `section_detect` after
`heading_detect` and before `split_semantic`, with explicit options:

```yaml
heading_detect:
  enabled: true
section_detect:
  enabled: true
  questions_path: /absolute/new-pending.json
  # answers_path: /absolute/reviewed-answers.json
  # interactive: true
```

The canonical input boundary hashes the source and reads original outline/layout.
The pure detector combines navigation, exact headings, typography, entry patterns
and continuity. Roles are body, contents, acknowledgments, permissions, references,
index or unknown, separately from AI utterance_type/tags. Metadata retains
section_role/id/signals/review_status/decision_origin and source/detector identity.
Contiguous sections split independently, including exact internal heading spans,
so merging/overlap cannot cross a section boundary.

Pending questions include original and adjacent page text, signals, source/settings/
detector identities and exact block/span identities. The agent reviews original
pages first; unsupported cases stay pending for the human. Answers are a JSON
array copied from questions, plus origin (agent/user), nonempty reason and role
or boolean boundary. Terminal prompts use the same validator. Extra, duplicate
or stale answers fail. No answer deletes text, changes exclusions or learns a
global pattern. Boundary=false refuses when roles conflict.

Callbacks extend DecisionKind with SECTION_ROLE/SECTION_BOUNDARY. Return
Decision(action="classify", section_role=..., section_boundary=..., origin="agent",
reason=...) or action="skip" to leave pending; remember remains once. Terminal
mode is interactive=true. Machine mode uses questions_path/answers_path; outputs
are exclusive private0600.

Explicit recover_structural_text=true substitutes hash-bound original TOC
page text when standard extraction erased entries. Extracted blocks and complete
original layout remain in sidecars; ordinary extraction, OCR, global patterns and
source files remain unchanged. New section mode guards ordered source spans
against short-tail truncation hidden by overlap word counts.

CPU-only preparation from the project root:

~~~bash
PYTHONPATH=. python scripts/prepare_section_pilots.py \
  --manifest local-section-reviews/sources.json \
  --output local-section-reviews/example-run
PYTHONPATH=. python scripts/verify_section_pilots.py \
  --directory local-section-reviews/example-run \
  --report local-section-reviews/example-verification.json
~~~

The external manifest maps arbitrary document keys to source, source_sha256 and
an artifacts list containing path/sha256 pairs. An empty artifacts list is valid
for a new source. Do not commit real source paths, document titles, source hashes
or generated outputs. The ignored local-section-reviews directory is intended
for these private inputs and reports.

Reports distinguish cleaned-extraction coverage, original contents recovery,
block/source spans and exact serialized-part reconstruction. Enrichment remains
pending: CPU acceptance does not establish retrieval quality or authorize external
publication. Review original pages and compare purpose-specific queries separately.
Retain original documents and failed reports for diagnosis and rollback.

When original-page recovery cannot account for every extracted token or a block
spans physical pages, the pass retains the extracted text and creates pending
role and boundary questions. Review questions include both original-page and
extracted text. The detector version and source page extents bind replayed
answers; previous detector answers must be reviewed again after this change.
