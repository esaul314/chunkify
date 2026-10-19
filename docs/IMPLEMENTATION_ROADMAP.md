# pdf-chunker Implementation Roadmap

**Status:** P0-A in progress (2026-10-11); Python 3.11 verified, repository-wide coverage remains open
**Prepared:** 2026-09-03
**Purpose:** Durable handoff for the next AI agent or maintainer

The implementation series is behaviorally green on the declared nox sessions and
the representative smoke/evaluation checks, but the repository-wide P0-A baseline
is not complete yet. The nox lint/typecheck sessions now cover the canonical
contract/runner boundary plus the previously maintained pass paths, while direct
checks over `pdf_chunker`, `scripts`, and `tests` still report remaining
Ruff/Black findings and package-wide mypy still reports remaining type errors.
Treat the nox result as an explicit incremental gate, not as evidence that every
maintained module is clean. Continue expanding that scope in small, reviewable
batches without applying a repository-wide formatting mutation blindly.

This document converts the repository audit into an execution plan. It is intentionally
implementation-oriented: each work package explains why it exists, where to work, what
to preserve, and how to prove completion.

## 1. Start here

Before changing code, read these documents in order:

1. [`AGENTS.md`](../AGENTS.md) — stewardship rules, architecture constraints, and output protocol.
2. [`ARCHITECTURE.md`](../ARCHITECTURE.md) — intended pipeline boundaries and contracts.
3. [`CODESTYLE.md`](../CODESTYLE.md) — functional/declarative coding conventions.
4. [`CONTRIBUTING.md`](../CONTRIBUTING.md) — quality gates and commit discipline.
5. This document — current findings, priorities, and acceptance criteria.

Then inspect the worktree:

```bash
git status --short
git log -8 --date=short --pretty=format:'%h %ad %s'
```

The worktree already contains pending user changes for plain-text/transcript ingestion
and metadata propagation. Preserve those changes. Do not reset, clean, or overwrite
unrelated files.

## 2. Mission and non-negotiable constraints

The mission is to make the existing PDF/EPUB/text-to-JSONL pipeline trustworthy,
maintainable, and extensible without replacing its working algorithms wholesale.

Maintain these principles throughout:

- Keep transformation logic pure and move file, subprocess, network, and prompt I/O to adapters or the CLI shell.
- Prefer small named functions, immutable returned values, standard-library solutions, and explicit data flow.
- Do not add a framework or plugin layer unless a concrete second implementation requires it.
- Do not silently change public behavior or delete compatibility shims.
- Do not refresh golden fixtures until a behavior change has been explained and approved.
- Treat text preservation, source traceability, and deterministic ordering as correctness properties.
- Make each change small, reversible, and independently testable.

## 3. Current architecture and target architecture

### Current intended flow

```text
PDF/EPUB/TXT adapter
        ↓
canonical PageBlocks artifact
        ↓
text_clean → structure/list/heading passes → split_semantic
        ↓
chunks → emit_jsonl pass → JSONL writer
        ↓
optional AI enrichment at the integration boundary
```

### Current implementation reality

- [`core_new.py`](../pdf_chunker/core_new.py) is the newer CLI runner.
- [`core.py`](../pdf_chunker/core.py) remains a separate public processing path.
- [`splitter.py`](../pdf_chunker/splitter.py) is still used by `split_semantic` through `_get_split_fn`.
- `split_modules/`, `sentence_fusion.py`, and `emit_jsonl.py` each contain substantial boundary/merge logic.
- `framework.py` provides a global pass registry and an untyped `Artifact` carrier.
- Several passes perform I/O, invoke callbacks, or invoke external clients despite the documented purity rule.
- The current worktree adds `adapters/io_text.py`, `passes/text_parse.py`, and `--metadata-file`; this is the correct direction for transcript ingestion, but it needs full contract and CLI verification.

### Target architecture

Keep the existing pipeline shape, but establish one execution path and four explicit
boundaries:

1. `Reader` adapters produce typed-in-practice `PageBlocks` values.
2. Pure passes transform page blocks into chunks and metadata.
3. `EnrichmentClient` and `DecisionCallback` are injected collaborators, not pass-owned side effects.
4. Writers serialize rows only; semantic merging does not occur during final emission.

## 4. Baseline findings to preserve in the handoff

The audit was performed against the current worktree on 2026-09-03.

### Health baseline

- `nox -s lint` passes only because [`noxfile.py`](../noxfile.py) checks a narrow list of files.
- `nox -s tests` reports `773 passed, 5 failed, 1 skipped, 25 deselected`.
- Direct `mypy pdf_chunker` reports 64 errors across 27 files.
- `nox -s typecheck` also encounters a local Python 3.14/NumPy stub incompatibility; use the supported CI Python version when judging project type health.
- Plain `pytest` collection fails because [`scripts/test_page_boundaries.py`](../scripts/test_page_boundaries.py) is discovered as a test and imports the removed `clean_text_traditional` symbol.
- The five failing tests cover PDF readability drift, EPUB metadata drift, list metadata parity, missing `litellm`, and readability regression.

### Highest-risk code locations

- Fixture-specific chunk repair: [`core.py`](../pdf_chunker/core.py:202).
- Trivial extraction quality scoring: [`fallbacks.py`](../pdf_chunker/fallbacks.py:24).
- Global timeout and repeated document access: [`adapters/io_pdf.py`](../pdf_chunker/adapters/io_pdf.py:249).
- Broad exception fallback around the semantic splitter: [`passes/split_semantic.py`](../pdf_chunker/passes/split_semantic.py:374).
- Duplicate merge/dedupe systems: [`passes/emit_jsonl.py`](../pdf_chunker/passes/emit_jsonl.py:1139).
- CLI module stubbing and duplicated frontend implementations: [`cli.py`](../pdf_chunker/cli.py:217).
- Global registry and shallowly immutable artifacts: [`framework.py`](../pdf_chunker/framework.py:8).
- Unwired learned-pattern loading: [`cli.py`](../pdf_chunker/cli.py:88).

## 5. Prioritized implementation plan

Work packages are ordered. Do not start a later package while an earlier package has
an unresolved correctness decision.

### P0-A — Establish a truthful green baseline

**Why:** The current checks can report partial health while production code remains
untyped, unformatted, or behaviorally failing. This makes every later refactor risky.

**Where:** `noxfile.py`, `pyproject.toml`, test discovery configuration, CI workflow,
dependency declarations.

**Do:**

1. Make pytest collection deterministic. Either exclude operational scripts explicitly or rename/move test-like scripts; do not hide the import failure with a blanket ignore.
2. Expand lint and format checks to all maintained production modules and relevant scripts.
3. Expand mypy targets to the package incrementally, starting with public boundaries and the pass framework.
4. Run the checks under the supported Python version used by CI, not only the local Python 3.14 environment.
5. Make the dependency source of truth explicit. Add enrichment dependencies as an optional extra if enrichment is not required for base conversion; otherwise declare them as runtime dependencies. Reconcile or retire the stale frozen `requirements.txt`.
6. Keep `ruff check --no-fix` as the default review command. Avoid automatic repository-wide mutation during diagnosis.

**Current increments (through 2026-10-11):**

- The declared lint/Black target set now includes the canonical runner boundary
  (`contracts.py`, `framework.py`, and `core_new.py`) plus the clean utility and
  quality-boundary modules (`env_utils.py`, `page_utils.py`, `fallbacks.py`,
  `evaluation.py`, and `language.py`), alongside the existing maintained pass
  paths.
- Mypy uses the same production target set, so the canonical runner and its
  typed artifact/registry boundary and adjacent pure utility modules are checked
  under the declared mypy configuration. Supported-CI-Python verification is
  addressed by the 2026-10-11 interpreter increment below.
- `tests/bootstrap/test_quality_gate_scope.py` locks the command construction to
  those declarations and keeps lint-only helpers (`noxfile.py` and
  `tests/bootstrap`) separate from production type targets.
- The 2026-09-26 utility batch was selected only after each module passed Ruff,
  Black, and mypy independently; geometry and other modules with outstanding
  findings remain outside the gate until their issues are handled deliberately.
- The 2026-09-26 pure-domain batch adds `source_matchers.py`, `patterns.py`, and
  `list_detection.py`. Each module has focused behavioral coverage and passed
  Ruff, Black, and mypy independently before being added; larger modules with
  unresolved findings remain outside the gate.
- The 2026-09-26 parser/observability batch adds `passes/epub_parse.py`,
  `passes/text_parse.py`, `passes/semantic_dedupe.py`, and
  `passes/transform_log.py`. These small pass and audit-log modules already have
  parser, deduplication, and transformation-trace coverage; each passed Ruff,
  Black, and mypy independently before being added.
- The October EPUB work also brought `epub_parsing.py`, `passes/ai_enrich.py`,
  and `passes/split_modules/segments.py` into the declared gate, with EPUB
  adapter and cross-page numbered-list tests included in lint/format checks.
- The 2026-10-11 ingress batch adds `adapters/io_epub.py`, `adapters/io_text.py`,
  `passes/pdf_parse.py`, and `passes/extraction_fallback.py` to Ruff, Black, and
  mypy. The EPUB adapter needed only import and formatting cleanup. New tests
  cover stable grouping of one-shot block iterators, unknown locations/pages,
  extraction-report propagation, and preservation of input values across all
  three parser passes. The parser test module is also lint/format checked.
  Existing text normalization and in-memory fallback tests cover the adjacent
  adapters and selection boundary; conversion behavior is unchanged.
- The 2026-10-11 interpreter increment explicitly selects Python 3.11 for all
  three nox sessions, matching CI, and treats an unavailable interpreter as a
  failure rather than a skipped check. Regression tests guard each session's
  interpreter, the failure policy, and agreement with CI's setup steps. See
  `CONTRIBUTING.md` for local installation or optional standalone provisioning
  through recent `nox[pbs]` tooling; explicit other-interpreter runs are
  exploratory, not a substitute for the supported-Python baseline.
- That supported-runtime check exposed one Python 3.12-only f-string in the
  readability helper in `utils.py`, which prevented package import on Python
  3.11. String concatenation preserves the existing pure transformation and
  exact output; twelve normalization cases lock punctuation, blank lines,
  indentation, and short-heading behavior. A parametrized syntax check covers
  every production module, including ones not imported by other tests, using
  the actual session interpreter. `tests/utils_test.py` is now lint/format
  checked; broader `utils.py` lint/format/type debt remains outside that scope.
- Local lint, typecheck, and the full non-legacy test session have now passed
  under Python 3.11.17 with the declared dependency profiles. Missing-runtime
  failure was also exercised with a deliberately nonexistent interpreter.
  No conversion policy or golden fixture changed; repository-wide scope
  expansion remains the next P0-A task.
- The 2026-10-11 semantic-helper batch adds `passes/chunk_options.py`,
  `passes/chunk_pipeline.py`, and `passes/sentence_fusion.py` to Ruff, Black,
  and mypy. Explicit aliases matching canonical dictionary blocks and a small
  keyword-index callback protocol describe the existing iterator/builder
  interface accurately; a typed initial accumulator makes the existing pure
  coalescing fold type-checkable.
  Two equivalent Boolean simplifications keep the same merge-budget decisions.
  Focused tests cover immutable option/metric updates, provenance precedence,
  one-shot iterator order and laziness, partially applied stages/builders,
  heading attachment, stable chunk IDs, budget boundaries, and sentence-run
  content preservation. Both new test modules are lint/format checked, and
  scope regression tests guard the expanded declarations. All three supported
  nox sessions, 163 focused tests, collection, generated documentation, and
  seven corpus cases pass; the canonical PDF smoke retains its sentinel.
  Comparison with the pre-change implementation gives identical results for
  1,600 budget states and 9,720 fusion cases. No new dependency, Python-version
  policy, semantic rule, or golden fixture is introduced. Broader package
  diagnostics still report 340 Ruff findings and 39 mypy errors in 16 files;
  these remain outside the completed batch, not outside the roadmap.
- The 2026-10-11 structure-pass batch adds `passes/heading_detect.py`,
  `passes/list_detect.py`, `passes/merge_footers.py`, and `passes/detect_doc_end.py`
  to Ruff, Black, and mypy, together with the small pure `passes/metrics.py`
  updater. Failing-first tests exposed nested metric mutation of input artifacts
  in all four passes. A data-first, keyword-configured copy-on-write transform
  preserves unrelated entries and supports partial application; property tests
  check right-biased composition, empty-update identity, and input preservation.
  The passes retain their detection rules, payload shape, ordering, and metric
  values. Malformed metric containers now raise an explicit `TypeError` instead
  of an incidental attribute/item error; no malformed data is silently discarded.
  Targeted import/type/format cleanup removes each pass's existing findings.
  The new lint/format-checked test module also covers no-op payloads, missing
  metadata, footer length/count boundaries and provenance, latest document-end
  marker/search-window behavior, one-shot heading annotation, and page-local
  list continuation. All three nox sessions pass on Python 3.11.17 (33 production
  type targets and 50 lint/format files). The new module has 54 passing behavior
  cases; scope guards, documentation consistency, and metadata integration give
  146 passing focused cases. Collection finds 1,055 tests without errors, generated
  merge documentation is current, all seven corpus cases pass, and the canonical
  PDF smoke retains its sentinel. An in-memory comparison against the pre-change
  passes gives identical output payloads/metadata for 8,037 cases while also
  verifying input-artifact preservation in the new implementation. Diff review
  found no further actionable issues. No dependency, Python-version policy,
  detection rule, or golden fixture changed. Broader package diagnostics now
  report 286 Ruff findings and 35 mypy errors in 12 files; P0-A remains open.
- The 2026-10-11 page-artifact batch adds `passes/detect_page_artifacts.py` to
  Ruff, Black, and mypy. Failing-first tests expose input-artifact metric mutation
  and a second run overwriting the first result's recorded count. Reusing the
  existing copy-on-write metric updater fixes both without a new abstraction.
  An eager ordered map and declarative aggregation replace the document fold's
  hidden decision-list mutation; the data-first helper accepts one-shot page
  iterables and can be partially applied to footer settings. Cleanup rules,
  page/block order, provenance, metric values, and runtime override precedence
  are unchanged. Table flattening precedes whole-block known-pattern removal.
  Callback/cache/interactive fields remain accepted but are not invoked by this
  path; misleading prompting documentation is corrected rather than silently
  introducing new behavior. Malformed option/metric containers now raise explicit
  `TypeError` instead of incidental attribute/item errors, and invalid regexes
  still propagate. The dedicated lint/format-checked regression module covers
  cleanup boundaries, ordered decisions, empty inputs, option precedence,
  callback/cache preservation, retained metrics, and page-concatenation laws.
  The shared purity suite now includes all five structure passes, adding six
  behavior cases alongside 37 dedicated cases and one new scope guard. All three
  nox sessions pass on Python 3.11.17 (34 production type targets and 52 lint/format
  files). The final focused run passes 243 cases, collection finds 1,099 tests
  without errors, generated merge documentation is current, all seven corpus
  cases pass, and the canonical PDF smoke retains its sentinel. An in-memory
  comparison against the prior pass gives identical payloads and metadata for
  19,200 valid cases across table/footer/plain text, page layouts, runtime overrides,
  and existing metrics; current inputs are preserved in every case. Diff review
  found no further actionable issues. Existing evaluation warnings remain, so
  passing corpus criteria are not a claim of perfect text preservation. No new
  dependency, Python-version policy, cleanup rule, or golden fixture changed.
  Broader package diagnostics still report 286 Ruff findings and now 31 mypy
  errors in 11 files; repository-wide P0-A coverage remains the next task.
- The 2026-10-11 text-cleaning batch adds `passes/text_clean.py` to Ruff, Black,
  and mypy. Failing-first checks expose mutation of nested normalization/document
  metrics for string and page-block inputs; copy-on-write updates reuse
  `passes/metrics.py` while preserving the legacy root `metrics.normalized` flag.
  Ordered maps with partially applied data-first block/page stages replace local
  accumulation loops. Counts still mean surviving non-empty blocks, style coverage
  is calculated over those blocks, page numbers retain precedence, and footer
  decisions retain page/block order. Existing text/style transforms, runtime pattern
  precedence, injected callback/cache effects, conditional metric retention, and
  string-only compatibility behavior remain unchanged. Invalid option/metric
  containers now raise explicit `TypeError` rather than incidental attribute/item
  errors; invalid regexes still propagate. Marker writes use the compatibility
  functions' attribute dictionaries, preserving wrapper identity, installation
  idempotence, and original-callable references without suppressing type errors.
  These import-time wrappers remain a compatibility boundary, not a claim of full
  module purity. The lint/format-checked regression module adds 47 behavior cases,
  including one-shot input, partial application, page-concatenation and whitespace
  idempotence laws, style/provenance preservation, callback format/order policy,
  empty inputs, retained metrics, and wrapper installation. One new scope guard
  accompanies the extended production assertion. All three nox sessions pass on
  Python 3.11.17 (35 production type targets and 54 lint/format files), 220 focused
  cases pass, collection finds 1,147 tests without errors, generated merge docs are
  current, all seven corpus cases pass, and the canonical PDF smoke retains its
  sentinel. An in-memory comparison against the previous pass matches exact
  payload/metadata outputs and callback traces for 9,396 cases across string and
  document inputs, styles, page layouts, formats, pattern overrides, and existing
  metrics; current input artifacts are preserved in every case. Diff review found
  no further actionable issues. Existing corpus warnings remain and do not imply
  perfect text preservation. No new dependency, Python policy, cleaning rule,
  public interface removal, or golden fixture change is introduced. Broader package
  diagnostics still report 286 Ruff findings and now 26 mypy errors in 10 files;
  repository-wide P0-A coverage remains the next task.
- The 2026-10-11 geometry batch adds `geometry.py` to Ruff, Black, and mypy.
  Independent checks identified two long prompt lines, an unused preview-loop
  index, and a stale type suppression. A pure candidate comprehension and bounded
  preview projection replace local accumulation without a new abstraction.
  Data-first selection/scoring helpers retain their existing signatures and
  support partial application; Python rounding, first-seen frequency ties,
  observation-based confidence, borrowed sample lists, strict margin boundaries,
  and exact prompt text are unchanged. The lint/format-checked in-memory contract
  module adds 50 behavior cases, including a stable-ranking/cutoff property,
  serialization, sampling after one-based exclusions, confidence thresholds,
  zero-based content-page order, partial extraction on errors, and unavailable
  backend behavior. Failing-first scope checks guard both new declarations.
  Documentation distinguishes pure scoring/presentation from existing document
  extraction and interactive effects; the latter and the extraction error policy
  remain compatibility boundaries, not a claim of whole-module purity.
  All three nox sessions pass on Python 3.11.17 (37 production type targets and
  59 lint/format files, including the preceding request-budget increment).
  The focused run passes 154 cases, collection finds 1,219 tests without errors,
  generated merge documentation is current, all seven corpus cases pass, and the
  canonical PDF smoke retains its sentinel. Exact before/after comparisons match
  7,775 scoring/consistency cases and 6,138 prompt cases with input preservation.
  Diff review found no further actionable issues. Existing test/corpus warnings
  remain; passing corpus criteria do not imply perfect text preservation. No new
  dependency, Python policy, detection rule, public API, or golden fixture changes.
  Broader package diagnostics report 283 Ruff findings and 25 mypy errors in nine
  files; repository-wide P0-A coverage remains open.
- The 2026-10-11 target-resolution increment closes a gap in the gate itself:
  missing declared paths were silently filtered out, and empty type/test targets
  could yield no command while their sessions succeeded. All three declaration
  groups (`QUALITY_PATHS`, `LINT_EXTRA_PATHS`, and `TEST_TARGETS`) are now required,
  non-empty, and free of blank paths. Each session validates its groups before
  dependency installation or tool execution. Missing targets raise
  `FileNotFoundError` listing the affected group and all its missing paths in
  first-seen order; empty groups or blank paths raise `ValueError`. This also
  prevents an empty string from accidentally targeting the repository root.
  The data-first resolver consumes one-shot declarations once and returns an
  ordered, deduplicated tuple; its diagnostic scope is keyword-configured and
  supports partial application. Filesystem validation remains in tooling, not
  document-processing passes. Valid target spelling, commands, dependencies,
  interpreter policy, and gate scope are unchanged (37 production type targets
  and 59 lint/format files); obsolete empty-command success paths are removed.
  Twenty-eight initial failing regression cases reproduce the gap. The final
  suite adds 36 cases covering every group/session, ordered diagnostics, validation
  before session effects, literal nonblank path spelling, duplicate handling,
  input-preservation/idempotence laws, and unchanged valid-session execution.
  All three nox sessions pass on Python 3.11.17, 128 focused cases pass, collection
  finds 1,255 tests without errors, and generated merge documentation is current.
  An isolated negative nox invocation reports all three sessions failed and exits
  nonzero for missing declarations, without installing dependencies or running
  check commands; no real checkout target is removed. Before/after comparisons
  match all three valid command plans and 363 ordered target-resolution cases.
  Diff review found no further actionable issues. Contributor and stewardship
  guidance document deliberate path updates after moves. Existing test warnings
  and broader repository debt remain; P0-A is still open. No document-processing
  code, dependency declaration, public API, Python policy, or golden fixture changed.
- The remaining repository-wide findings are intentionally left for later
  batches; they are not silently excluded from the documented baseline.

**Acceptance criteria:**

- `pytest --collect-only -q` completes without collection errors.
- `nox -s lint`, `nox -s typecheck`, and `nox -s tests` run the intended scope.
- CI and local supported-Python results are reproducible from declared dependencies.
- Any excluded or optional test is explicit, documented, and not hiding a production failure.

### P0-B — Resolve the five current test failures deliberately

**Why:** These failures are the current contract conflicts. Updating goldens before
understanding them would lock in unknown regressions.

**Do:**

1. Compare PDF golden text, metadata, and readability fields separately. Decide whether readability algorithm drift is intended; if not, fix the algorithm. Refresh fixtures only after the decision.
2. Investigate EPUB location semantics (`nav.xhtml` versus chapter content). The output must not treat navigation metadata as chapter body provenance.
3. Trace the first `list_kind` divergence between the legacy and refactored split paths. Preserve the richer correct metadata or document a reviewed exception.
4. Decide whether `litellm` is a mandatory or optional enrichment dependency. Base conversion must remain usable without AI enrichment; enabled enrichment must fail clearly when its dependency/client is unavailable.
5. Keep parity exceptions narrow and documented in `tests/parity/EXCEPTIONS.md`.

**Acceptance criteria:**

- Each failure has either a code fix or a written, reviewed compatibility decision.
- No golden is refreshed solely to make the suite green.
- Base conversion works without enrichment dependencies.
- Enrichment failure is explicit and actionable.

### P0-C — Replace fallback scoring with real extraction quality signals

**Why:** The current scorer gives every non-empty text value a score of `1.0`, so
fallback extraction is not genuinely quality-driven.

**Where:** [`fallbacks.py`](../pdf_chunker/fallbacks.py), PDF adapter, extraction tests,
run-report metrics.

**Do:**

- Measure page coverage and missing pages.
- Measure text density, whitespace/word plausibility, replacement-glyph frequency, and suspiciously repeated lines.
- Preserve per-engine metrics and the fallback reason.
- Compare candidates using the same normalized quality model.
- Do not discard a candidate merely because it has fewer blocks; compare meaningful page/content coverage.
- Keep subprocess and file access in adapters. The quality scorer itself must remain pure.

**Acceptance criteria:**

- A non-empty but obviously degraded primary extraction can trigger fallback.
- A successful primary extraction does not trigger fallback unnecessarily.
- Run reports state the selected engine, score, candidate scores, and reason.
- Tests cover empty, partial, malformed, artifact-heavy, and good extraction cases.

### P0-D — Remove fixture-specific semantic behavior

**Why:** Special cases for literal fixture phrases are not maintainable production
rules and can distort unrelated documents.

**Where:** [`core.py`](../pdf_chunker/pdf_chunker/core.py:202), semantic split tests,
parity tests.

**Do:**

- Remove the `cattle-train`/`lambs.` repairs and chunk reordering.
- Reproduce the intended behavior with general list-continuation, sentence-boundary, or overflow rules.
- Add an input fixture that expresses the structural condition without relying on a document-specific phrase.

**Acceptance criteria:**

- No production code branches on fixture-specific text.
- The original regression behavior remains covered by a structural test.
- Property tests confirm no loss, accidental reorder, or unexplained duplication.

### P1-A — Establish one canonical runner and typed boundary contracts

**Why:** Two orchestration paths and untyped dictionaries make it unclear which
behavior is authoritative.

**Where:** `core_new.py`, `core.py`, `framework.py`, a small new standard-library-only
contracts module if needed.

**Do:**

1. Select `core_new.py` as the canonical runner for CLI conversion.
2. Turn `core.process_document` into a compatibility wrapper or clearly freeze it as a legacy API with tests and deprecation documentation.
3. Introduce small `TypedDict` or dataclass contracts for `Block`, `PageBlocks`, `Chunk`, metadata, and run metrics. Keep external JSON compatibility.
4. Make `Artifact` generic or otherwise typed at the boundary, and ensure metadata copying is deep enough to prevent hidden nested mutation.
5. Make registry registration reject accidental name collisions instead of silently overwriting.
6. Centralize invariant validation in the runner: parser selection, pass order, payload type, and required output shape.

**Acceptance criteria:**

- CLI and library conversion use the same canonical pipeline.
- Compatibility entry points have characterization tests.
- Core public boundaries are type-checkable without `Any` leaks.
- Invalid pass names, payload shapes, and option combinations fail early with actionable errors.

### P1-B — Restore pass purity and explicit side-effect boundaries

**Why:** The documented functional-core/imperative-shell architecture is valuable,
but the implementation currently crosses that boundary.

**Do:**

- Move extraction fallback execution into the PDF adapter; pass only receives extracted candidates and metrics.
- Move LLM calls into an injected `EnrichmentClient` adapter. The enrichment pass should transform a supplied result or use a clearly injected collaborator, with side effects visible at the runner boundary.
- Move prompt construction/execution and learned-decision persistence into callback adapters.
- Make run reports consume supplied environment/dependency metadata rather than reading source files from a supposedly pure assembler.
- Replace broad `except Exception` fallback logic with narrow, logged domain errors. Never silently switch algorithms because of an unrelated programming error.

**Acceptance criteria:**

- Pass tests can run with in-memory artifacts and no filesystem, subprocess, network, or stdin access.
- Side effects are mockable at adapter boundaries.
- Unexpected exceptions fail loudly with pass and input context.
- Interactive and enrichment behavior remains deterministic when collaborators are stubbed.

### P1-C — Consolidate semantic and emission responsibilities

**Why:** Boundary decisions are spread across `splitter.py`, `sentence_fusion.py`,
`split_modules/`, and `emit_jsonl.py`. Multiple merge/dedupe passes increase the risk
of silent data loss.

**Do:**

1. Define one semantic pipeline responsible for stitching, heading/list boundaries, sentence splitting, overlap, and chunk metadata.
2. Keep `emit_jsonl` responsible for row normalization, metadata-key compatibility, serialization preparation, and final size validation.
3. Remove or deprecate duplicate implementations only after parity/property tests establish the canonical behavior.
4. Replace opaque nested `reduce`/`accumulate` state machines with named pure transitions where the state is non-obvious.
5. Maintain a transformation log for decisions that materially change text.

**Acceptance criteria:**

- Each merge or dedupe rule has one owner and one documented precedence position.
- Chunk text is not silently removed by two independent dedupe systems.
- Tests cover sentence starts, overlap, repeated text, lists, headings, footnotes, and page transitions.
- The semantic pass can be tested independently of JSONL writing.

### P1-D — Simplify configuration and CLI wiring

**Why:** The current CLI duplicates implementations and uses dynamic module stubbing;
configuration accepts too many invalid states.

**Do:**

- Keep one internal conversion function and make Typer/argparse thin frontends over it.
- Remove the `sys.modules` adapter fabrication once dependency injection is available.
- Add per-pass option validation for known keys and numeric ranges.
- Make missing explicit spec files an error; reserve empty defaults for an intentional no-spec path.
- Restore environment variables after temporary CLI overrides.
- Make `choose_parser` raise a specific unsupported-format error.
- Wire `LearnedPatterns` into the callback/pattern registry and test that a saved decision changes the next run.

**Acceptance criteria:**

- Both CLI frontends produce identical parsed options and output.
- Invalid options fail before processing input.
- `--teach` has an end-to-end test: prompt decision → persisted pattern → automatic reuse.
- CLI tests do not depend on import-order tricks.

### P2-A — Build a result-quality evaluation layer

**Status: COMPLETE (2026-09-06)**

**Delivered:**
- docs/corpus_manifest.yaml covers tracked PDF, EPUB, transcript, and text-normalization
  fixtures; base64 fixtures are materialized only in temporary evaluation storage.
- scripts/evaluate_corpus.py produces a stable JSON report with separate preservation,
  structure, metadata, fallback, artifact-removal, chunk-size, overlap, and duplicate
  categories.
- Sentinel/order checks and representative-format tests live in
  tests/corpus_evaluation_test.py. Retrieval overlap tuning remains a separate,
  evidence-gated activity.

**Why:** Unit tests and small goldens catch regressions, but they do not measure
quality on representative documents or downstream retrieval behavior.

**Do:**

- Create a corpus manifest with representative PDFs, EPUBs, and transcripts.
- Record extraction engine, page coverage, artifact removals, chunk counts, chunk sizes, overlap, duplicate rate, metadata completeness, and fallback rate.
- Add preservation checks for text order and expected sentinel phrases.
- Add reviewed fixtures for difficult layouts: running headers/footers, footnotes, tables, lists, captions, scanned/poor text, and cross-page sentences.
- Use retrieval evaluation for overlap tuning only after extraction and chunk integrity are stable.

**Acceptance criteria:**

- A pipeline change produces a reviewable quality report, not only a pass/fail test result.
- Intentional differences are visible by category: text, structure, metadata, or metrics.
- Overlap changes are justified by retrieval evidence rather than a generic percentage rule.

### P2-B — Complete transcript/text ingestion as a supported path

**Status: COMPLETE (2026-09-24)**

**Delivered:**
- `.txt` input is selected by the canonical parser map and runs through the
  shared `text_clean → split_semantic → emit_jsonl` pipeline.
- The text adapter normalizes BOM/CRLF, WebVTT cues, blank paragraphs, and
  uppercase/lowercase/bracketed speaker labels; cue timing is provenance
  metadata rather than semantic text.
- Episode metadata is propagated while generated chunk identity and source
  location remain authoritative; PDF artifact passes are rejected for text.
- README, architecture/agent guidance, CLI help, and focused tests document the
  supported contract.

**Why:** The current worktree has the right small adapter-based direction, but the
public documentation and test contract must agree with the implementation.

**Do:**

- Verify `.txt` parser selection, adapter normalization, and reuse of `text_clean → split_semantic → emit_jsonl`.
- Add tests for UTF-8 BOM/CRLF, WebVTT cues, timestamps, speaker labels, blank paragraphs, lowercase or bracketed speaker labels, and timestamp provenance policy.
- Ensure episode-level metadata is copied to every chunk without overwriting generated `chunk_id` or source location fields.
- Update README, AGENTS guidance, and CLI help so supported formats are consistent.
- Keep transcript-specific logic limited to ingress normalization and metadata; do not create a second chunking engine.

**Acceptance criteria:**

- A documented `.txt` command produces valid JSONL with stable document/chunk identity.
- Transcript metadata survives chunking and optional enrichment.
- PDF-only footer and geometric heuristics do not run accidentally on transcripts.

### P2-C — Documentation and dependency cleanup

**Status: COMPLETE (2026-09-24)**

**Delivered:**
- The active plan and architecture documents now point to one current roadmap;
  completed refactoring and stabilization plans are explicitly historical.
- `pyproject.toml` is the dependency source of truth, with supported EPUB
  dependencies in the base runtime and enrichment/PDF integrations as opt-in
  extras; `requirements.txt` is only a compatibility shim.
- README, agent guidance, architecture, pipeline specs, and the generated pass
  table agree on `.pdf`/`.epub`/`.txt`, parser-slot behavior, and registered
  pass names; consistency tests guard the agreement.
- Generated/diagnostic scripts and historical AI notes are labeled separately
  from the maintained conversion path.

**Why:** Several documents describe completed phases or old failures while the code
has moved on. This makes future agents repeat old investigations.

**Do:**

- Make this roadmap the active implementation plan.
- Mark `REFACTORING_ROADMAP.md`, `STRATEGIC_REFACTORING_PLAN.md`, and `test_stabilization_plan.md` as historical or update them to link here.
- Reconcile README, AGENTS, ARCHITECTURE, pipeline YAML, and actual registered passes.
- Replace the oversized environment snapshot with a declared dependency policy and, if needed, a generated lock/constraints workflow.
- Keep generated artifacts and experimental scripts clearly separated from maintained source.

**Acceptance criteria:**

- A new agent can determine current status from one active roadmap and one architecture document.
- Supported formats, commands, pass names, and dependency requirements agree across docs.
- Historical plans are not presented as current work.

## 6. Required correctness invariants

Every implementation package that changes processing behavior should test the relevant
invariants below:

- Input order is preserved unless an explicit structural rule says otherwise.
- No text is lost except for an identified artifact-removal rule.
- No chunk starts in the middle of an unfinished sentence without an explicit hard-limit explanation.
- Intentional overlap is bounded and does not become cumulative duplication.
- Source filename, source kind, page/location, and document-level metadata remain traceable.
- Chunk IDs are stable for the same source/configuration and distinguish separate chunks.
- Metadata generation is deterministic without AI enrichment.
- Optional enrichment cannot prevent base conversion from running.
- Passes do not perform hidden I/O.
- Invalid configuration fails before partial output is emitted.

## 7. Verification protocol

Run these from the repository root in the supported development environment:

```bash
source pdf-env/bin/activate
pytest --collect-only -q
nox -s lint
nox -s typecheck
nox -s tests
python scripts/generate_merge_docs.py --check
git diff --check
```

For semantic changes, also run targeted tests and the canonical smoke conversion:

```bash
pytest tests/semantic_chunking_test.py \
  tests/passes/test_split_semantic_parity.py \
  tests/emit_jsonl_coalesce_test.py -q

pdf_chunker convert platform-eng-excerpt.pdf \
  --spec pipeline.yaml \
  --out /tmp/platform-eng.jsonl \
  --no-enrich
```

Verify the smoke output retains the sentinel phrase:

> The marbled newt is listed as vulnerable by the IUCN due to habitat loss

For any intentional output change, capture a before/after diff of text, metadata,
metrics, and warnings. Do not compare only chunk counts.

## 8. Suggested delivery sequence

Use separate coherent commits, in this order:

1. `test: make repository quality gates truthful`
2. `fix: resolve current golden and parity regressions`
3. `fix: make extraction fallback quality-driven`
4. `fix: remove fixture-specific chunk repairs`
5. `refactor: establish canonical runner and contracts`
6. `refactor: restore pure pass and adapter boundaries`
7. `refactor: consolidate semantic emission ownership`
8. `refactor: simplify CLI and configuration validation`
9. `test: add corpus quality evaluation` — complete
10. `docs: reconcile roadmap and supported input formats` — complete
11. `docs: reconcile active documentation and dependency policy` — complete

If a package is too large for one review, split it by boundary, not by arbitrary
file count. Each commit must leave the repository in a testable state.

## 9. Stop conditions and decision points

Stop and request direction rather than guessing when:

- a golden change would alter user-visible text or metadata without a clear intended behavior;
- legacy compatibility conflicts with a proposed improvement;
- a public API must be removed or renamed;
- a new dependency is required but its operational role is unclear;
- a transcript metadata field would collide with generated metadata semantics;
- a refactor requires deleting large compatibility paths before replacement tests exist.

Do not:

- refresh all goldens as a first step;
- delete `core.py`, `splitter.py`, or compatibility shims wholesale;
- introduce a general plugin framework for a single new adapter;
- tune overlap before measuring preservation and retrieval effects;
- silence broad exceptions to preserve an apparently green test run.

## 10. Definition of done

The roadmap is complete when:

- the supported quality gates are green in a declared environment;
- one documented runner owns conversion behavior;
- pure passes and side-effect adapters have clear boundaries;
- extraction fallback decisions are evidence-based and observable;
- semantic merge/dedupe ownership is consolidated;
- output metadata and IDs satisfy a stable contract across PDF, EPUB, and text;
- representative corpus evaluation exists;
- current documentation describes the actual code rather than historical phase status.


## 2026-10-19 candidate - pre-enrichment JSONL size admission

The isolated branch codex/early-jsonl-size-admission-20261019 starts from
2fe64ae44ef759dd1b3663e8d318933d0b3c06a5. The candidate package version is
0.1.9. In the canonical pipeline, the configured JSONL record limit reaches AI
enrichment. Before classification, chunks are measured as complete serialized
rows with conservative enrichment metadata; oversized text is partitioned by
exact string slices and marked with chunk_part. The payload adapter groups split
results by original item index, retaining interleaved non-chunk items in place.
Final emission preserves marked text boundaries and rejects unexpected
post-enrichment overflow.

Six focused admission/reconstruction tests pass, including wrapped and
unwrapped payloads with interleaved items. The full suite passes with the
original checkout as its fixture working directory, the project virtualenv
activated, and candidate code imported from the isolated worktree. Python 3.11
nox lint and typecheck pass. No production conversion or source retry has been
run; the candidate remains separate from the active checkout.
