_Last updated: 2025-08-18_
# CODEx_START — Refactor Kickoff and Guardrails

This document tells the Codex agent exactly how to proceed. It complements:
- `AGENTS.md` (domain map & constraints)
- `ARCHITECTURE_REFACTOR_TASKS.md` (story cards & acceptance)
- `PROJECT_BOOTSTRAP.md` (files to scaffold, configs, and test setup)

---

## For Codex agent — Objectives

- Introduce a minimal, declarative pipeline with registered passes (no heavy external framework).
- Keep side-effects at the edges (adapters). Passes stay pure(ish) and composable.
- Preserve existing behavior; improve structure and testability.
- Enforce invariants (clean-before-split; PDF vs EPUB separation).
- Provide logging/metrics and a simple run report.

---

## For Codex agent — Ground Rules

- Work in **small, reversible commits**.
- Before each change, **present a short plan**.
- After each commit, run: `nox -s lint typecheck tests`.
- Always show a `git diff` for the commit (single block).
- Prefer **wrapping/extracting** over rewriting.
- Reuse existing functions; only add new ones if necessary.
- Keep code **declarative and functional** (pure transforms, composition, immutable data where practical).
- Use `pipeline.yaml` for configuration rather than conditionals in pass code.

---

## For Codex agent — Merge Policy for `core.py`

1. **Do not delete or rewrite `core.py` wholesale.**
2. Try to **wrap** existing logic from `core.py` inside the new orchestration (e.g., call existing functions from the pipeline runner).
3. If wrapping is awkward, create `core_new.py` with the new invariant checks and pipeline glue; point `cli.py` at it.
4. When moving functions out of `core.py` into `passes/*`, leave thin import **shims** to avoid breakage.
5. For any large edit, **propose the patch** first, then apply in small steps.

---

## For Codex agent — Files to Scaffold First

Implement everything listed in `PROJECT_BOOTSTRAP.md`, including:
- `pyproject.toml`, `.pre-commit-config.yaml`, `noxfile.py`
- `pipeline.yaml`
- `pdf_chunker/framework.py`, `pdf_chunker/config.py`, `pdf_chunker/cli.py`, `pdf_chunker/core.py` (or `core_new.py`)
- `pdf_chunker/adapters/*` (IO only), `pdf_chunker/passes/*` (pure transforms)
- Tests under `tests/unit` and `tests/golden`

Then run:

```bash
pip install -e ".[dev]"
pre-commit install
nox -s lint
nox -s typecheck
nox -s tests
```
For Codex agent — Results to paste

After each command block above (and for any subsequent commands you run), paste the following in this exact structure:
    * System info (once per session or when it changes):
      * Python version (python --version)
      * Pip list versions for key tools (ruff --version, black --version, mypy --version, pytest --version, nox --version)

    * Install output (first run that installs deps):
      * Last ~100 lines of pip install -e ".[dev]" (trimmed), or full error if it fails

    * Lint summary (nox -s lint):
      * Count of files changed by ruff --fix

    * Black check pass/fail summary line(s)
      * Typecheck summary (nox -s typecheck):

    * Mypy overall status line + the first 10 errors if failing (include full trace for any new error you introduce)

    * Tests summary (nox -s tests):
      * Pytest “passed/failed/xfail/xpass/skipped” line
      * Names of failing tests and their first traceback block (full trace if flaky or intermittent)

    * If anything fails:
      * Paste the full error/traceback for the failing command
      * State your diagnosis in 1–3 bullets and the fix you’ll attempt next

Use section headers in your message like:

```yaml
For Codex agent — results:
- System info: …
- Install: …
- Lint: …
- Typecheck: …
- Tests: …
```
For Codex agent — Invariants to Enforce
    * `text_clean` must run before any split step.
    * Do not mix PDF and EPUB passes in one pipeline.
    * Record fallback reasons & quality metrics.
    * Emit run_report.json with timings, metrics, and warnings for known issues.

For Codex agent — Execution Order

Follow the story cards in ARCHITECTURE_REFACTOR_TASKS.md:
    * Story A: Scaffold Framework & CLI.
    * Story B: Move Passes under passes/ and Register.
    * Story C: Adapters & IO Boundary.
    * Story D: Enforce Invariants.
    * Story E: Fallback Metrics & Run Report.
    * Story F: Golden & Property Tests.
    * Story G: Keep AGENTS.md in sync (auto-update script).

One (sub)story per small commit. After each step:
    * Show plan
    * Run nox sessions
    * Paste “results” (as above)
    * Show git diff
    * Propose the next step

For Codex agent — Commit Message Style
Use concise, conventional commits, e.g.:
    * `build: add pyproject and nox scaffolding`
    * `feat(pipeline): add registry and Typer CLI`
    * `refactor(passes): register text_clean and splitter`
    * `test(golden): add sample PDF snapshot`
    * `chore: enforce clean-before-split invariant`

For Codex agent — Safety Checks
    * If a change risks behavior drift, gate it behind tests or snapshot comparisons.
    * When moving code, preserve public signatures; add shims if needed.
    * Never remove core.py without explicit instruction; prefer core_new.py and a gradual cut-over.

For Codex agent — Non-destructive change policy (must follow)
    * Do not remove or overwrite existing, working logic unless explicitly instructed.
    * Additive first: when introducing stubs or new boundaries, add new functions or modules (e.g., `describe()` or `*_stub.py`) rather than replacing existing functions.
    * Keep public signatures stable. If you need a different shape, add an adapter/wrapper and leave a thin shim.
    * Propose before large edits: if you must modify >10 lines of an existing function, show the proposed diff first and wait for approval.
    * If a task description is ambiguous, default to non-destructive changes and ask to confirm before removing code.


For Codex agent — Legacy-aware migration rules

1. **Target the real functions (don’t re-invent):**
   Wrap these legacy entrypoints in passes/adapters exactly as named in the repo:

   * PDF parse: `pdf_parsing.extract_text_blocks_from_pdf`&#x20;
   * EPUB parse: `epub_parsing.extract_text_blocks_from_epub`&#x20;
   * Cleaning: `text_cleaning.clean_paragraph` / `clean_text`&#x20;
   * Headings: `heading_detection._detect_heading_fallback`&#x20;
   * Lists: `list_detection.*` helpers&#x20;
   * Splitter: `splitter.semantic_chunker`&#x20;
   * Fallbacks: `_extract_with_pdftotext` / `_extract_with_pdfminer`&#x20;
   * Enrichment: `_load_tag_configs`, `init_llm`, `classify_chunk_utterance`&#x20;
   * JSONL write (legacy): `scripts/chunk_pdf.py::main` (stdout)&#x20;

2. **Canonical artifact contracts (framework side):**

   * **After parse:** `PageBlocks` dict `{"type":"page_blocks","source_path":..., "pages":[{"page":int,"blocks":[{"text":..., ...}], ...}]}`. (Legacy blocks include `type`, `text`, `language`, `source{filename,page,location}`, optional `bbox`—preserve when lifting into blocks.)&#x20;
   * **After split:** `Chunks` dict `{"type":"chunks","items":[{"id":str,"text":str,"meta":{...}}]}`. Legacy row metadata fields (page, location, block\_type, language, readability, utterance\_type, importance, list\_kind) must be carried into `meta`.&#x20;

3. **IO boundaries (move side-effects into adapters):**

   * PDF open via `fitz.open` → `adapters.io_pdf.read` (and fallbacks via `subprocess.run`).&#x20;
   * EPUB open via `epub.read_epub` → `adapters.io_epub.read_epub`.&#x20;
   * LLM calls (`litellm.completion`) → enrichment client behind `ai_enrich` pass, disabled in tests.&#x20;
   * Legacy writer (stdout) → `adapters.emit_jsonl.write`.&#x20;

4. **Config ingestion (preserve flags & env):**

   * Env: `PDF_CHUNKER_USE_PYMUPDF4LLM`, `OPENAI_API_KEY`.&#x20;
   * YAML: tag vocabularies under `config/tags`.&#x20;
   * CLI flags to map into `pipeline.yaml` options: `--chunk_size` (400 default), `--overlap` (50), `--exclude-pages`, `--no-metadata`, `--list-spines`.&#x20;

5. **Behavioral invariants / edge cases to keep:**

   * Strip footnotes to avoid mid-sentence splits; remove header/footer artifacts (incl. trailing “|”); repair hyphenation and bullets; propagate list metadata (incl. PyMuPDF4LLM underscores); allow cross-page merges and comma-continuation fixes.&#x20;

6. **Performance limits to respect:**

   * Soft text limit 8 k chars (hard 25 k); `_truncate_chunk` trims beyond 8 k; default target 400 chars with 50 overlap; `min_chunk_size = max(8, chunk_size//10)`; `pdftotext` timeout 60 s; LLM completions ≤100 tokens.&#x20;

7. **Test guardrails:**

   * Keep listed tests green; add golden checks for `sample_*` PDFs and `test_data/sample_test.pdf`. &#x20;

8. **AGENTS.md write-safety:**

   * Only update content **between** `<!-- BEGIN AUTO-PASSES --> … <!-- END AUTO-PASSES -->`. Never edit outside the fenced block.

(**Non-destructive change policy** remains in force.)

Pass purity & IO rules

> **Pass purity (must):**
>
> * Passes **MUST NOT** open files, shell out, access network, or import modules whose top-level imports do IO.
> * A pass may only transform in-memory values (Artifacts).
> * Adapters perform all IO (PDF/EPUB read, subprocess fallbacks, JSONL write, LLM calls).
> * If a pass needs data, it must be **given** that data as its input Artifact.
Passes MUST NOT call legacy extractors or perform IO. Adapters (io_pdf, io_epub, emit_jsonl) perform all IO. A pass only transforms in-memory Artifacts.
>
> **Allowed imports inside passes:** `typing`, `itertools`, `collections.abc`, our `framework`, pure helpers under `passes/*`.
> **Disallowed inside passes:** `fitz`, `PyPDF*`, `subprocess`, `requests`, `litellm` (and any adapter or legacy extractor).
>
> **Testable guarantee:** we can monkeypatch any legacy extractor to raise; running the pass with in-memory payload should not call it.

For Codex agent — Reference files are read-only

* Treat `ARCHITECTURE_REFACTOR_TASKS.md`, `PROJECT_BOOTSTRAP.md`, and `MIGRATION_MAP.md` as **reference**.
* Do **not** modify or regenerate them unless explicitly instructed.

For Codex agent — Idempotent task behavior

* If the required file(s)/function(s) already exist and satisfy acceptance criteria, **make no changes** and mark the task **DONE (no-op)** with the evidence (nox output / CLI check).
* If a task would replace or delete working code, default to **non-destructive** (add wrappers/adapters) and ask for approval.

## Pre-merge review protocol (use this for M-parse(PDF) and similar)

**Checklist for the patch you’re reviewing:**

1. **No IO inside the pass.** No `fitz.open`, file reads, subprocess, or network calls.
2. **Input handling:** pass accepts either canonical `PageBlocks` **or** legacy `list[block]` (wraps via small helper).
3. **Output shape:** returns canonical `PageBlocks`
   `{"type":"page_blocks","source_path":..., "pages":[{"page":int,"blocks":[{...}], ...}]}`
   and preserves legacy block fields when present: `type`, `text`, `language`, `source {filename,page,location}`, optional `bbox`.
4. **Meta/metrics preserved:** returns a new `Artifact` with prior `meta` merged and `metrics["pdf_parse"]["pages"]` set.
5. **Registry key unchanged:** still registers as `"pdf_parse"`; `passes/__init__.py` only appends import (no deletions).
6. **Non-destructive:** legacy functions remain; patch adds wrappers/adapters only.
7. **Scope:** ≤ 2 files (3 only if unavoidable), ≤ 150 changed lines.
8. **Green checks:** scoped `nox -s lint typecheck tests` pass; `python -m pdf_chunker.cli inspect` lists `pdf_parse` with dict→dict.

For Codex agent — Start Now

Begin with Story A from `ARCHITECTURE_REFACTOR_TASKS.md`
Show your plan and proceed in small commits.
