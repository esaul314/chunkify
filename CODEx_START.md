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
    * ``text_clean`` must run before any split step.
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

For Codex agent — Start Now

Begin with Story A from `ARCHITECTURE_REFACTOR_TASKS.md`
Show your plan and proceed in small commits.
