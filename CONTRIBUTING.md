# CONTRIBUTING.md — Workflow, Quality Gates, and Review Discipline

This repo values small diffs, strong tests, and obvious structure.

**pdf-chunker** is a modular Python library for processing large PDF and EPUB documents. Contributions should preserve the functional core / imperative shell architecture and maintain pass purity.

---

## Related Documentation

| Document | Purpose |
|----------|--------|
| [AGENTS.md](AGENTS.md) | Primary guidance for AI agents—domain map, constraints, workflow |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Structure, boundaries, and mental models |
| [CODESTYLE.md](CODESTYLE.md) | Code style, patterns, and formatting standards |

---

## 1) Development environment (bash)

Create and activate a venv:

```bash
python -m venv pdf-env
source pdf-env/bin/activate
```

Install system dependencies (for pdftotext fallback):

```bash
apt-get install -y poppler-utils  # provides pdftotext
```

Install in editable mode with dev dependencies (EPUB support is part of the
base runtime):

```bash
pip install -e ".[dev]"
```

Install the optional enrichment client when running enrichment-enabled
conversions locally:

```bash
pip install -e ".[dev,enrichment]"
```

For the base runtime only:

```bash
pip install -e .
```

## 2) Quality gates (run before pushing)

Use nox sessions for consistency:

```bash
nox -s lint       # ruff check + black --check
nox -s typecheck  # mypy pdf_chunker
nox -s tests      # pytest
```

All three sessions explicitly use Python 3.11, matching CI, regardless of the
interpreter running nox. A missing interpreter fails the run instead of skipping
the checks. Install Python 3.11 locally, or let a recent nox provision an isolated
standalone interpreter:

```bash
python -m pip install --upgrade "nox[pbs]"
nox -s lint typecheck tests
```

Provisioning needs network access on the first run and does not replace
`pdf-env`. The downloader is optional developer tooling, not a project runtime
dependency. An exploratory run on another interpreter is explicit, for example
`nox --force-python 3.14 -s lint typecheck tests`; it does not substitute for the
supported-Python baseline.

The lint and typecheck sessions enforce the incremental production target set
declared in `noxfile.py`. Expand that set only in reviewable batches; use
`ruff check --no-fix pdf_chunker scripts tests` and `mypy pdf_chunker` as broader
diagnostic checks while repository-wide debt remains.

Target declarations are required, not best-effort discovery. `QUALITY_PATHS`,
`LINT_EXTRA_PATHS`, and `TEST_TARGETS` must each be non-empty and contain no blank
paths. Each session validates its groups before installing dependencies or running
tools. A missing file or directory raises `FileNotFoundError` with the affected
group and every missing path in first-seen order; an empty group or blank path
raises `ValueError`. Restore a missing target or deliberately update the declaration
and scope tests after a move—do not rely on a deleted file silently leaving the
gate. Valid declarations retain their spelling, ordered deduplication, commands,
and dependency profiles. For example, after moving a gated module, update its path
in `noxfile.py`, then run:

```bash
pytest tests/bootstrap/test_quality_gate_scope.py
nox -s lint typecheck tests
```

Or run tools directly:

```bash
pytest
ruff check --no-fix .
black .
mypy pdf_chunker
```

Quick smoke test:

```bash
pdf_chunker convert ./platform-eng-excerpt.pdf --spec pipeline.yaml --out ./data/test.jsonl --no-enrich
```

## 3) Branching and commits

- Prefer short-lived branches.
- Keep commits coherent: one commit per conceptual change.

Commit message style:
- If the repo uses Conventional Commits, follow it:
  - `fix: ...`, `feat: ...`, `refactor: ...`, `test: ...`, `docs: ...`
- Otherwise, use an imperative sentence:
  - `Fix temp dir creation on Fedora`
Include a brief commit message body describing the change, key tests run, and any notable risks.

## 4) Pull request expectations

A PR should include:

- **Problem**: what’s broken or missing
- **Approach**: what you changed and why
- **Evidence**: tests, commands run, or reproducible verification
- **Risk**: what might still be fragile
- **Docs**: updated if behavior or usage changed

Checklist:
- [ ] Tests added/updated
- [ ] `pytest` passes
- [ ] Lint/format pass
- [ ] Types pass (if applicable)
- [ ] No unnecessary refactor churn
- [ ] Structure remains easy to grasp

## 5) Refactoring policy (anti-overengineering)

Refactoring is welcome when it:
- reduces duplication,
- shrinks cognitive load,
- isolates boundaries,
- improves testability.

Refactoring is *not* welcome when it:
- is broad and stylistic,
- adds abstraction without clear payoff,
- changes behavior “incidentally.”

When in doubt: ship the smallest correct fix first.

## 6) How to add new functionality

Prefer this sequence:
1. Add/extend pure core functions (easy to test).
2. Add adapters for new I/O or integrations.
3. Wire via a thin orchestration layer.
4. Expose via CLI or API boundary.

Keep boundaries crisp; keep side effects out of the core.

## Publication privacy (required before sharing)

Before any public push, pull request, tag, release or patch export, complete the
[publication privacy gate](docs/PUBLICATION_PRIVACY.md). Review the entire outgoing
history and inherited content, including intermediate/deleted blobs, commit
metadata and encoded fixtures. Keep actual processed materials and private
operational details outside tracked source. Passing tests or a clean newest commit
do not clear older commits; unresolved or unreviewed content blocks publication.

## 7) Security and dependency hygiene

- Avoid adding dependencies unless necessary.
- Prefer standard library solutions when adequate.
- If you must add a dependency, document why and what alternatives were rejected.

## 8) Using AI agents in this repo

If an AI agent is used to author changes, it must follow [AGENTS.md](AGENTS.md), including:

- **Output Protocol: Voice + Ledger** reporting format (Voice ≤ 5 lines; Ledger audit-grade with explicit goals accomplished and evidence)
- **Pass purity constraints**: no I/O inside passes; adapters handle all side effects
- **Non-destructive change policy**: wrap/adapt existing functions rather than deleting; keep public signatures stable
- **Small, reversible commits**: one concern per commit; show plan before non-trivial work

---

## 9) Project-specific workflow

### Running the CLI

```bash
# Convert PDF to JSONL
pdf_chunker convert input.pdf --spec pipeline.yaml --out output.jsonl

# Trace a phrase through the pipeline (debugging)
pdf_chunker convert input.pdf --spec pipeline.yaml --out output.jsonl --trace "search phrase"

# Inspect registered passes
pdf_chunker inspect
```

### Golden tests

Golden fixtures in `tests/golden/` use Base64-encoded samples. To update goldens after intentional behavior changes:

```bash
python scripts/refresh_goldens.py --approve
```

### Debugging directions

- When JSONL lines begin mid-sentence or phrases repeat, inspect the `split_semantic` pass first
- Use `--trace <phrase>` to pinpoint which pass introduces anomalies
- Set `PDF_CHUNKER_DEDUP_DEBUG=1` to emit warnings for dropped duplicates
