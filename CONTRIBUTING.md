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

Install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

If no extras exist yet:

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

Or run tools directly:

```bash
pytest
ruff check --fix .
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

## 7) Security and dependency hygiene

- Avoid adding dependencies unless necessary.
- Prefer standard library solutions when adequate.
- If you must add a dependency, document why and what alternatives were rejected.

## 8) Using AI agents in this repo

If an AI agent is used to author changes, it must follow [AGENTS.md](AGENTS.md), including:

- **Mode B Voice + Ledger** reporting format (Voice ≤ 5 lines; Ledger audit-grade with explicit goals accomplished and evidence)
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
