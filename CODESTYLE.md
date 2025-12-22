# CODESTYLE.md — Declarative Python, Functional Bias, Practical Discipline

This guide favors code that is **readable, testable, and structurally obvious**.

**pdf-chunker** transforms PDF/EPUB documents into semantically coherent JSONL chunks. The codebase emphasizes pure functions, explicit data flow, and strict separation between transformation logic (passes) and I/O (adapters).

---

## Related Documentation

| Document | Purpose |
|----------|--------|
| [AGENTS.md](AGENTS.md) | Primary guidance for AI agents—domain map, constraints, workflow |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Structure, boundaries, and mental models |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution workflow, quality gates, PR expectations |

---

## 1) Defaults

Prefer:
- pure functions,
- explicit data flow,
- small modules,
- type hints on public interfaces,
- comprehensions and generator expressions.

Avoid:
- cleverness for its own sake,
- deeply nested mutation,
- implicit globals,
- “action at a distance” side effects.

## 2) Declarative > imperative (most of the time)

Prefer describing *what*:

### Good (declarative transform)
```python
clean = [x.strip() for x in lines if x.strip()]
```

### Acceptable (imperative shell)
```python
clean = []
for x in lines:
    s = x.strip()
    if s:
        clean.append(s)
```

Rule of thumb:
- Use comprehensions for small-to-medium transforms.
- Use loops when clarity genuinely improves or when control flow is complex.

## 3) Generator expressions: default for streaming

If data may be large, prefer generators:

```python
def normalized(lines: Iterable[str]) -> Iterable[str]:
    return (ln.strip() for ln in lines if ln.strip())
```

Consume at the boundary (I/O layer), not in the core.

## 4) Mutation policy

- Mutation is allowed, but it must be **local, obvious, and bounded**.
- Prefer returning new values from functions instead of mutating inputs.
- If you mutate, do it near where the variable is created.

Avoid:
- mutating arguments in-place unless clearly documented,
- hidden shared state.

## 5) Error handling

- Use exceptions for exceptional situations.
- Prefer domain-specific exceptions (small custom types) at boundaries.
- Don’t swallow exceptions silently.
- Error messages must be actionable (what failed, where, and why).

## 6) Types & data modeling

- Add type hints to all public functions and core logic.
- Prefer `dataclasses` for structured data.
- Prefer `Protocol` for structural typing at boundaries.

Keep models simple:
- avoid deep inheritance trees,
- prefer composition.

## 7) Docstrings and comments

Docstrings should explain:
- purpose,
- inputs/outputs,
- invariants,
- edge cases.

Comments should explain:
- why a choice was made,
- what tradeoff exists,
- what invariant must hold.

No comments that merely restate code.

## 8) Design patterns: use when they reduce confusion

Approved patterns (when useful):
- Strategy, Adapter, Factory, Command, Pipeline

Avoid:
- pattern maximalism,
- abstract base classes with one implementation,
- needless indirection.

## 9) “Small sharp tools” discipline (UNIX-ish)

Prefer multiple small functions over one sprawling procedure.
Prefer modules with one clear theme.
Prefer stable interfaces and boring internals.

## 10) Readability constraints (heuristics)

- One screen per function if possible.
- One responsibility per module.
- No “clever” one-liners that require rereading.

If a comprehension becomes cryptic:
- extract a named function,
- or use a simple loop.

Clarity beats brevity. Brevity beats boilerplate. Boilerplate beats confusion.

---

## 11) Formatting and tooling

This project uses:

| Tool | Purpose | Command |
|------|---------|--------|
| **Black** | Code formatting | `black .` |
| **Ruff** | Linting + import sorting | `ruff check --fix .` |
| **mypy** | Static type checking | `mypy pdf_chunker` |
| **pytest** | Testing | `pytest` or `nox -s tests` |
| **nox** | Session runner | `nox -s lint`, `nox -s typecheck`, `nox -s tests` |

Run all checks before committing:
```bash
nox -s lint
nox -s typecheck
nox -s tests
```

---

## 12) Project-specific conventions

### Pass purity
Passes (in `pdf_chunker/passes/`) must remain pure:
- No `fitz.open`, file reads, subprocess calls, or network access
- Transform in-memory `Artifact` values only
- All I/O belongs in adapters (`pdf_chunker/adapters/`)

### Artifact immutability
- `Artifact` is frozen (`frozen=True`)
- Build new dicts/lists on write; don't mutate inputs
- Copy metadata forward when transforming

### Configuration over conditionals
- Parameterize via `pipeline.yaml` options
- Avoid branching on flags inside pass logic
- Use `options[pass_name]` for pass-specific settings

### Inline styles
- Inline style spans follow the schema in [docs/inline_style_schema.md](docs/inline_style_schema.md)
- Consumers must gracefully handle `inline_styles is None`
- Adjacent spans with identical attributes are merged during normalization
