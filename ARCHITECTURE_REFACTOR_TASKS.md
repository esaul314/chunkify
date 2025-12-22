---

# Architecture Refactor — Codex Tasks

**Intent:** tame growth and complexity by introducing a minimal, declarative pipeline structure that treats each module as a pure transformation pass with strong boundaries and tests. Keep side-effects at the edges. Reuse small libraries where helpful; avoid heavy orchestrators for now.

**Anchors from AGENTS.md (must stay true):**

* Current responsibilities map to passes: `parsing`, `text_cleaning`, `heading_detection`, `extraction_fallbacks`, `splitter`, `ai_enrichment`, etc.
* Guidance: strict pass separation, log fallback reasons/metrics, run cleaning before splitting, and don’t mix EPUB logic in PDF modules.
* Known issues to surface in reports: mid-chunk footnote anchors; flaky page exclusion; metadata gaps on fallback; underscore emphasis loss in PyMuPDF4LLM cleanup.

---

## 0) Architectural Choice (pragmatic)

**We will not adopt a heavy external pipeline framework.**
Instead, we’ll compose a *micro-framework* from small, stable pieces:

* **Typer** (CLI)
* **Pydantic v2** + **YAML** (declarative config)
* **funcy/toolz** (function composition; optional)
* **structlog** (structured logging)
* **pluggy** (optional plugin discovery for alternate parse engines)
* A \~30-line **internal registry** for passes (simplest, fast, testable)

> If later we need scheduling/observability, we can wrap passes as Prefect tasks with thin adapters without changing internal contracts.

---

## 1) Target Layout

```
pdf_chunker/
  pdf_chunker/
    __init__.py
    framework.py            # Artifact/Pass/registry + pipeline runner
    core.py                 # load config, enforce invariants, run pipeline
    cli.py                  # Typer CLI: convert / inspect / bench
    config.py               # Pydantic Settings + YAML loader
    passes/
      parsing.py
      pdf_parsing.py
      epub_parsing.py
      text_cleaning.py
      heading_detection.py
      extraction_fallbacks.py
      page_artifacts.py
      splitter.py
      ai_enrichment.py
      source_matchers.py
      list_detection.py
      text_processing.py
      pymupdf4llm_integration.py
    adapters/
      io_pdf.py
      io_epub.py
      emit_jsonl.py
    utils/
      page_utils.py
      env_utils.py
      utils.py
  tests/
    unit/
    golden/
  pipeline.yaml
  pyproject.toml
  .pre-commit-config.yaml
  noxfile.py
```

---

## 2) Invariants (enforced)

* **Order:** `text_cleaning` must run **before** any splitting step.
* **Separation:** EPUB-specific logic must never execute in a PDF pipeline (and vice versa).
* **Fallbacks:** reasons + quality metrics must be logged & summarized.
* **Known issues:** detect & warn (not fail) for mid-chunk footnote anchors; flag page-exclusion no-ops; surface metadata gaps on fallback; note underscore loss from PyMuPDF4LLM cleanup.

---

## 3) Config — `pipeline.yaml` (declarative)

```yaml
pipeline:
  - pdf_parse
  - text_clean
  - heading_detect
  - split_semantic
  - ai_enrich
  - emit_jsonl

options:
  pdf_parse:
    engine: "native"        # or "pymupdf4llm"
  text_clean:
    normalize_quotes: true
    repair_ligatures: true
  split_semantic:
    target_tokens: 800
    hard_page_boundaries: false
  ai_enrich:
    tags_file: "tags.yaml"
  emit_jsonl:
    output_path: "out.jsonl"
```

---

## 4) Libraries to add (pyproject)

* `typer[all]`, `pydantic>=2`, `pyyaml`, `structlog`, `funcy` (or `toolz`), `pluggy`, `pytest`, `hypothesis`, `ruff`, `black`, `mypy`, `nox`, `pytest-regressions` (for golden snapshots).

---

## 5) Minimal framework (copy into `framework.py`)

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Dict, List, Type

@dataclass(frozen=True)
class Artifact:
    payload: Any
    meta: Dict[str, Any] | None = None

@runtime_checkable
class Pass(Protocol):
    name: str
    input_type: Type
    output_type: Type
    def __call__(self, a: Artifact) -> Artifact: ...

_REGISTRY: Dict[str, Pass] = {}

def register(p: Pass) -> Pass:
    _REGISTRY[p.name] = p
    return p

def run_pipeline(steps: List[str], a: Artifact) -> Artifact:
    for s in steps:
        a = _REGISTRY[s](a)
    return a

def registry() -> Dict[str, Pass]:
    return dict(_REGISTRY)
```

---

## 6) Config loader — `config.py`

```python
from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel
import yaml, os

class PipelineSpec(BaseModel):
    pipeline: List[str]
    options: Dict[str, Dict[str, Any]] = {}

def load_spec(path: str | None) -> PipelineSpec:
    data = {} if not path else yaml.safe_load(open(path)) or {}
    # Env overrides: STEP__key=value → options.step.key=value
    for k, v in os.environ.items():
        if "__" in k:
            step, key = k.lower().split("__", 1)
            data.setdefault("options", {}).setdefault(step, {})[key] = v
    return PipelineSpec.model_validate(data)
```

---

## 7) CLI — `cli.py` (Typer)

```python
import typer, json
from pdf_chunker.config import load_spec
from pdf_chunker.core import run_convert, run_inspect, run_bench

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command()
def convert(input_path: str, spec: str = "pipeline.yaml"):
    run_convert(input_path, load_spec(spec))

@app.command()
def inspect():
    print(json.dumps(run_inspect(), indent=2))

@app.command()
def bench(samples_dir: str = "tests/golden"):
    run_bench(samples_dir)

if __name__ == "__main__":
    app()
```

---

## 8) Orchestration — `core.py`

```python
from __future__ import annotations
from typing import Dict, Any
from copy import deepcopy
import structlog, time
from pdf_chunker.framework import Artifact, run_pipeline, registry
from pdf_chunker.config import PipelineSpec
from pdf_chunker.adapters import io_pdf, io_epub, emit_jsonl

log = structlog.get_logger(__name__)

PDF_STEPS = {"pdf_parse", "pymupdf4llm_parse"}
EPUB_STEPS = {"epub_parse"}
SPLIT_STEPS = {"split_semantic", "splitter", "split_pages"}  # whichever you use

def _enforce_invariants(spec: PipelineSpec):
    steps = spec.pipeline
    # Clean before split
    if any(s in SPLIT_STEPS for s in steps):
        try:
            clean_idx = steps.index("text_clean")
            split_idx = min(i for i, s in enumerate(steps) if s in SPLIT_STEPS)
            assert clean_idx < split_idx, "text_clean must run before any split step"
        except ValueError:
            raise AssertionError("text_clean is required before splitting")
    # Separation: PDF vs EPUB
    if (PDF_STEPS & set(steps)) and (EPUB_STEPS & set(steps)):
        raise AssertionError("Do not mix EPUB and PDF passes in one pipeline")

def run_convert(input_path: str, spec: PipelineSpec):
    _enforce_invariants(spec)
    # choose adapter by extension
    adapter = io_epub if input_path.lower().endswith((".epub",)) else io_pdf
    a = Artifact(payload=adapter.read(input_path), meta={"metrics": {}, "input": input_path})
    timings: Dict[str, float] = {}
    for s in spec.pipeline:
        t0 = time.time()
        a = run_pipeline([s], a)
        timings[s] = time.time() - t0
    # terminal write if last step packaged rows
    emit_jsonl.maybe_write(a, spec.options.get("emit_jsonl", {}), timings)
    return a

def run_inspect():
    return {name: {"input": str(p.input_type), "output": str(p.output_type)}
            for name, p in registry().items()}

def run_bench(samples_dir: str):
    # stub: iterate samples, run pipeline, write run_report.json (timings, metrics, warnings)
    pass
```

---

## 9) Example pass — `passes/text_cleaning.py`

```python
from __future__ import annotations
from typing import Dict, Any
from funcy import compose
from pdf_chunker.framework import Artifact, register

def _normalize_quotes(blocks): ...
def _repair_ligatures(blocks): ...
def _strip_ctrl(blocks): ...

class _TextCleanPass:
    name = "text_clean"
    input_type = dict    # replace with your PageBlocks type
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        transform = compose(_strip_ctrl, _repair_ligatures, _normalize_quotes)
        return Artifact(payload=transform(a.payload), meta=a.meta)

text_clean = register(_TextCleanPass())
```

> Repeat the pattern for the other passes listed in AGENTS.md: parsing, heading detection, fallbacks, splitter, AI enrichment, list detection, etc.

---

## 10) Adapters (side-effects only)

**`adapters/io_pdf.py`** — read & return a pure structure (e.g., `Document`/`PageBlocks`).
**`adapters/io_epub.py`** — analogous for EPUB.
**`adapters/emit_jsonl.py`** — write rows if the terminal pass didn’t already persist.

Also implement **optional** `pymupdf4llm_integration` pass that registers only if the import succeeds (alternate `pdf_parse`), and ensure cleanup lives there.

---

## 11) Logging & Metrics

* Initialize **structlog** in `core.py`.
* Each pass can attach metrics by returning a new `Artifact` with `meta["metrics"][pass_name] = {...}` (copy-on-write to preserve immutability).
* **extraction\_fallbacks** must record **reason** and **score**, and totals appear in a run report.

---

## 12) Tests

1. **Golden snapshots:** run 2–3 PDFs + 1 EPUB → assert JSONL equality (use `pytest-regressions`).
2. **Invariants:** reject pipelines violating clean-before-split or mixing EPUB/PDF.
3. **Property tests:** idempotence of hyphen/line joins; list detection robustness.
4. **Known issues surfaced as warnings** in a `run_report.json`.

---

## 13) Tooling

* `pre-commit` (ruff, black, mypy, eof-fixer, trailing-ws)
* `nox` sessions: `lint`, `typecheck`, `tests`
* `mypy` in strict mode; keep pass input/output types honest.

---

## 14) Keep AGENTS.md in Sync

Add `scripts/update_agents_md.py`: introspect `framework.registry()` and overwrite the Responsibilities table to mirror real passes + modules, so the doc never drifts from code. (AGENTS.md explicitly asks to keep structure current.)

---

## 15) Optional: Borrow an External Orchestrator Later

If/when you need UI/observability/scheduling:

* Wrap each pass `__call__` in a thin Prefect `@task` and the configured pipeline in a `@flow`.
* Preserve pure transforms; keep IO in adapters; don’t leak Prefect imports into pass internals.
* This can be added without touching the declarative config or pass contracts.

---

## 16) Definition of Done

* `pdf_chunker convert input.pdf` runs with `pipeline.yaml`, produces the **same JSONL** (or a documented delta).
* `pdf_chunker inspect` lists registered passes + declared input/output types.
* `pdf_chunker bench` generates `run_report.json` with timings, fallback reasons, chunk/token histograms, and warnings for known issues.
* Invariants enforced; violations fail fast with clear messages.
* Tests green; pre-commit clean.

---

## 17) Codex Work Plan (Story Cards)

> Use these “stories” verbatim. Each has acceptance criteria.

* **Story A: Scaffold Framework & CLI**
  Create `framework.py`, `config.py`, `core.py`, `cli.py`; add Typer, Pydantic, structlog; wire `convert`, `inspect`, `bench`.
  **AC:** CLI runs; `inspect` shows empty/partial registry; unit tests pass.

* **Story B: Move Passes Under `passes/` and Register**
  Convert each AGENTS.md responsibility into a registered pass with declared `input_type`/`output_type`. Keep pure transforms internal.
  **AC:** Registry lists passes: parsing, text\_cleaning, heading\_detection, extraction\_fallbacks, splitter, ai\_enrichment, list\_detection, etc.

* **Story C: Adapters & IO Boundary**
  Implement `io_pdf`, `io_epub`, `emit_jsonl`. No IO inside passes.
  **AC:** `convert` reads input by extension and writes output only via `emit_jsonl`.

* **Story D: Enforce Invariants**
  Implement order/separation checks in `core._enforce_invariants`.
  **AC:** Pipelines that violate clean-before-split or mix EPUB/PDF fail fast.

* **Story E: Fallback Metrics & Run Report**
  Record fallback reasons/scores; write `run_report.json` with timings, histograms, warnings for known issues.
  **AC:** Report contains counts + reasons from `extraction_fallbacks`; warnings include footnotes/page-exclusion/metadata/underscore issues.

* **Story F: Golden & Property Tests**
  Add golden tests for sample docs; property tests for text repair & list detection.
  **AC:** Test suite green; regressions visible.

* **Story G: Keep AGENTS.md Fresh**
  Script to regenerate the Responsibilities section from `registry()`.
  **AC:** Running the script updates the table; doc matches code.

---

### Notes on Style (Programming Philosophy)

* Prefer small, **pure** transforms and **composition** (`funcy.compose`).
* Use **immutable** `Artifact` (`frozen=True`) and build new dicts/lists on write.
* Keep **configuration declarative**; avoid branching on flags inside pass logic—parameterize via `options[pass_name]`.
* Side-effects belong only in adapters or CLI.

---
