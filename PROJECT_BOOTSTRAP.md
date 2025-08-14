---

This file defines the base project scaffolding, dev tooling, and “starter” tests. Codex can read this and create/modify the listed files.

## 1) Files to add

### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "pdf-chunker"
version = "0.1.0"
description = "CLI pipeline to convert PDFs/EPUBs into JSONL for LoRA/RAG, with declarative passes."
authors = [{name = "Alex"}]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
  "typer[all]>=0.12",
  "pydantic>=2.4",
  "PyYAML>=6.0",
  "structlog>=24.1",
  "funcy>=2.0",
  "pluggy>=1.5",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "pytest-regressions>=2.6",
  "hypothesis>=6.108",
  "ruff>=0.5.5",
  "black>=24.8.0",
  "mypy>=1.11.2",
  "nox>=2024.4.15",
  "types-PyYAML>=6.0.12",
]

[project.scripts]
pdf_chunker = "pdf_chunker.cli:app"

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E","F","I","UP","B","SIM"]
ignore = ["E501"] # handled by Black
fix = true

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_configs = true
warn_redundant_casts = true
warn_return_any = true
warn_unreachable = true
pretty = true
show_error_codes = true
plugins = []

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace

  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.5
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies:
          - types-PyYAML>=6.0.12
```

### `noxfile.py`

```python
import nox

@nox.session
def lint(session):
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "--fix", ".")
    session.run("black", "--check", ".")

@nox.session
def typecheck(session):
    session.install("-e", ".[dev]")
    session.run("mypy", "pdf_chunker")

@nox.session
def tests(session):
    session.install("-e", ".[dev]")
    session.run("pytest")
```

### `pipeline.yaml` (default)

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
    engine: "native"
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

### `pdf_chunker/framework.py`

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

### `pdf_chunker/config.py`

```python
from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel
import yaml, os

class PipelineSpec(BaseModel):
    pipeline: List[str]
    options: Dict[str, Dict[str, Any]] = {}

def load_spec(path: str | None) -> PipelineSpec:
    data: Dict[str, Any] = {} if not path else (yaml.safe_load(open(path)) or {})
    # Env overrides: STEP__key=value → options.step.key=value
    for k, v in os.environ.items():
        if "__" in k:
            step, key = k.lower().split("__", 1)
            data.setdefault("options", {}).setdefault(step, {})[key] = v
    return PipelineSpec.model_validate(data)
```

### `pdf_chunker/cli.py`

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

### `pdf_chunker/core.py`

```python
from __future__ import annotations
from typing import Dict, Any
import structlog, time, json, pathlib
from pdf_chunker.framework import Artifact, run_pipeline, registry
from pdf_chunker.config import PipelineSpec
from pdf_chunker.adapters import io_pdf, io_epub, emit_jsonl

log = structlog.get_logger(__name__)

PDF_STEPS = {"pdf_parse", "pymupdf4llm_parse"}
EPUB_STEPS = {"epub_parse"}
SPLIT_STEPS = {"split_semantic", "splitter", "split_pages"}

def _enforce_invariants(spec: PipelineSpec):
    steps = spec.pipeline
    # Clean before split
    if any(s in SPLIT_STEPS for s in steps):
        if "text_clean" not in steps:
            raise AssertionError("text_clean is required before splitting")
        clean_idx = steps.index("text_clean")
        split_idx = min(i for i, s in enumerate(steps) if s in SPLIT_STEPS)
        if clean_idx >= split_idx:
            raise AssertionError("text_clean must run before any split step")
    # Separation: PDF vs EPUB
    if (PDF_STEPS & set(steps)) and (EPUB_STEPS & set(steps)):
        raise AssertionError("Do not mix EPUB and PDF passes in one pipeline")

def run_convert(input_path: str, spec: PipelineSpec):
    _enforce_invariants(spec)
    adapter = io_epub if input_path.lower().endswith((".epub",)) else io_pdf
    a = Artifact(payload=adapter.read(input_path), meta={"metrics": {}, "input": input_path})
    timings: Dict[str, float] = {}
    for s in spec.pipeline:
        t0 = time.time()
        a = run_pipeline([s], a)
        timings[s] = time.time() - t0
    emit_jsonl.maybe_write(a, spec.options.get("emit_jsonl", {}), timings)
    _write_run_report(timings, a.meta or {})
    return a

def run_inspect():
    return {name: {"input": str(p.input_type), "output": str(p.output_type)}
            for name, p in registry().items()}

def run_bench(samples_dir: str):
    # Placeholder: iterate PDFs/EPUBs and call run_convert on each
    pass

def _write_run_report(timings: Dict[str, float], meta: Dict[str, Any]):
    report = {"timings": timings, "metrics": (meta.get("metrics") or {}), "warnings": meta.get("warnings", [])}
    pathlib.Path("run_report.json").write_text(json.dumps(report, indent=2))
```

### Adapters (stubs to keep side-effects at the edges)

`pdf_chunker/adapters/io_pdf.py`

```python
def read(path: str):
    # TODO: implement actual PDF read → return pure structure (e.g., Document/PageBlocks)
    return {"type": "pdf_document", "path": path}
```

`pdf_chunker/adapters/io_epub.py`

```python
def read(path: str):
    # TODO: implement actual EPUB read → return pure structure
    return {"type": "epub_document", "path": path}
```

`pdf_chunker/adapters/emit_jsonl.py`

```python
import json

def maybe_write(artifact, options, timings):
    # If the terminal pass already wrote the file, this can no-op.
    out_path = options.get("output_path")
    if not out_path:
        return
    payload = artifact.payload
    # Expect list[dict] for JSONL; adapt as needed.
    if isinstance(payload, list):
        with open(out_path, "w") as f:
            for row in payload:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

### Example pass (pattern)

`pdf_chunker/passes/text_cleaning.py`

```python
from __future__ import annotations
from funcy import compose
from pdf_chunker.framework import Artifact, register

def _normalize_quotes(blocks): return blocks
def _repair_ligatures(blocks): return blocks
def _strip_ctrl(blocks): return blocks

class _TextCleanPass:
    name = "text_clean"
    input_type = dict    # replace with your concrete PageBlocks type
    output_type = dict

    def __call__(self, a: Artifact) -> Artifact:
        transform = compose(_strip_ctrl, _repair_ligatures, _normalize_quotes)
        return Artifact(payload=transform(a.payload), meta=a.meta)

text_clean = register(_TextCleanPass())
```

> Repeat the same pattern for: `pdf_parsing.py`, `epub_parsing.py`, `heading_detection.py`, `extraction_fallbacks.py`, `splitter.py` (or `split_semantic.py`), `ai_enrichment.py`, `source_matchers.py`, `list_detection.py`, `pymupdf4llm_integration.py` (register only if import works).

---

## 2) Test scaffolding

```
tests/
  unit/
    test_invariants.py
    test_registry_sync.py
  golden/
    README.md
    samples/               # put a few small PDFs/EPUBs here (you add)
    expected/              # JSONL snapshots here (committed)
    test_conversion.py
```

### `tests/unit/test_invariants.py`

```python
import pytest
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core import _enforce_invariants

def test_requires_text_clean_before_split():
    spec = PipelineSpec(pipeline=["pdf_parse", "split_semantic"])
    with pytest.raises(AssertionError):
        _enforce_invariants(spec)

def test_pdf_epub_separation():
    spec = PipelineSpec(pipeline=["pdf_parse", "epub_parse"])
    with pytest.raises(AssertionError):
        _enforce_invariants(spec)
```

### `tests/unit/test_registry_sync.py`

```python
from pdf_chunker.framework import registry

def test_expected_passes_exist():
    # Adjust to your actual list as you implement
    expected = {
        "pdf_parse", "text_clean", "heading_detect", "split_semantic",
        "ai_enrich", "emit_jsonl"
    }
    assert expected.issubset(set(registry().keys()))
```

### `tests/golden/README.md`

```md
Place a few representative PDFs/EPUBs in `samples/`. Run the CLI to generate JSONL, then copy the outputs into `expected/` and commit.

We use `pytest-regressions` to compare structured results; prefer normalizing to lists/dicts before asserting.
```

### `tests/golden/test_conversion.py`

```python
import json, pathlib
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core import run_convert

SAMPLES = pathlib.Path(__file__).parent / "samples"
EXPECTED = pathlib.Path(__file__).parent / "expected"

def _read_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def test_sample_pdf(tmp_path):
    sample = str(SAMPLES / "sample.pdf")   # you provide this
    out = tmp_path / "out.jsonl"
    spec = PipelineSpec(pipeline=[
        "pdf_parse","text_clean","heading_detect","split_semantic","ai_enrich","emit_jsonl"
    ], options={"emit_jsonl": {"output_path": str(out)}})
    run_convert(sample, spec)
    got = _read_jsonl(out)
    want = _read_jsonl(EXPECTED / "sample.pdf.jsonl")
    assert got == want
```

---

## 3) (Optional) Keep AGENTS.md in sync

`scripts/update_agents_md.py` (skeleton)

```python
from pdf_chunker.framework import registry

def main():
    reg = registry()
    lines = ["| Pass | Module |", "|---|---|"]
    for name, p in sorted(reg.items()):
        mod = p.__class__.__module__
        lines.append(f"| `{name}` | `{mod}` |")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
```

---

## 4) How to run locally

```bash
# Install runtime + dev deps
pip install -e ".[dev]"

# One-time: enable pre-commit
pre-commit install

# Lint / typecheck / tests
nox -s lint
nox -s typecheck
nox -s tests

# CLI
pdf_chunker --help
pdf_chunker convert path/to/input.pdf --spec pipeline.yaml
pdf_chunker inspect
```

---
