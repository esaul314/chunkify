# STRATEGIC_REFACTORING_PLAN.md — From Complexity to Clarity

**Date:** 2026-01-26  
**Status:** APPROVED FOR IMPLEMENTATION  
**Author:** Codebase Steward (with system analysis)

---

## Executive Summary

This plan transforms the pdf_chunker codebase from a 1,900+ line monolith (`split_semantic.py`) into a **declarative, composable, self-documenting pipeline**. The key insight: most complexity comes from **ambiguous merge decisions** that should be surfaced to users via interactive mode, not buried in heuristics.

### Core Principles

1. **Interactive mode as the escape hatch** — When patterns conflict, ask the user
2. **Declarative over imperative** — Rules > nested conditionals
3. **Composition over inheritance** — Pipe-style transforms > class hierarchies
4. **Audit trails over debugging** — Every decision logged, every merge explained

---

## Phase 0: Foundation — Transformation Audit Trail (2-3 days)

**Goal:** Know what happened to any piece of text.

### Task 0.1: Create `TransformationLog` dataclass

```python
# pdf_chunker/passes/transform_log.py

from dataclasses import dataclass, field
from typing import Literal
import hashlib

TransformKind = Literal[
    "extracted", "cleaned", "merged", "split", 
    "heading_attach", "deduplicated", "pattern_match",
    "interactive_decision"
]

@dataclass(frozen=True)
class TransformEntry:
    kind: TransformKind
    pass_name: str
    reason: str  # Human-readable explanation
    source_hash: str
    result_hash: str

@dataclass
class TransformationLog:
    fragment_id: str
    entries: list[TransformEntry] = field(default_factory=list)
    
    def record(self, kind: TransformKind, pass_name: str, reason: str,
               source: str, result: str) -> None:
        self.entries.append(TransformEntry(
            kind=kind, pass_name=pass_name, reason=reason,
            source_hash=hashlib.md5(source.encode()).hexdigest()[:8],
            result_hash=hashlib.md5(result.encode()).hexdigest()[:8],
        ))
```

### Task 0.2: Wire into `--trace` mode

- Extend existing `--trace <phrase>` to include transformation log
- Output format: `[pass_name] kind: reason (source_hash → result_hash)`

### Task 0.3: Add logging to existing merge points

Add `reason` returns to:
- `_stitch_block_continuations`
- `_merge_styled_list_records`  
- `_is_qa_sequence_continuation`

**Acceptance criteria:**
- [ ] Any text can be traced through entire pipeline
- [ ] Merge reasons visible in trace output
- [ ] No performance impact when `--trace` not used

---

## Phase 1: Pattern Registry — Declarative Merge Rules (3-4 days)

**Goal:** Replace scattered regex checks with a single, testable registry.

### Task 1.1: Create `TextPatternRegistry`

```python
# pdf_chunker/patterns.py

from dataclasses import dataclass
from enum import IntEnum
import re

class Precedence(IntEnum):
    CRITICAL = 10   # Q&A, numbered lists — always merge
    HIGH = 20       # Bullet continuations with clear signals
    MEDIUM = 30     # Heading attachment
    LOW = 40        # Heuristic-based merges
    BOUNDARY = 100  # Always split (chapter headings)

class MergeBehavior(Enum):
    MERGE = "merge"
    ASK = "ask"       # Use interactive callback
    SPLIT = "split"
    BOUNDARY = "boundary"

@dataclass(frozen=True)
class Pattern:
    name: str
    match: re.Pattern[str]
    precedence: Precedence
    behavior: MergeBehavior
    description: str
    continuation: re.Pattern[str] | None = None
    
DEFAULT_PATTERNS = [
    Pattern("qa_sequence", re.compile(r"[QA]\d+:", re.I), 
            Precedence.CRITICAL, MergeBehavior.MERGE,
            "Q&A interview format",
            re.compile(r"^[QA]\d+:", re.I)),
    Pattern("numbered_list", re.compile(r"^\s*\d+[\.\)]\s"),
            Precedence.CRITICAL, MergeBehavior.MERGE,
            "Numbered list items"),
    Pattern("step_sequence", re.compile(r"^Step\s+\d+[:\.]", re.I),
            Precedence.CRITICAL, MergeBehavior.MERGE,
            "Step-by-step instructions"),
    Pattern("bullet_list", re.compile(r"^\s*[•\-\*]\s"),
            Precedence.HIGH, MergeBehavior.MERGE,
            "Bullet points"),
    Pattern("chapter_heading", re.compile(r"^Chapter\s+\d+", re.I),
            Precedence.BOUNDARY, MergeBehavior.BOUNDARY,
            "Chapter heading — always split"),
    # ... more from REFACTORING_ROADMAP.md Appendix
]
```

### Task 1.2: Refactor `_is_qa_sequence_continuation` → registry lookup

Replace:
```python
# Before
if _is_qa_sequence_continuation(prev_text, curr_text):
    ...
```
With:
```python
# After
decision = registry.should_merge(prev_text, curr_text)
if decision.should_merge:
    ...
```

### Task 1.3: Interactive learning for ambiguous patterns

When `MergeBehavior.ASK` and interactive mode enabled:
```
┌─────────────────────────────────────────────────────────────┐
│ Pattern "dialogue_tags" detected:                           │
│                                                             │
│ Previous ends: "...we're spread really thin."               │
│ Current starts: "Alice: So what do we do about it?"         │
│                                                             │
│ Should these be kept together? [Y/n/always/never]           │
└─────────────────────────────────────────────────────────────┘
```

- "always" → pattern behavior becomes MERGE for this document
- "never" → pattern behavior becomes SPLIT for this document

**Acceptance criteria:**
- [ ] All existing pattern checks use registry
- [ ] Interactive learning works for ambiguous patterns
- [ ] Unit tests for each pattern in isolation
- [ ] Precedence ordering is explicit and testable

---

## Phase 2: Pipe-Style Composition (2-3 days)

**Goal:** Make the pipeline readable as data flow.

### Task 2.1: Adopt `pipe` library for internal transforms

Add to `pyproject.toml`:
```toml
"pipe>=2.0",
```

### Task 2.2: Rewrite `_rows` pipeline with pipe notation

```python
# Before (imperative)
def _rows(doc, *, preserve, explicit_small, strategy):
    items = _sanitize_items(doc.get("items", []))
    if not explicit_small:
        items = _merge_very_short_forward(list(items), strategy=heuristics)
    processed = (
        items if preserve else
        _dedupe(_coalesce(items, strategy=heuristics), log=debug_log)
    )
    # ... more nesting

# After (declarative pipe)
from pipe import Pipe, where

sanitize = Pipe(_sanitize_items)
merge_short = Pipe(lambda items: _merge_very_short_forward(list(items), strategy=heuristics))
coalesce = Pipe(lambda items: _coalesce(items, strategy=heuristics))
dedupe = Pipe(lambda items: _dedupe(items, log=debug_log))

def _rows(doc, *, preserve, explicit_small, strategy):
    pipeline = (
        doc.get("items", [])
        | sanitize
        | (merge_short if not explicit_small else identity)
        | (identity if preserve else coalesce | dedupe)
    )
    return list(pipeline)
```

### Task 2.3: Create named pipeline stages

```python
# pdf_chunker/pipelines/text_pipeline.py

from pipe import Pipe

# Each stage is independently testable
strip_footers = Pipe(_strip_footer_artifacts)
merge_continuations = Pipe(_stitch_block_continuations)
attach_headings = Pipe(pipeline_attach_headings)
split_chunks = Pipe(_split_to_chunks)

# Composed pipeline is self-documenting
text_to_chunks = (
    strip_footers
    | merge_continuations  
    | attach_headings
    | split_chunks
)
```

**Acceptance criteria:**
- [ ] At least 3 major functions use pipe composition
- [ ] Pipeline stages are independently unit-testable
- [ ] No regression in test suite

---

## Phase 3: Module Decomposition (4-5 days)

**Goal:** `split_semantic.py` < 300 lines.

### Task 3.1: Create `split_semantic/` subpackage

```
pdf_chunker/passes/
├── split_semantic.py           # Entry point only (~150 lines)
└── split_semantic/
    ├── __init__.py             # Re-exports public API
    ├── stitching.py            # _stitch_block_continuations etc.
    ├── footers.py              # Footer detection
    ├── overlap.py              # Boundary overlap handling
    ├── emission.py             # Segment emission
    └── registry.py             # Pattern registry integration
```

### Task 3.2: Move functions one group at a time

**Order (least → most interconnected):**
1. `footers.py` — self-contained footer detection
2. `overlap.py` — boundary overlap, depends only on text utils
3. `emission.py` — segment emission, depends on metadata
4. `stitching.py` — block stitching, depends on patterns
5. Update `split_semantic.py` to import from subpackage

### Task 3.3: Run tests after each move

```bash
# After each file move:
nox -s tests -- tests/passes/test_split_semantic*.py
nox -s tests -- tests/emit_jsonl*.py
```

**Acceptance criteria:**
- [ ] `split_semantic.py` ≤ 300 lines
- [ ] All imports work from existing consumers
- [ ] No test regressions
- [ ] Each submodule has own docstring explaining responsibility

---

## Phase 4: Interactive Mode as Core Feature (3-4 days)

**Goal:** Use interactive mode to resolve ambiguity instead of adding heuristics.

### Task 4.1: Unify interactive callbacks

Current state: separate callbacks for footers, lists, patterns.

Proposed: Single `InteractiveDecisionCallback` protocol:

```python
class DecisionKind(Enum):
    FOOTER = "footer"
    LIST_CONTINUATION = "list_continuation"
    PATTERN_MERGE = "pattern_merge"
    HEADING_BOUNDARY = "heading_boundary"

class InteractiveDecisionCallback(Protocol):
    def __call__(
        self,
        kind: DecisionKind,
        context: DecisionContext,
    ) -> Decision:
        ...

@dataclass
class DecisionContext:
    prev_text: str | None
    curr_text: str
    page: int
    pattern: Pattern | None
    confidence: float
    
@dataclass
class Decision:
    action: Literal["merge", "split", "skip"]
    remember: Literal["once", "always", "never"]
```

### Task 4.2: Add `--teach` mode for persistent learning

```bash
# Run once to teach the system about a document type
pdf_chunker convert ./example.pdf --teach --out ./trained.jsonl

# Decisions saved to ~/.config/pdf_chunker/learned_patterns.yaml
# Subsequent runs use learned patterns automatically
```

### Task 4.3: Default to interactive for new edge cases

Instead of:
```python
# Adding yet another heuristic
if _is_weird_edge_case(text):
    return maybe_merge
```

Do:
```python
# Let the user decide
if confidence < 0.7:
    return registry.ask(prev_text, curr_text, callback=interactive_callback)
```

**Acceptance criteria:**
- [ ] Single unified interactive callback protocol
- [ ] `--teach` mode persists decisions
- [ ] At least 2 previously-hard-coded heuristics converted to interactive

---

## Phase 5: Documentation as Code (2 days)

**Goal:** The code explains itself.

### Task 5.1: Add `MergeDecision` return type

Every merge function returns explicit decision with reason:

```python
@dataclass
class MergeDecision:
    should_merge: bool
    reason: str
    pattern: Pattern | None = None
    confidence: float = 1.0
```

### Task 5.2: Generate decision documentation from code

```python
# Auto-generate docs/MERGE_DECISIONS.md from pattern registry
def generate_decision_docs():
    lines = ["# Merge Decision Reference\n"]
    for pattern in registry.patterns:
        lines.append(f"## {pattern.name}")
        lines.append(f"**Precedence:** {pattern.precedence.name}")
        lines.append(f"**Behavior:** {pattern.behavior.value}")
        lines.append(f"**Description:** {pattern.description}")
        lines.append(f"**Pattern:** `{pattern.match.pattern}`")
        lines.append("")
    return "\n".join(lines)
```

### Task 5.3: Add inline precedence documentation

```python
def _evaluate_merge(prev: Record, curr: Record, *, registry: PatternRegistry) -> MergeDecision:
    """Evaluate merge decision with explicit precedence.
    
    Order matters! Checks evaluated in this sequence:
    
    1. CRITICAL patterns (Q&A, numbered lists) → MERGE
    2. BOUNDARY patterns (chapter headings) → SPLIT  
    3. HIGH patterns (bullets with signals) → MERGE
    4. Continuation signals (And, But, However) → MERGE
    5. DEFAULT → SPLIT (preserve block boundaries)
    
    See docs/MERGE_DECISIONS.md for full reference.
    """
```

**Acceptance criteria:**
- [ ] All merge functions return MergeDecision with reason
- [ ] Auto-generated decision docs match actual code behavior
- [ ] Docstrings explain precedence order

---

## Success Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| `split_semantic.py` lines | 1,888 | < 300 | `wc -l` |
| Time to trace merge issue | ~2 hours | < 15 min | Manual timing |
| Pattern detection locations | 5+ files | 1 registry | `grep -r "re.compile"` |
| Documented merge precedence | 0% | 100% | Code review |
| Interactive mode coverage | footers, lists | all decisions | Feature audit |
| Test coverage for patterns | ~60% | > 90% | `pytest --cov` |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Regressions during decomposition | High | Run full suite after each move; golden tests |
| Pipe library learning curve | Low | Use only for internal transforms; keep API unchanged |
| Interactive mode fatigue | Medium | Good defaults; "always/never" options; `--teach` persistence |
| Pattern registry complexity | Medium | Start with existing patterns only; add new ones incrementally |

---

## Dependencies to Add

```toml
# pyproject.toml additions
"pipe>=2.0",  # Infix composition for internal pipelines
```

**Note:** No other new dependencies. We're simplifying by using existing tools better, not adding frameworks.

---

## Appendix: Libraries Considered But Rejected

| Library | Why Considered | Why Rejected |
|---------|----------------|--------------|
| `dry-python/classes` | Type-safe polymorphism | Too complex for our use case; mypy plugin overhead |
| `prefect` / `dagster` | Pipeline orchestration | Overkill; we have simple linear pipelines |
| `pypipelines` | Pipeline composition | Unmaintained (404) |
| `toolz` | Functional utilities | `funcy` already in deps; redundant |

---

## Related Documents

- [REFACTORING_ROADMAP.md](REFACTORING_ROADMAP.md) — Original analysis
- [emit_jsonl_refactoring_assessment.md](emit_jsonl_refactoring_assessment.md) — Successful prior refactoring
- [ARCHITECTURE.md](../ARCHITECTURE.md) — Module boundaries
- [AGENTS.md](../AGENTS.md) — Codebase stewardship contract
