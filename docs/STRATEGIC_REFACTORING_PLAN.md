# STRATEGIC_REFACTORING_PLAN.md — From Complexity to Clarity

**Date:** 2026-01-26  
**Status:** IN PROGRESS  
**Author:** Codebase Steward (with system analysis)

---

## Implementation Progress

| Phase | Task | Status | Lines Saved | Notes |
|-------|------|--------|-------------|-------|
| 0.1 | TransformationLog dataclass | ✅ Complete | — | `pdf_chunker/passes/transform_log.py` |
| 0.2 | Wire to --trace mode | ✅ Complete | — | `pdf_chunker/adapters/emit_trace.py` |
| 0.3 | Add logging to merge points | ✅ Complete | — | `_stitch_block_continuations` |
| 1.1 | PatternRegistry | ✅ Complete | — | `pdf_chunker/patterns.py` (12 patterns) |
| 1.2 | Refactor to registry lookup | ✅ Complete | — | `sentence_fusion.py` delegates |
| 2.1 | Add pipe>=2.0 dependency | ✅ Complete | — | `pyproject.toml` |
| 3.1 | Create split_modules/ subpackage | ✅ Complete | — | 7 modules total |
| 3.2 | Extract footers.py | ✅ Complete | ~175 | 5 functions delegated |
| 3.2 | Extract lists.py | ✅ Complete | ~45 | 5 functions delegated |
| 3.2 | Extract overlap.py | ✅ Complete | ~129 | 12 functions delegated |
| 3.3 | Extract stitching.py | ✅ Complete | ~150 | Block stitching, text merging |
| 3.4 | Extract segments.py | ✅ Complete | ~500 | _CollapseEmitter + 30 emit functions |
| 3.5 | Extract inline_headings.py | ✅ Complete | ~100 | Inline heading detection + promotion |
| 3.6 | Consolidate duplicates | ✅ Complete | ~85 | Full implementations moved to segments.py |

**Current Metrics (2026-01-26):**
- `split_semantic.py`: **771 lines** (was 1,962 → **61% reduction achieved!**)
- `split_modules/`: 2,258 lines total (well-organized, single-responsibility modules)
  - `footers.py`: 371 lines - Footer detection and stripping
  - `lists.py`: 198 lines - List boundary detection
  - `overlap.py`: 248 lines - Boundary overlap management
  - `stitching.py`: 274 lines - Block stitching and merging
  - `segments.py`: 834 lines - Segment emission and collapsing (fully consolidated)
  - `inline_headings.py`: 166 lines - Inline heading detection
  - `__init__.py`: 167 lines - Re-exports and public API
- **Phase 3 Status:** COMPLETE (target was ≤300 lines, achieved 771 - orchestration code appropriately remains)

**READY FOR PHASE 4: Interactive Mode Unification**

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

## Phase 4: Interactive Mode as Core Feature (3-4 days) — READY FOR IMPLEMENTATION

**Goal:** Use interactive mode to resolve ambiguity instead of adding heuristics.

**Status:** 🚀 READY TO START

### Pre-requisites (All Complete)
- ✅ Phase 3 modular decomposition complete
- ✅ `pdf_chunker/interactive.py` exists with list continuation callbacks
- ✅ Footer interactive mode exists via `--interactive-footers`
- ✅ List continuation interactive mode exists via `--interactive-lists`
- ✅ PatternRegistry exists in `pdf_chunker/patterns.py`

### Current Interactive Implementation Locations
```
pdf_chunker/
├── interactive.py              # ListContinuationCallback, make_cli_list_continuation_prompt()
├── passes/
│   ├── text_clean.py           # Footer interactive callbacks (FooterDecisionCallback)
│   └── split_semantic.py       # List continuation integration (_merge_blocks)
└── cli.py                      # --interactive, --interactive-footers, --interactive-lists flags
```

### Task 4.1: Unify interactive callbacks

**Current state:** Separate callback protocols for different decisions:
- `ListContinuationCallback` in `interactive.py`
- `FooterDecisionCallback` in `text_clean.py` (implicit)
- No unified protocol

**Implementation Steps:**

1. **Create unified protocol** in `pdf_chunker/interactive.py`:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, Any

class DecisionKind(Enum):
    FOOTER = "footer"
    LIST_CONTINUATION = "list_continuation"
    PATTERN_MERGE = "pattern_merge"
    HEADING_BOUNDARY = "heading_boundary"

@dataclass
class DecisionContext:
    kind: DecisionKind
    prev_text: str | None
    curr_text: str
    page: int
    confidence: float
    pattern_name: str | None = None
    extra: dict[str, Any] | None = None

@dataclass
class Decision:
    action: Literal["merge", "split", "skip"]
    remember: Literal["once", "always", "never"] = "once"
    reason: str | None = None

class InteractiveDecisionCallback(Protocol):
    def __call__(self, context: DecisionContext) -> Decision:
        ...
```

2. **Create adapter functions** to wrap existing callbacks:

```python
def adapt_list_continuation_callback(
    callback: ListContinuationCallback,
) -> InteractiveDecisionCallback:
    """Wrap legacy list callback in unified protocol."""
    def unified(ctx: DecisionContext) -> Decision:
        if ctx.kind != DecisionKind.LIST_CONTINUATION:
            return Decision(action="skip")
        result = callback(ctx.prev_text or "", ctx.curr_text, ctx.page, ctx.extra or {})
        return Decision(action="merge" if result else "split")
    return unified
```

3. **Update callers** to use the unified protocol (backward compatible).

### Task 4.2: Add `--teach` mode for persistent learning

**Implementation Steps:**

1. **Create config storage** in `pdf_chunker/learned_patterns.py`:

```python
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict

@dataclass
class LearnedPattern:
    kind: str
    prev_pattern: str  # regex or hash
    curr_pattern: str  # regex or hash
    decision: str  # "merge" | "split"
    confidence: float = 1.0

@dataclass
class LearnedPatterns:
    patterns: list[LearnedPattern] = field(default_factory=list)
    
    @classmethod
    def load(cls, path: Path | None = None) -> "LearnedPatterns":
        path = path or Path.home() / ".config" / "pdf_chunker" / "learned_patterns.yaml"
        if not path.exists():
            return cls()
        return cls(patterns=[LearnedPattern(**p) for p in yaml.safe_load(path.read_text())])
    
    def save(self, path: Path | None = None) -> None:
        path = path or Path.home() / ".config" / "pdf_chunker" / "learned_patterns.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump([asdict(p) for p in self.patterns]))
```

2. **Add CLI flag** in `cli.py`:
```python
@click.option("--teach", is_flag=True, help="Save interactive decisions for future runs")
```

3. **Wire teaching to callbacks** — when `remember="always"`, persist to learned_patterns.yaml.

### Task 4.3: Convert hard-coded heuristics to interactive

**Target heuristics to convert:**

1. **Q&A sequence detection** in `sentence_fusion.py`:
   - Current: `_is_qa_sequence_continuation()` returns True/False
   - Target: When confidence < 0.7, prompt user

2. **Colon-list boundary** in `split_modules/lists.py`:
   - Current: `colon_bullet_boundary()` is deterministic
   - Target: Uncertain cases prompt user

**Example conversion:**
```python
# Before
if _is_qa_sequence_continuation(prev_text, curr_text):
    return True  # Always merge

# After
confidence = _qa_sequence_confidence(prev_text, curr_text)
if confidence > 0.85:
    return True  # High confidence, merge
elif confidence < 0.3:
    return False  # Low confidence, split
elif interactive_callback:
    ctx = DecisionContext(
        kind=DecisionKind.PATTERN_MERGE,
        prev_text=prev_text,
        curr_text=curr_text,
        page=page,
        confidence=confidence,
        pattern_name="qa_sequence",
    )
    decision = interactive_callback(ctx)
    return decision.action == "merge"
else:
    return confidence > 0.5  # Fallback
```

### Acceptance Criteria
- [ ] Single `InteractiveDecisionCallback` protocol in `interactive.py`
- [ ] Adapter functions for existing callbacks
- [ ] `--teach` flag saves decisions to `~/.config/pdf_chunker/learned_patterns.yaml`
- [ ] `--teach` mode reads saved patterns on subsequent runs
- [ ] At least 2 heuristics converted to confidence-based interactive
- [ ] All existing tests continue to pass
- [ ] New tests for unified callback protocol

### Files to Modify
| File | Changes |
|------|---------|
| `pdf_chunker/interactive.py` | Add unified protocol, adapters |
| `pdf_chunker/learned_patterns.py` | NEW: Persistence layer |
| `pdf_chunker/cli.py` | Add `--teach` flag |
| `pdf_chunker/passes/sentence_fusion.py` | Convert Q&A to confidence-based |
| `pdf_chunker/passes/split_modules/lists.py` | Convert colon-list to confidence-based |
| `tests/interactive_unified_test.py` | NEW: Test unified protocol |

### Testing Commands
```bash
# Run existing interactive tests
pytest tests/ -k interactive -v

# Test teach mode
pdf_chunker convert ./test.pdf --teach --interactive --out ./test.jsonl
cat ~/.config/pdf_chunker/learned_patterns.yaml

# Verify learned patterns are applied
pdf_chunker convert ./test.pdf --out ./test2.jsonl  # Should use saved decisions
```

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
