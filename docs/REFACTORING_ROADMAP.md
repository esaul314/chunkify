# REFACTORING_ROADMAP.md — Structural Improvements for Maintainability

**Date:** 2026-01-26  
**Status:** PROPOSAL  
**Author:** System (following Q&A sequence merging debug session)

---

## Executive Summary

This document outlines structural refactoring opportunities identified during the Q&A sequence merging bug fix. The primary goal is to **reduce cognitive load** when debugging text transformation issues and to **make the code self-documenting** about which transformations have occurred.

### Key Problems Identified

1. **1,886-line `split_semantic.py`** — Far exceeds the 300-500 line guideline in ARCHITECTURE.md
2. **Implicit transformation state** — No way to know what merges/splits a text fragment has undergone
3. **Order-sensitive logic** — Q&A fix required moving code BEFORE heading checks; such dependencies are invisible
4. **Pattern detection scattered** — Similar patterns (Q&A sequences, numbered lists, bullets) handled in multiple places
5. **"Legacy" terminology confusion** — Documentation mentions "legacy" but refers to pre-pass module organization, not deprecated code

---

## 1. Legacy Code Clarification

### What "Legacy" Actually Means

The term "legacy" in our documentation refers to **architectural migration**, not deprecated code paths:

| Term in Docs | Actual Meaning |
|--------------|----------------|
| "Legacy Function" table in AGENTS.md | Maps old module locations → new pass locations |
| `_legacy_extract_text_blocks_from_pdf` | Backward-compatible shim (7 lines, wraps new code) |
| `_legacy_counts` | Helper to extract counts from new metrics format |
| `legacy_parity` pytest marker | Tests that compare old vs new pipeline output |

**Verdict:** There is no significant legacy code to remove. The "legacy" functions are thin shims for backward compatibility that delegate to the new implementation. Removing them would break external consumers without reducing complexity.

### Recommendation

- **Rename** `_legacy_extract_text_blocks_from_pdf` → `extract_text_blocks_from_pdf_compat`
- **Update AGENTS.md** to clarify that "Legacy-Aware Migration Rules" is a historical reference, not an indication of code to remove
- **Consider deprecation warnings** for compatibility shims if we want to phase them out

---

## 2. Transformation State Tracking (TextFragment Registry)

### The Problem

During the Q&A debugging session, we had to trace text through multiple passes to find where Q1 got separated from Q2. Each transformation is implicit—there's no audit trail.

### Proposed Solution: `TransformationLog`

A lightweight dataclass that travels with text through the pipeline:

```python
from dataclasses import dataclass, field
from typing import Literal

TransformKind = Literal[
    "extracted",      # Initial extraction from PDF/EPUB
    "cleaned",        # text_clean pass
    "merged",         # Cross-page merge, Q&A sequence merge, etc.
    "split",          # Chunk boundary split
    "heading_attach", # Heading attached to block
    "deduplicated",   # Duplicate sentence removed
]

@dataclass(frozen=True)
class TransformEntry:
    """Single transformation event."""
    kind: TransformKind
    pass_name: str
    reason: str
    source_hash: str  # Hash of input text
    result_hash: str  # Hash of output text
    
@dataclass
class TransformationLog:
    """Audit trail for a text fragment."""
    fragment_id: str  # Stable ID (hash of original extraction)
    entries: list[TransformEntry] = field(default_factory=list)
    
    def record(self, kind: TransformKind, pass_name: str, reason: str,
               source: str, result: str) -> None:
        self.entries.append(TransformEntry(
            kind=kind,
            pass_name=pass_name,
            reason=reason,
            source_hash=_short_hash(source),
            result_hash=_short_hash(result),
        ))
    
    def debug_view(self) -> str:
        """Human-readable transformation history."""
        return "\n".join(
            f"[{e.pass_name}] {e.kind}: {e.reason}"
            for e in self.entries
        )
```

### Integration Points

1. **Block extraction** (`pdf_parse`, `epub_parse`): Create initial `TransformationLog`
2. **Merges** (`_stitch_block_continuations`, `_merge_styled_list_records`): Record merge reason
3. **Splits** (`_soft_segments`, chunk boundaries): Record split reason
4. **Dedupe** (`emit_jsonl._dedupe`): Record dropped duplicates

### Benefits

- **Debugging**: `--trace` mode can dump full transformation history
- **Regression detection**: Compare transformation logs across versions
- **Self-documentation**: Each merge decision is explicitly justified

---

## 3. Pattern Registry with Interactive Learning

### The Problem

Pattern detection (Q&A sequences, numbered lists, bullet continuations) is scattered across modules with inconsistent handling. During the Q&A fix, we had to add a new pattern check in a specific location relative to other checks.

### Proposed Solution: `TextPatternRegistry`

A centralized registry where patterns are:
1. **Declared** with precedence and merge behavior
2. **Learnable** via interactive mode
3. **Testable** in isolation

```python
from dataclasses import dataclass
from enum import Enum
import re

class PatternPrecedence(Enum):
    """Determines which pattern wins when multiple match."""
    CRITICAL = 10   # Q&A sequences, numbered lists (always merge)
    HIGH = 20       # Bullet continuations with clear signals
    MEDIUM = 30     # Heading attachment
    LOW = 40        # Heuristic-based merges
    
class MergeBehavior(Enum):
    MERGE_ALWAYS = "merge"        # Always combine matching text
    MERGE_INTERACTIVE = "ask"     # Prompt user if ambiguous
    SPLIT_ALWAYS = "split"        # Always keep separate
    BOUNDARY = "boundary"         # Marks chunk boundary

@dataclass(frozen=True)
class TextPattern:
    """A learnable text pattern with merge behavior."""
    name: str
    pattern: re.Pattern[str]
    precedence: PatternPrecedence
    behavior: MergeBehavior
    description: str
    continuation_pattern: re.Pattern[str] | None = None  # What follows
    
    def matches(self, text: str) -> bool:
        return bool(self.pattern.search(text))
    
    def is_sequence_continuation(self, prev: str, curr: str) -> bool:
        if self.continuation_pattern is None:
            return False
        return (self.pattern.search(prev) is not None and
                self.continuation_pattern.match(curr.lstrip()) is not None)

# Pre-defined patterns
PATTERNS = [
    TextPattern(
        name="qa_sequence",
        pattern=re.compile(r"[QA]\d+:", re.IGNORECASE),
        precedence=PatternPrecedence.CRITICAL,
        behavior=MergeBehavior.MERGE_ALWAYS,
        description="Q&A interview format (Q1:, A1:, Q2:, etc.)",
        continuation_pattern=re.compile(r"^[QA]\d+:", re.IGNORECASE),
    ),
    TextPattern(
        name="numbered_list",
        pattern=re.compile(r"^\s*\d+[\.\)]\s"),
        precedence=PatternPrecedence.CRITICAL,
        behavior=MergeBehavior.MERGE_ALWAYS,
        description="Numbered list (1. Item, 2. Item)",
        continuation_pattern=re.compile(r"^\s*\d+[\.\)]\s"),
    ),
    TextPattern(
        name="bullet_list",
        pattern=re.compile(r"^\s*[•\-\*]\s"),
        precedence=PatternPrecedence.HIGH,
        behavior=MergeBehavior.MERGE_ALWAYS,
        description="Bullet list (• Item, - Item)",
    ),
    TextPattern(
        name="step_sequence",
        pattern=re.compile(r"^Step\s+\d+[:\.]", re.IGNORECASE),
        precedence=PatternPrecedence.CRITICAL,
        behavior=MergeBehavior.MERGE_ALWAYS,
        description="Step-by-step instructions (Step 1:, Step 2:)",
        continuation_pattern=re.compile(r"^Step\s+\d+[:\.]", re.IGNORECASE),
    ),
    TextPattern(
        name="chapter_heading",
        pattern=re.compile(r"^Chapter\s+\d+", re.IGNORECASE),
        precedence=PatternPrecedence.MEDIUM,
        behavior=MergeBehavior.BOUNDARY,
        description="Chapter heading (Chapter 1, Chapter 2)",
    ),
    TextPattern(
        name="lettered_points",
        pattern=re.compile(r"^\s*[a-zA-Z][\.\)]\s"),
        precedence=PatternPrecedence.HIGH,
        behavior=MergeBehavior.MERGE_ALWAYS,
        description="Lettered points (a. Item, b. Item)",
        continuation_pattern=re.compile(r"^\s*[a-zA-Z][\.\)]\s"),
    ),
    # Add more patterns as discovered...
]

class PatternRegistry:
    """Centralized pattern detection with learning capability."""
    
    def __init__(self, patterns: list[TextPattern] | None = None):
        self._patterns = sorted(
            patterns or PATTERNS,
            key=lambda p: p.precedence.value
        )
        self._learned: dict[str, MergeBehavior] = {}
    
    def detect(self, text: str) -> list[TextPattern]:
        """Return all patterns matching ``text``, sorted by precedence."""
        return [p for p in self._patterns if p.matches(text)]
    
    def should_merge(
        self,
        prev_text: str,
        curr_text: str,
        *,
        interactive_callback: callable | None = None,
    ) -> tuple[bool, str]:
        """Determine if texts should merge based on pattern matches.
        
        Returns (should_merge, reason).
        """
        for pattern in self._patterns:
            if pattern.is_sequence_continuation(prev_text, curr_text):
                if pattern.behavior == MergeBehavior.MERGE_ALWAYS:
                    return True, f"pattern:{pattern.name}"
                if pattern.behavior == MergeBehavior.MERGE_INTERACTIVE:
                    if interactive_callback:
                        decision = interactive_callback(
                            prev_text, curr_text, pattern
                        )
                        self._learned[pattern.name] = (
                            MergeBehavior.MERGE_ALWAYS if decision
                            else MergeBehavior.SPLIT_ALWAYS
                        )
                        return decision, f"interactive:{pattern.name}"
        return False, "no_pattern_match"
    
    def learn(self, pattern_name: str, behavior: MergeBehavior) -> None:
        """Record a learned behavior for a pattern."""
        self._learned[pattern_name] = behavior
```

### Interactive Learning Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Detected potential Q&A sequence across page boundary:       │
│                                                             │
│ Previous block ends with:                                   │
│   "...we're spread really thin.                            │
│   Q1: For this reason, many organizations..."              │
│                                                             │
│ Next block starts with:                                     │
│   "Q2: So why aren't organizations hiring?"                │
│                                                             │
│ This looks like a Q&A sequence (Q1: → Q2:).                │
│ Should these be kept together? [Y/n/always/never]          │
└─────────────────────────────────────────────────────────────┘

User enters: always
→ Pattern "qa_sequence" behavior set to MERGE_ALWAYS for this document
→ All subsequent Q&A sequences merge without prompting
```

### Benefits

- **Centralized logic**: All pattern checks in one place with clear precedence
- **Self-documenting**: Pattern descriptions explain what they match
- **Learnable**: Interactive mode teaches the system about document-specific patterns
- **Testable**: Each pattern can be unit-tested independently

---

## 4. Module Decomposition: `split_semantic.py`

### Current State

| Responsibility | Lines (est.) |
|----------------|--------------|
| Block stitching/merging | ~400 |
| Footer detection | ~200 |
| List handling | ~300 |
| Overlap management | ~200 |
| Segment emission | ~300 |
| Inline heading detection | ~150 |
| Utility functions | ~336 |
| **Total** | **1,886** |

### Proposed Structure

```
pdf_chunker/passes/
├── split_semantic.py           # Main pass (< 300 lines)
│   ├── __call__                # Entry point
│   └── _split_pass             # Orchestration
│
├── split_semantic/             # Subpackage
│   ├── __init__.py             # Public exports
│   ├── stitching.py            # Block stitching (~400 lines)
│   │   ├── _stitch_block_continuations
│   │   ├── _is_heading
│   │   ├── _starts_list_like
│   │   └── ...
│   ├── footers.py              # Footer detection (~200 lines)
│   │   ├── _resolve_footer_suffix
│   │   ├── _is_footer_artifact_record
│   │   └── ...
│   ├── overlap.py              # Overlap management (~200 lines)
│   │   ├── _restore_overlap_words
│   │   ├── _trim_boundary_overlap
│   │   └── ...
│   ├── emission.py             # Segment emission (~300 lines)
│   │   ├── _emit_buffer_segments
│   │   ├── _emit_individual_records
│   │   └── ...
│   └── patterns.py             # Pattern registry (NEW)
│       └── PatternRegistry
```

### Migration Strategy

1. **Create subpackage** with `__init__.py` re-exporting all public symbols
2. **Move functions one group at a time**, updating imports
3. **Run tests after each move** to catch regressions
4. **Update main `split_semantic.py`** to import from subpackage

---

## 5. Order-Sensitive Logic Documentation

### The Problem

The Q&A fix required moving the Q&A check BEFORE heading checks. This ordering dependency is invisible without reading the code.

### Proposed Solution: Explicit Decision Chain

```python
@dataclass
class MergeDecision:
    """Result of evaluating whether to merge blocks."""
    should_merge: bool
    reason: str
    pattern: TextPattern | None = None
    
def _evaluate_merge(
    prev: tuple[int, Block, str],
    curr: tuple[int, Block, str],
    *,
    registry: PatternRegistry,
) -> MergeDecision:
    """Evaluate merge decision with explicit precedence.
    
    Order matters! Checks are evaluated in this sequence:
    
    1. PATTERN SEQUENCES (highest priority)
       - Q&A sequences (Q1: → Q2:)
       - Numbered lists (1. → 2.)
       - Step sequences (Step 1: → Step 2:)
       → If matched: MERGE regardless of other factors
    
    2. HEADING BOUNDARIES
       - Current block is heading type
       - Previous block is heading type
       → If matched: SPLIT (don't merge into/out of headings)
    
    3. LIST BOUNDARIES
       - Current block starts list
       - Previous block doesn't invite continuation
       → If matched: SPLIT
    
    4. CONTINUATION SIGNALS (lowest priority)
       - Sentence continuation words (And, But, However...)
       - Cross-page sentence completion
       → If matched: MERGE with context
    
    5. DEFAULT: SPLIT (preserve block boundaries)
    """
    prev_text = prev[2]
    curr_text = curr[2]
    curr_block = curr[1]
    prev_block = prev[1]
    
    # 1. Pattern sequences (highest priority)
    should_merge, reason = registry.should_merge(prev_text, curr_text)
    if should_merge:
        return MergeDecision(True, reason)
    
    # 2. Heading boundaries
    if _is_heading(curr_block):
        return MergeDecision(False, "curr_is_heading")
    if _is_heading(prev_block):
        return MergeDecision(False, "prev_is_heading")
    
    # 3. List boundaries
    if _starts_list_like(curr_block, curr_text):
        return MergeDecision(False, "curr_starts_list")
    
    # 4. Continuation signals
    lead = curr_text.lstrip()
    if lead and _is_continuation_lead(lead):
        return MergeDecision(True, "continuation_word")
    
    # 5. Default: split
    return MergeDecision(False, "default_boundary")
```

### Benefits

- **Self-documenting**: Docstring explains the exact order and why
- **Auditable**: Every decision returns a reason
- **Testable**: Can unit test each branch independently

---

## 6. Implementation Phases

### Phase 1: Foundation (Low Risk)
- [ ] Create `TransformationLog` dataclass
- [ ] Add to Block/Chunk metadata as optional field
- [ ] Wire into `--trace` mode for debugging

### Phase 2: Pattern Registry (Medium Risk)
- [ ] Create `TextPatternRegistry` with pre-defined patterns
- [ ] Refactor `_is_qa_sequence_continuation` to use registry
- [ ] Add interactive learning capability to CLI

### Phase 3: Module Decomposition (Medium Risk)
- [ ] Create `split_semantic/` subpackage
- [ ] Move footer detection functions
- [ ] Move overlap management functions
- [ ] Move segment emission functions
- [ ] Move stitching functions last (most interconnected)

### Phase 4: Order Documentation (Low Risk)
- [ ] Add `MergeDecision` return type to stitching
- [ ] Document precedence in function docstrings
- [ ] Add integration tests for precedence behavior

---

## 7. Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| `split_semantic.py` lines | 1,886 | < 300 |
| Time to trace a merge issue | ~2 hours | < 15 minutes |
| Pattern detection locations | 5+ modules | 1 registry |
| Documented merge precedence | 0 | 100% |

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Regressions during decomposition | High | Run full test suite after each move |
| Performance overhead from logging | Low | Make TransformationLog optional |
| Pattern registry complexity | Medium | Start with existing patterns only |
| Interactive mode UX | Medium | Design clear prompts, support "always/never" |

---

## Appendix: Commonly Needed Patterns

Based on debugging sessions and document analysis, these patterns appear frequently:

1. **Q&A Sequences**: `Q1:`, `Q2:`, `A1:`, `A2:`
2. **Numbered Lists**: `1.`, `2.`, `(1)`, `(2)`
3. **Bullet Lists**: `•`, `-`, `*`
4. **Step Instructions**: `Step 1:`, `Step 2:`
5. **Lettered Points**: `a.`, `b.`, `A)`, `B)`
6. **Chapter Headings**: `Chapter 1`, `CHAPTER ONE`
7. **Section Markers**: `Section 1.2`, `§1`
8. **Figure/Table References**: `Figure 1:`, `Table 2:`
9. **Dialogue Tags**: `Alice:`, `Bob:`
10. **Footnote Markers**: `¹`, `²`, `[1]`, `[2]`
11. **Continuation Ellipsis**: `...continued`
12. **Part Markers**: `Part I`, `PART ONE`

---

## Related Documents

- [ARCHITECTURE.md](../ARCHITECTURE.md) — Module boundaries and patterns
- [CODESTYLE.md](../CODESTYLE.md) — Code style guidelines
- [emit_jsonl_refactoring_assessment.md](emit_jsonl_refactoring_assessment.md) — Previous successful refactoring
- [merge_strategy_design.md](merge_strategy_design.md) — Merge function consolidation
