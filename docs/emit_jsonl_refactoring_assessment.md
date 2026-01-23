# emit_jsonl.py Refactoring Assessment

**Date:** 2026-01-23  
**File:** `pdf_chunker/passes/emit_jsonl.py`  
**Lines:** 1,483  
**Functions:** 63  
**Classes:** 1

---

## Executive Summary

**Is refactoring necessary?** Yes.  
**Is significant improvement possible?** Yes, with careful incremental work.

The `emit_jsonl.py` module has grown organically to handle numerous edge cases in text chunking, deduplication, and list handling. While the code works, it has accumulated significant complexity that makes debugging and modification challenging. The recent debugging session (fixing short chunks with `--interactive` flag) required tracing through multiple functions with overlapping responsibilities, identifying a shadowed function definition, and understanding implicit state passed through nested closures.

---

## Complexity Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Total lines | 1,483 | ⚠️ Very large for a single module |
| Functions | 63 | ⚠️ High; many are tightly coupled |
| Nested `step` functions | 3 | ⚠️ Pattern repeated; hard to test |
| Top function cyclomatic complexity | 13+ conditionals | 🔴 Hard to reason about |
| Environment variables | 8+ | ⚠️ Hidden configuration surface |

### Most Complex Functions

| Function | Lines | Conditionals | Loops | Assessment |
|----------|-------|--------------|-------|------------|
| `_reserve_for_list` | 70 | 13 | 3 | 🔴 Critical complexity |
| `_has_incomplete_list` | 61 | 11 | 3 | ⚠️ Multiple detection modes |
| `_split` | 80 | 11 | 1 | 🔴 Uses nested closure `step` |
| `_merge_very_short_forward` | 88 | 8 | 1 | ⚠️ Overlaps with similar functions |
| `_merge_short_rows` | 88 | 7 | 1 | ⚠️ Overlaps with above |
| `_merge_incomplete_lists` | 66 | 7 | 1 | ⚠️ Recently added, similar purpose |

---

## Key Challenges Identified

### 1. **Overlapping Merge Functions**

Three functions handle "merging short/incomplete content":
- `_merge_very_short_forward` (line 1002) - merges items below threshold
- `_merge_incomplete_lists` (line 606) - merges rows with incomplete lists  
- `_merge_short_rows` (line 1319) - final pass merging short rows

**Problem:** Unclear responsibilities. The recent bug was caused by:
1. Adding `_merge_incomplete_lists` 
2. But `_merge_short_rows` calls it, while `_rows` calls `_merge_very_short_forward` separately
3. Both functions have similar logic but operate on different data structures (`Row` vs `dict`)

**Symptom:** Debugging required understanding which function runs when, in what order, and with what data shape.

### 2. **Nested `step` Closures**

Three functions use internal `step` closures with `itertools.accumulate`:
- `_split` (line 442)
- `_merge_sentence_pieces` (line 821)  
- `_dedupe` (line 1131)

**Problem:** 
- Cannot unit test `step` in isolation
- State is implicitly threaded through tuple accumulation
- Hard to add logging/tracing inside the closure
- The pattern is clever but non-obvious

**Symptom:** During debugging, adding print statements inside `step` required understanding the accumulate pattern and tuple unpacking.

### 3. **Implicit Configuration via Environment Variables**

The module reads 8+ environment variables:
- `PDF_CHUNKER_JSONL_MIN_WORDS`
- `PDF_CHUNKER_JSONL_META_KEY`
- `PDF_CHUNKER_JSONL_MAX_CHARS`
- `PDF_CHUNKER_TARGET_CHUNK_CHARS`
- `PDF_CHUNKER_VERY_SHORT_THRESHOLD`
- `PDF_CHUNKER_MAX_MERGE_CHARS`
- `PDF_CHUNKER_MIN_ROW_WORDS`
- `PDF_CHUNKER_DEDUP_DEBUG`

**Problem:** 
- Functions like `_target_chunk_chars()` read env vars on every call
- Thresholds are scattered across the file
- Testing requires setting multiple env vars or mocking
- No single place to see all configuration

### 4. **"Detection" vs "Action" Coupling**

Functions like `_has_incomplete_list` both:
1. Detect a condition (returns bool)
2. Encode complex heuristics about what "incomplete" means

**Problem:** The detection logic has grown to 61 lines with 11 conditionals covering:
- Colon-only endings
- Inline list starts
- Single bullet items
- Missing sentence terminators

When detection fails, it's hard to know which heuristic branch was relevant.

### 5. **Multiple Data Shapes**

The module operates on:
- `Row = dict[str, Any]` - JSONL row with `text` and optional `metadata`
- `dict[str, Any]` - item from upstream passes with `text` and `meta`
- Plain `str` - text content during splitting

**Problem:** Functions accept different shapes but the type hints don't always distinguish them. `_merge_very_short_forward` takes `list[dict[str, Any]]` while `_merge_incomplete_lists` takes `list[Row]` - but `Row` is just an alias for `dict[str, Any]`.

### 6. **Call Graph Complexity**

```
_rows (entry point)
  ├── _rows_from_item
  │     └── _split
  │           └── _reserve_for_list
  ├── _merge_very_short_forward
  │     └── _coherent
  │           └── _has_incomplete_list
  ├── _coalesce
  │     └── _merge_items
  │           └── _should_merge
  ├── _dedupe
  └── _merge_short_rows
        ├── _merge_incomplete_lists
        │     └── _has_incomplete_list
        └── _coherent
              └── _has_incomplete_list
```

**Problem:** `_has_incomplete_list` is called from 3 different code paths with different contexts. A change to its logic can have unexpected effects.

---

## Proposed Refactoring Avenues

### Avenue 1: Extract Configuration Object (Low Risk, High Value)

**Current:**
```python
def _target_chunk_chars() -> int:
    return int(os.getenv("PDF_CHUNKER_TARGET_CHUNK_CHARS", "2000"))
```

**Proposed:**
```python
@dataclass(frozen=True)
class EmitConfig:
    target_chunk_chars: int = 2000
    max_chars: int = 8000
    min_row_words: int = 15
    very_short_threshold: int = 30
    max_merge_chars: int = 2000
    # ... all thresholds in one place

    @classmethod
    def from_env(cls) -> "EmitConfig":
        return cls(
            target_chunk_chars=int(os.getenv("PDF_CHUNKER_TARGET_CHUNK_CHARS", "2000")),
            # ...
        )
```

**Benefits:**
- All configuration visible in one place
- Easy to test with different configs
- No repeated env var reads
- Type-safe

### Avenue 2: Consolidate Merge Functions (Medium Risk, High Value)

Replace the three merge functions with a single `MergeStrategy` approach:

```python
class MergeDecision(Enum):
    KEEP_SEPARATE = auto()
    MERGE_FORWARD = auto()
    MERGE_BACKWARD = auto()

def should_merge(current: Row, next_row: Row | None, prev_row: Row | None, config: EmitConfig) -> MergeDecision:
    """Single decision point for all merge logic."""
    # Consolidate all heuristics here
```

**Benefits:**
- One place to debug merge decisions
- Can log the decision and why
- Easier to add new merge rules

### Avenue 3: Replace `step` Closures with Explicit State Machines (Medium Risk, Medium Value)

**Current:**
```python
def _split(text: str, limit: int) -> list[str]:
    def step(state, _):
        pieces, remaining, intro_hint = state
        # ... 60 lines of logic
        return pieces, rest.lstrip(), next_intro

    states = accumulate(repeat(None), step, initial=([], text, None))
    return next(p for p, r, _ in states if not r)
```

**Proposed:**
```python
@dataclass
class SplitState:
    pieces: list[str]
    remaining: str
    intro_hint: str | None
    
    def is_complete(self) -> bool:
        return not self.remaining

class TextSplitter:
    def __init__(self, limit: int, is_list_line: Callable):
        self.limit = limit
        self.is_list_line = is_list_line
    
    def split(self, text: str) -> list[str]:
        state = SplitState([], text, None)
        while not state.is_complete():
            state = self._step(state)
        return state.pieces
    
    def _step(self, state: SplitState) -> SplitState:
        # Logic here - can be unit tested!
```

**Benefits:**
- State is explicit and inspectable
- `_step` can be unit tested directly
- Easy to add logging/tracing
- Debugger-friendly

### Avenue 4: Decompose Detection Functions (Low Risk, Medium Value)

Break `_has_incomplete_list` into named predicates:

```python
def _ends_with_list_intro_colon(text: str) -> bool:
    """Text ends with ':' and has no bullets - pure list introduction."""
    ...

def _has_single_inline_bullet(text: str) -> bool:
    """Text has colon followed by single bullet item inline."""
    ...

def _has_incomplete_list(text: str) -> bool:
    """Composite check for any incomplete list pattern."""
    return (
        _ends_with_list_intro_colon(text)
        or _has_single_inline_bullet(text)
        or _has_unterminated_bullet_item(text)
    )
```

**Benefits:**
- Each predicate is testable and named
- Easy to see which check matched
- Can add logging per predicate

### Avenue 5: Extract List Handling to Submodule (Medium Risk, High Value)

Move all list-related logic to `pdf_chunker/passes/emit_jsonl_lists.py`:
- `_is_list_line`
- `_reserve_for_list`
- `_collapse_list_gaps`
- `_rebalance_lists`
- `_has_incomplete_list`
- `_starts_with_orphan_bullet`

**Benefits:**
- Reduces main file by ~300 lines
- List logic is cohesive and isolated
- Easier to test list handling separately

---

## Recommended Refactoring Order

1. **Phase 1 (Safe, Immediate):**
   - Extract `EmitConfig` dataclass
   - Add logging to merge decision points
   - Write characterization tests for current behavior

2. **Phase 2 (Medium Risk):**
   - Consolidate merge functions into unified strategy
   - Decompose detection functions into named predicates

3. **Phase 3 (Higher Risk):**
   - Replace `step` closures with explicit state machines
   - Extract list handling to submodule

---

## Testing Recommendations

Before refactoring:
1. **Increase test coverage** on merge functions - currently only `emit_jsonl_coalesce_test.py` covers this
2. **Add property-based tests** for split/merge invariants (e.g., merged text should contain all original content)
3. **Add characterization tests** that capture current behavior on known PDFs

During refactoring:
1. Use **snapshot testing** to detect behavior changes
2. Run **full pipeline tests** after each phase
3. Use the `--trace` flag on known problematic PDFs to verify behavior

---

## Risks of Not Refactoring

1. **Debugging time will increase** - each bug requires understanding the full 1,483 lines
2. **New features will add more complexity** - the merge/split logic is already fragile
3. **Onboarding new contributors is difficult** - no clear entry points
4. **Test coverage will remain low** - nested closures resist testing

---

## Conclusion

The `emit_jsonl.py` module is a good candidate for refactoring. The complexity is not inherent to the problem domain but rather accumulated through organic growth. The proposed avenues offer incremental improvements that can be applied without rewriting the entire module.

**Recommended first step:** Extract `EmitConfig` and add logging to merge decisions. This provides immediate debugging value with minimal risk.
