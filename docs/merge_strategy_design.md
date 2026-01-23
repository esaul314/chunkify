# Merge Strategy Consolidation Design

## Current State

`emit_jsonl.py` has three overlapping merge functions that share similar logic
but operate at different granularities:

| Function | Input Type | Primary Goal | Lines |
|----------|-----------|--------------|-------|
| `_merge_incomplete_lists` | `list[Row]` | Merge list intros with continuations | ~75 |
| `_merge_very_short_forward` | `list[dict]` | Merge short items/orphaned bullets | ~110 |
| `_merge_short_rows` | `list[Row]` | Catch-all for remaining short fragments | ~115 |

### Common Patterns

All three functions share:
1. **Iterator pattern**: Loop through items with index tracking
2. **Pending accumulator**: Hold items that need merging
3. **Size guard**: Check `_max_merge_chars()` before merging
4. **Forward-first logic**: Try forward merge, fall back to backward
5. **Coherence check**: Preserve coherent items even if short

### Key Differences

| Aspect | `_merge_incomplete_lists` | `_merge_very_short_forward` | `_merge_short_rows` |
|--------|---------------------------|-----------------------------|--------------------|
| Trigger predicate | `_has_incomplete_list` or `words < min_words` | `words < threshold` or orphan bullet | `words < min_words` and not coherent |
| Preserve condition | None explicit | `_coherent` + `>= coherent_min` words | `_coherent` + `>= 8` words |
| Merge text join | `"\n\n"` | `_merge_text()` (handles bullets) | `"\n\n"` |
| Critical force-merge | No | No | Yes (`< 5` words always merges) |
| Strategy param | No | Yes (`BulletHeuristicStrategy`) | No |

### Call Order

In `_rows()`:
1. `_merge_very_short_forward(items)` — first pass on items
2. Items → split into rows → deduplication
3. `_merge_short_rows(rows)` — final pass on rows
4. Inside `_merge_short_rows`: calls `_merge_incomplete_lists(rows)` as first sub-pass

## Design Goals

1. **Reduce duplication**: Single merge loop with pluggable predicates
2. **Preserve behavior**: Pass characterization tests unchanged
3. **Improve testability**: Named predicates for each merge condition
4. **Simplify debugging**: Single point for merge decision logging

## Proposed Architecture

### Option A: Predicate Composition (Recommended)

Extract merge decisions into a `MergePredicate` protocol:

```python
from typing import Protocol

class MergePredicate(Protocol):
    """Predicate that determines if an item should merge."""
    
    def should_hold(self, item: dict, words: int, chars: int) -> tuple[bool, str]:
        """Return (should_hold_for_merge, reason)."""
        ...
    
    def should_preserve(self, item: dict, words: int, chars: int) -> bool:
        """Return True if item should be preserved even when short."""
        ...
```

Concrete predicates:
- `IncompletListPredicate`: Uses `_has_incomplete_list`
- `VeryShortPredicate`: Uses word threshold + orphan bullet check
- `ShortRowPredicate`: Uses min words + coherence check

Generic merge function:

```python
def _merge_with_predicate(
    items: list[dict],
    predicate: MergePredicate,
    *,
    max_chars: int,
    join_sep: str = "\n\n",
    force_critical: bool = False,
    critical_threshold: int = 5,
) -> list[dict]:
    """Single-pass merge loop with pluggable predicate."""
```

**Pros**:
- Clean separation of concerns
- Easy to add new merge strategies
- Testable predicates in isolation

**Cons**:
- Adds protocol abstraction
- May be overengineered for 3 use cases

### Option B: Unified Merge with Flags (Simpler)

Single function with behavioral flags:

```python
def _merge_items(
    items: list[dict],
    *,
    check_incomplete_list: bool = False,
    check_orphan_bullet: bool = False,
    word_threshold: int = 15,
    coherent_min_words: int = 8,
    force_critical: bool = False,
    strategy: BulletHeuristicStrategy | None = None,
) -> list[dict]:
    """Unified merge with configurable behavior."""
```

Call sites:
```python
# Replace _merge_incomplete_lists
_merge_items(rows, check_incomplete_list=True)

# Replace _merge_very_short_forward  
_merge_items(items, check_orphan_bullet=True, word_threshold=threshold)

# Replace _merge_short_rows
_merge_items(rows, force_critical=True, coherent_min_words=8)
```

**Pros**:
- No new abstractions
- Easier incremental migration
- Matches current implementation style

**Cons**:
- Flag proliferation could become unwieldy
- Less composable for future extensions

### Recommendation

**Option B (Unified Merge with Flags)** aligns better with the anti-over-engineering
mandate. The three merge functions have enough overlap that a single parameterized
function can handle all cases without introducing new protocols or abstractions.

## Migration Plan

### Phase 1: Extract core loop (this task)

1. Create `_merge_items_core()` implementing the shared loop pattern
2. Keep existing functions as thin wrappers calling the core
3. Verify all characterization tests pass

### Phase 2: Consolidate wrappers

1. Inline the thin wrappers at call sites
2. Remove duplicate code
3. Update logging to use consistent format

### Phase 3: Clean up (optional)

1. Consider removing `_merge_incomplete_lists` since it's only called from `_merge_short_rows`
2. Evaluate if `_merge_very_short_forward` and `_merge_short_rows` can be a single pass

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Behavior change | Medium | High | Characterization tests catch regressions |
| Performance regression | Low | Low | Same algorithmic complexity |
| Debugging harder | Low | Medium | Unified logging actually helps |

## Acceptance Criteria

- [ ] Single `_merge_items_core()` function exists
- [ ] All 50 merge tests pass
- [ ] No change in behavior (same outputs for same inputs)
- [ ] Logging shows merge reason and size info
- [ ] Code lines reduced by ~50-100 lines total

## Implementation Notes

The core loop structure should be:

```python
def _merge_items_core(
    items: list[dict],
    *,
    should_hold: Callable[[dict, int, int], tuple[bool, str]],
    should_preserve: Callable[[dict, int, int], bool],
    max_chars: int,
    join_sep: str = "\n\n",
    force_critical: bool = False,
    critical_threshold: int = 5,
    merge_fn: Callable[[str, str], str] | None = None,
) -> list[dict]:
    """Core merge loop with pluggable predicates."""
    if not items:
        return items
    
    result: list[dict] = []
    pending: dict | None = None
    pending_reason: str = ""
    
    for item in items:
        text = item.get("text", "")
        words = _word_count(text)
        chars = len(text)
        
        # Try to merge pending item
        if pending is not None:
            merged = _try_merge(pending, item, max_chars, join_sep, merge_fn)
            if merged is not None:
                item = merged
                pending = None
                # Recalculate
                text = item.get("text", "")
                words = _word_count(text)
                chars = len(text)
            else:
                result.append(pending)
                pending = None
        
        # Check if item should be held
        if should_preserve(item, words, chars):
            result.append(item)
        else:
            hold, reason = should_hold(item, words, chars)
            if hold:
                pending = item
                pending_reason = reason
            else:
                result.append(item)
    
    # Handle trailing pending
    if pending is not None:
        # Try backward merge or keep as-is
        ...
    
    return result
```

The `should_hold` and `should_preserve` callables encode the merge-specific
logic without requiring a formal Protocol.
