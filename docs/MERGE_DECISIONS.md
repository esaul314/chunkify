# Merge Decision Reference

> **Auto-generated:** 2026-01-26  
> **Source:** `pdf_chunker/patterns.py`  
> **Regenerate:** `python scripts/generate_merge_docs.py`

This document describes how pdf_chunker decides whether to merge adjacent
text blocks. These decisions are critical for maintaining semantic coherence
in chunked output.

---

## Quick Reference

| Pattern | Precedence | Behavior | Description |
|---------|------------|----------|-------------|
| `qa_sequence` | CRITICAL | ✅ Merge | Q&A interview format (Q1:, A1:, Q2:, etc.) |
| `numbered_list` | CRITICAL | ✅ Merge | Numbered list items (1. Item, 2. Item) |
| `step_sequence` | CRITICAL | ✅ Merge | Step-by-step instructions (Step 1:, Step 2:) |
| `lettered_list` | CRITICAL | ✅ Merge | Lettered list items (a. Item, b. Item) |
| `bullet_list` | HIGH | ✅ Merge | Bullet list items (• Item, - Item, * Item) |
| `dialogue_tag` | HIGH | ❓ Ask | Dialogue or speaker tag (Alice:, Bob:) |
| `figure_reference` | MEDIUM | ❌ Split | Figure or table reference (Figure 1:, Table 2) |
| `footnote_marker` | MEDIUM | ❓ Ask | Possible footnote marker ([1], (2), 3 ) |
| `continuation_word` | LOW | ✅ Merge | Sentence continuation word (And, But, However...) |
| `chapter_heading` | BOUNDARY | 🚧 Boundary | Chapter heading (Chapter 1, Chapter 2) |
| `part_marker` | BOUNDARY | 🚧 Boundary | Part marker (Part I, Part One) |
| `section_marker` | BOUNDARY | ❌ Split | Section marker (Section 1, §1) |

---

## Precedence Order

Patterns are evaluated in precedence order. When multiple patterns match,
the one with the **lowest precedence number** wins:

| Precedence | Value | Description |
|------------|-------|-------------|
| CRITICAL | 10 | Q&A sequences, numbered lists — always merge |
| HIGH | 20 | Bullet continuations with clear signals |
| MEDIUM | 30 | Context-dependent (figure refs, footnotes) |
| LOW | 40 | Heuristic-based (continuation words) |
| BOUNDARY | 100 | Chapter headings — always split |

**Example:** If text matches both `bullet_list` (HIGH=20) and
`continuation_word` (LOW=40), the bullet_list pattern determines behavior.

---

## Merge Behaviors

| Behavior | Effect | Interactive Mode |
|----------|--------|------------------|
| **MERGE** | Always combine matching blocks | No prompt |
| **ASK** | Prompt user if ambiguous | Shows prompt, learns choice |
| **SPLIT** | Always keep blocks separate | No prompt |
| **BOUNDARY** | Never merge across this point | Chunk boundary marker |

When `--teach` mode is enabled, user responses to ASK behaviors are
persisted to `~/.config/pdf_chunker/learned_patterns.yaml` for future runs.

---

## Pattern Details

### CRITICAL Precedence (priority 10)

#### `qa_sequence`

**Description:** Q&A interview format (Q1:, A1:, Q2:, etc.)

**Behavior:** merge

**Pattern:** `[QA]\d+:`

**Continuation:** `^[QA]\d+:`

#### `numbered_list`

**Description:** Numbered list items (1. Item, 2. Item)

**Behavior:** merge

**Pattern:** `^\s*\d+[\.\)]\s`

**Continuation:** `^\s*\d+[\.\)]\s`

#### `step_sequence`

**Description:** Step-by-step instructions (Step 1:, Step 2:)

**Behavior:** merge

**Pattern:** `^Step\s+\d+[:\.]`

**Continuation:** `^Step\s+\d+[:\.]`

#### `lettered_list`

**Description:** Lettered list items (a. Item, b. Item)

**Behavior:** merge

**Pattern:** `^\s*[a-zA-Z][\.\)]\s`

**Continuation:** `^\s*[a-zA-Z][\.\)]\s`

### HIGH Precedence (priority 20)

#### `bullet_list`

**Description:** Bullet list items (• Item, - Item, * Item)

**Behavior:** merge

**Pattern:** `^\s*[•\-\*]\s`

**Continuation:** `^\s*[•\-\*]\s`

#### `dialogue_tag`

**Description:** Dialogue or speaker tag (Alice:, Bob:)

**Behavior:** ask

**Pattern:** `^[A-Z][a-z]+:\s`

### MEDIUM Precedence (priority 30)

#### `figure_reference`

**Description:** Figure or table reference (Figure 1:, Table 2)

**Behavior:** split

**Pattern:** `^(?:Figure|Fig\.?|Table|Exhibit)\s+\d+`

#### `footnote_marker`

**Description:** Possible footnote marker ([1], (2), 3 )

**Behavior:** ask

**Pattern:** `^[\[\(]?\d+[\]\)]?\s`

### LOW Precedence (priority 40)

#### `continuation_word`

**Description:** Sentence continuation word (And, But, However...)

**Behavior:** merge

**Pattern:** `^(?:And|But|So|However|Therefore|Yet|Still|Also|Meanwhile|Additionally|Then|Thus|Instead|Nevertheless|Nonetheless|Consequently|Moreover)\b`

### BOUNDARY Precedence (priority 100)

#### `chapter_heading`

**Description:** Chapter heading (Chapter 1, Chapter 2)

**Behavior:** boundary

**Pattern:** `^Chapter\s+\d+`

#### `part_marker`

**Description:** Part marker (Part I, Part One)

**Behavior:** boundary

**Pattern:** `^Part\s+(?:\d+|[IVX]+|One|Two|Three)`

#### `section_marker`

**Description:** Section marker (Section 1, §1)

**Behavior:** split

**Pattern:** `^(?:Section|§)\s*\d+`


---

## Usage in Code

```python
from pdf_chunker.patterns import PatternRegistry, MergeDecision

registry = PatternRegistry()

# Evaluate merge decision
decision: MergeDecision = registry.should_merge(prev_text, curr_text)
if decision.should_merge:
    merged = prev_text + " " + curr_text
    print(f"Merged: {decision.reason}")

# With interactive callback for ambiguous cases
def my_prompt(prev, curr, pattern, ctx):
    choice = input(f"Merge {pattern.name}? [Y/n] ")
    return (choice.lower() != "n", "once")

decision = registry.should_merge(
    prev_text, curr_text,
    interactive_callback=my_prompt
)
```

---

## Confidence-Based Decisions

For patterns marked with `ASK` behavior, the system uses confidence scoring:

- **High confidence (≥0.85):** Decision applied automatically
- **Medium confidence (0.30–0.85):** Interactive prompt shown
- **Low confidence (<0.30):** Default behavior applied

Special confidence functions:
- `qa_sequence_confidence()`: Detects Q&A interview patterns
- `colon_list_boundary_confidence()`: Detects colon-prefixed list items
- `evaluate_merge_with_confidence()`: Combined confidence evaluation

See `pdf_chunker/patterns.py` for implementation details.
