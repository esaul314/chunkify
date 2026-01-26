#!/usr/bin/env python3
"""Generate MERGE_DECISIONS.md from the pattern registry.

This script auto-generates documentation for merge decisions, ensuring
the docs always reflect the actual code behavior.

Usage:
    python scripts/generate_merge_docs.py
    python scripts/generate_merge_docs.py --check  # Verify docs are up to date
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_chunker.patterns import (
    DEFAULT_PATTERNS,
    MergeBehavior,
    Pattern,
    Precedence,
)


def generate_pattern_table(patterns: list[Pattern]) -> str:
    """Generate a markdown table of all patterns."""
    lines = [
        "| Pattern | Precedence | Behavior | Description |",
        "|---------|------------|----------|-------------|",
    ]
    for p in patterns:
        behavior_icon = {
            MergeBehavior.MERGE: "✅ Merge",
            MergeBehavior.ASK: "❓ Ask",
            MergeBehavior.SPLIT: "❌ Split",
            MergeBehavior.BOUNDARY: "🚧 Boundary",
        }.get(p.behavior, p.behavior.value)
        lines.append(f"| `{p.name}` | {p.precedence.name} | {behavior_icon} | {p.description} |")
    return "\n".join(lines)


def generate_precedence_section() -> str:
    """Generate precedence order explanation."""
    return """## Precedence Order

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
`continuation_word` (LOW=40), the bullet_list pattern determines behavior."""


def generate_behavior_section() -> str:
    """Generate merge behavior explanation."""
    return """## Merge Behaviors

| Behavior | Effect | Interactive Mode |
|----------|--------|------------------|
| **MERGE** | Always combine matching blocks | No prompt |
| **ASK** | Prompt user if ambiguous | Shows prompt, learns choice |
| **SPLIT** | Always keep blocks separate | No prompt |
| **BOUNDARY** | Never merge across this point | Chunk boundary marker |

When `--teach` mode is enabled, user responses to ASK behaviors are
persisted to `~/.config/pdf_chunker/learned_patterns.yaml` for future runs."""


def generate_pattern_details(patterns: list[Pattern]) -> str:
    """Generate detailed documentation for each pattern."""
    by_precedence: dict[Precedence, list[Pattern]] = {}
    for p in patterns:
        by_precedence.setdefault(p.precedence, []).append(p)

    sections = []
    for prec in Precedence:
        if prec not in by_precedence:
            continue
        sections.append(f"### {prec.name} Precedence (priority {prec.value})\n")
        for p in by_precedence[prec]:
            lines = [
                f"#### `{p.name}`\n",
                f"**Description:** {p.description}\n",
                f"**Behavior:** {p.behavior.value}\n",
                f"**Pattern:** `{p.match.pattern}`\n",
            ]
            if p.continuation:
                lines.append(f"**Continuation:** `{p.continuation.pattern}`\n")
            sections.append("\n".join(lines))
    return "\n".join(sections)


def generate_merge_docs() -> str:
    """Generate the full MERGE_DECISIONS.md content."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    return f"""# Merge Decision Reference

> **Auto-generated:** {timestamp}  
> **Source:** `pdf_chunker/patterns.py`  
> **Regenerate:** `python scripts/generate_merge_docs.py`

This document describes how pdf_chunker decides whether to merge adjacent
text blocks. These decisions are critical for maintaining semantic coherence
in chunked output.

---

## Quick Reference

{generate_pattern_table(DEFAULT_PATTERNS)}

---

{generate_precedence_section()}

---

{generate_behavior_section()}

---

## Pattern Details

{generate_pattern_details(DEFAULT_PATTERNS)}

---

## Usage in Code

```python
from pdf_chunker.patterns import PatternRegistry, MergeDecision

registry = PatternRegistry()

# Evaluate merge decision
decision: MergeDecision = registry.should_merge(prev_text, curr_text)
if decision.should_merge:
    merged = prev_text + " " + curr_text
    print(f"Merged: {{decision.reason}}")

# With interactive callback for ambiguous cases
def my_prompt(prev, curr, pattern, ctx):
    choice = input(f"Merge {{pattern.name}}? [Y/n] ")
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
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if docs are up to date (exit 1 if not)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "MERGE_DECISIONS.md",
        help="Output file path",
    )
    args = parser.parse_args()

    content = generate_merge_docs()

    if args.check:
        if not args.output.exists():
            print(f"ERROR: {args.output} does not exist", file=sys.stderr)
            return 1
        existing = args.output.read_text()
        # Compare ignoring timestamp line
        existing_lines = [
            l for l in existing.splitlines() if not l.startswith("> **Auto-generated:**")
        ]
        new_lines = [l for l in content.splitlines() if not l.startswith("> **Auto-generated:**")]
        if existing_lines != new_lines:
            print(
                f"ERROR: {args.output} is out of date. Run: python scripts/generate_merge_docs.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {args.output} is up to date")
        return 0

    args.output.write_text(content)
    print(f"Generated {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
