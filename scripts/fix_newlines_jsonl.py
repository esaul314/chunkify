#!/usr/bin/env python3
"""
fix_newlines_jsonl.py

Reads JSONL from stdin (or a file), fixes spurious \n\n inside words or clauses,
and writes cleaned JSONL to stdout.

Algorithm:
  1. **Merge true word-splits**
     Pattern: letter+  \n\n  lowercase-letter+
     → try head+tail; if spell-checker says it’s a real word, glue;
       else fall back to head + “ ” + tail.
  2. **Collapse clause-splits**
     Any \n\n followed by lowercase or ‘(’ → replace \n\n with a space.
  3. Leave all other \n\n (e.g. before uppercase, quotes, headings) intact.
"""

import re, sys, json, subprocess
from functools import partial, reduce

SPELLER_CMD = ["aspell", "list"]


def is_real_word(w: str) -> bool:
    """Return True if aspell knows this word."""
    p = subprocess.Popen(
        SPELLER_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    out, _ = p.communicate(w + "\n")
    return out.strip() == ""


# Phase 1: merge hyphen-style or pure letter splits
_merge_re = re.compile(r"([A-Za-z]+)\n\n([a-z][A-Za-z]+)")


def merge_splits(text: str) -> str:
    def repl(m):
        head, tail = m.group(1), m.group(2)
        candidate = head + tail
        return candidate if is_real_word(candidate) else head + " " + tail

    return _merge_re.sub(repl, text)


# Phase 2: collapse spurious clause splits (lowercase or parentheses)
_collapse_re = re.compile(r"\n\n(?=[a-z(])")


def collapse_clause_breaks(text: str) -> str:
    return _collapse_re.sub(" ", text)


# Full pipeline
def fix_text(text: str) -> str:
    return reduce(lambda acc, fn: fn(acc), (merge_splits, collapse_clause_breaks), text)


def process_stream(inp, outp):
    for line in inp:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            outp.write(line + "\n")
            continue

        txt = obj.get("text")
        if isinstance(txt, str) and "\n\n" in txt:
            obj["text"] = fix_text(txt)

        outp.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            process_stream(f, sys.stdout)
    else:
        process_stream(sys.stdin, sys.stdout)
