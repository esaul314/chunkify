#!/usr/bin/env python3
"""
fix_newlines.py

Reads from stdin (or a file) and writes cleaned text to stdout.
Whenever it sees a double‐newline between two letter‐only fragments,
it will:

  1. Try to merge them (fragment1 + fragment2).
  2. Test the merged form against your system’s spellchecker (aspell).
     - If it’s a real word, remove the newline entirely.
     - Otherwise, replace the break with a single space.

Paragraph breaks where the second fragment starts with an uppercase letter
will be left alone.
"""

import re
import subprocess
import sys

# Spell‐checker command: we’ll use aspell in “list” mode.
# Make sure you have `aspell` installed (`dnf install aspell`).
SPELLER_CMD = ["aspell", "list"]


def is_real_word(word: str) -> bool:
    """Return True if `word` is recognized by aspell."""
    # aspell list prints unknown words—so if our word does NOT show up,
    # it’s known.
    p = subprocess.Popen(
        SPELLER_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    out, _ = p.communicate(word + "\n")
    return out.strip() == ""


# Regex to find letter‐only fragments separated by double‐newline.
PAT = re.compile(r"([A-Za-z]+)\n\n([a-z][A-Za-z]+)")


def repl(match: re.Match) -> str:
    head, tail = match.group(1), match.group(2)
    candidate = (head + tail).lower()
    if is_real_word(candidate):
        # e.g. "Sci\n\nentific" -> "Scientific"
        # Preserve original casing on head + tail
        return head + tail
    else:
        # e.g. "find an\n\naudience" -> "find an audience"
        return head + " " + tail


def fix_stream(in_stream, out_stream):
    text = in_stream.read()
    # Perform a global substitution
    cleaned = PAT.sub(repl, text)
    out_stream.write(cleaned)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f_in:
            fix_stream(f_in, sys.stdout)
    else:
        fix_stream(sys.stdin, sys.stdout)
