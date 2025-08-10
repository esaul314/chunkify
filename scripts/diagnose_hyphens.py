#!/usr/bin/env python3
"""
diagnose_hyphens.py

Reads a JSONL file and prints all non-ASCII characters found in the "text" field,
along with their counts, so you can spot exactly which hyphen-like codepoint is in use.
"""

import json
import sys
from collections import Counter


def diagnose(path: str) -> None:
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            # Count any char outside the basic ASCII range
            for ch in text:
                if ord(ch) > 127:
                    counts[ch] += 1

    # Print results sorted by frequency
    for ch, cnt in counts.most_common():
        print(f"U+{ord(ch):04X}  {ch!r}    {cnt}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_hyphens.py yourfile.jsonl")
        sys.exit(1)
    diagnose(sys.argv[1])
