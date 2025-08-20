"""Normalization utilities for row comparison in parity tests."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

VOLATILE_FIELDS = frozenset({"timings"})


def load_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from ``path`` line by line."""
    with Path(path).open(encoding="utf-8") as handle:
        yield from (
            json.loads(line)
            for line in handle
            if line.strip()
        )


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def normalize(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical representation of ``row``.

    - Keys are sorted for deterministic ordering.
    - Leading/trailing whitespace is removed from string values.
    - Volatile fields (e.g., timings) are excluded.
    """
    return {
        key: _strip(value)
        for key, value in sorted(row.items())
        if key not in VOLATILE_FIELDS
    }


def canonical_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Chain :func:`load_rows` and :func:`normalize` for convenience."""
    return (normalize(row) for row in load_rows(path))
