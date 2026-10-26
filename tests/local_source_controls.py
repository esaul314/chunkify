"""Explicit private controls for optional local-source regressions.

No document location, fingerprint or expectation is bundled with public source.
Set PDF_CHUNKER_TEST_CONTROLS to a private version-1 JSON object with a values
mapping. A missing optional manifest supplies no fixtures; incomplete, malformed
or changed explicit controls fail. Tests never download source material.
"""

import json
import os
from pathlib import Path

import pytest


def private_value(key, default=None):
    path = os.environ.get("PDF_CHUNKER_TEST_CONTROLS")
    if not path:
        return default
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or not isinstance(data.get("values"), dict)
    ):
        raise ValueError("Invalid private test controls")
    if key not in data["values"]:
        raise ValueError("Incomplete explicit private test controls")
    return data["values"][key]


def private_rows(key, width):
    rows = private_value(key, ())
    if not rows:
        return [
            pytest.param(
                *([None] * width), marks=pytest.mark.skip(reason="optional private controls absent")
            )
        ]
    if not isinstance(rows, (tuple, list)) or any(
        not isinstance(row, (tuple, list)) or len(row) != width for row in rows
    ):
        raise ValueError("Invalid private fixture rows")
    return rows
