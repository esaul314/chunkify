from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdf_chunker.utils import _compute_readability


_SAMPLE_GOLDEN_PATH = Path("tests/golden/expected/pdf.jsonl")


def _load_sample_text() -> str:
    first_line = _SAMPLE_GOLDEN_PATH.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(first_line)["text"]


def test_readability_matches_expected_grade() -> None:
    readability = _compute_readability(_load_sample_text())
    assert readability["flesch_kincaid_grade"] == pytest.approx(11.8675)
    assert readability["difficulty"] == "high_school"


def test_readability_handles_empty_text() -> None:
    readability = _compute_readability("")
    assert readability["flesch_kincaid_grade"] == 0.0
    assert readability["difficulty"] == "elementary"
