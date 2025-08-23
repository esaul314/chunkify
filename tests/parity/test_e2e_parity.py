from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from scripts.parity import run_parity
from tests.parity.normalize import canonical_rows

SAMPLES = Path("tests/golden/samples")
PDFS = sorted(SAMPLES.glob("*.pdf"))


def _project(row: Mapping[str, object]) -> dict:
    base = {"text": row.get("text", "")}
    meta_key = "metadata" if "metadata" in row else "meta" if "meta" in row else None
    return base if meta_key is None else {**base, "metadata": row[meta_key]}


def _rows(path: Path) -> list[dict]:
    return [_project(r) for r in canonical_rows(path)]


def _assert_text_only(path: Path) -> None:
    assert all(row.keys() == {"text"} for row in canonical_rows(path))


def _equal(pdf: Path, tmp: Path, flags: Sequence[str]) -> bool:
    legacy, new = run_parity(pdf, tmp, flags)
    if "--no-metadata" in flags:
        _assert_text_only(legacy)
        _assert_text_only(new)
    return _rows(legacy) == _rows(new)


def _flag_args() -> dict[str, str | None]:
    return {
        "--exclude-pages": "1",
        "--chunk-size": "200",
        "--overlap": "10",
        "--no-metadata": None,
    }


def flag_sets() -> list[tuple[str, ...]]:
    items = list(_flag_args().items())
    return [
        tuple(arg for flag, val in combo for arg in ([flag, val] if val is not None else [flag]))
        for r in range(len(items) + 1)
        for combo in combinations(items, r)
    ]


@pytest.mark.parametrize("flags", flag_sets(), ids=lambda f: " ".join(f) or "base")
def test_e2e_parity_flags(tmp_path: Path, flags: tuple[str, ...]) -> None:
    assert all(_equal(pdf, tmp_path / f"{i}", flags) for i, pdf in enumerate(PDFS))
