from __future__ import annotations

from itertools import combinations, chain
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from scripts.parity import run_parity
from tests.parity.normalize import canonical_rows
from tests.parity import exceptions
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.emit_jsonl import emit_jsonl

SAMPLES = Path("tests/golden/samples")
PDFS = sorted(SAMPLES.glob("*.pdf"))


def _project(row: Mapping[str, object]) -> dict:
    base = {"text": row.get("text", "")}
    meta_key = "metadata" if "metadata" in row else "meta" if "meta" in row else None
    return base if meta_key is None else {**base, "metadata": row[meta_key]}


def _rows(path: Path) -> list[dict]:
    return [_project(r) for r in canonical_rows(path)]


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
    pairs = [(pdf, run_parity(pdf, tmp_path / f"{i}", flags)) for i, pdf in enumerate(PDFS)]
    assert all(
        exceptions.apply(_rows(l), exceptions.get("test_e2e_parity_flags", flags, pdf.name))
        == exceptions.apply(_rows(n), exceptions.get("test_e2e_parity_flags", flags, pdf.name))
        for pdf, (l, n) in pairs
    )
    if "--no-metadata" in flags:
        assert all(
            row.keys() == {"text"}
            for path in chain.from_iterable(pairs)
            for row in canonical_rows(path)
        )


@pytest.mark.parametrize("pdf", PDFS)
def test_exclude_pages_yields_no_rows(tmp_path: Path, pdf: Path) -> None:
    legacy, new = run_parity(pdf, tmp_path / pdf.stem, ("--exclude-pages", "1"))
    assert list(canonical_rows(legacy)) == []
    assert list(canonical_rows(new)) == []


def test_emit_jsonl_omits_meta_when_absent() -> None:
    doc = {"type": "chunks", "items": [{"text": "hello"}]}
    result = emit_jsonl(Artifact(payload=doc, meta={})).payload
    assert result == [{"text": "hello"}]
    assert all(row.keys() == {"text"} for row in result)
