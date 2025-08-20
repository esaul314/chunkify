from __future__ import annotations

from pathlib import Path

from scripts.parity import run_parity
from tests.parity.normalize import canonical_rows

SAMPLES = Path("tests/golden/samples")


def _pdfs() -> list[Path]:
    return sorted(SAMPLES.glob("*.pdf"))


def _rows(path: Path) -> list[dict]:
    return [
        {"text": row.get("text", ""), "metadata": row.get("metadata") or row.get("meta")}
        for row in canonical_rows(path)
    ]


def _equal(pdf: Path, tmp: Path) -> bool:
    legacy, new = run_parity(pdf, tmp)
    return _rows(legacy) == _rows(new)


def test_new_matches_legacy(tmp_path: Path) -> None:
    assert all(_equal(pdf, tmp_path / pdf.stem) for pdf in _pdfs())
