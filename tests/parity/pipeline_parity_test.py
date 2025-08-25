from __future__ import annotations

from pathlib import Path

from scripts import parity as sp
from tests.parity.normalize import canonical_rows
from tests.parity import exceptions

SAMPLES = Path("tests/golden/samples")
ARTIFACTS = Path("artifacts/parity")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _pdfs() -> list[Path]:
    return sorted(SAMPLES.glob("*.pdf"))


def _rows(path: Path) -> list[dict]:
    return [
        {"text": row.get("text", ""), "metadata": row.get("metadata") or row.get("meta")}
        for row in canonical_rows(path)
    ]


def _equal(pdf: Path, tmp: Path, *, test: str, flags: tuple[str, ...]) -> bool:
    legacy, new = sp.run_parity(pdf, tmp, flags=flags, diffdir=ARTIFACTS)
    rule = exceptions.get(test, flags, pdf.name)
    return exceptions.apply(_rows(legacy), rule) == exceptions.apply(_rows(new), rule)


def test_new_matches_legacy(tmp_path: Path) -> None:
    assert all(
        _equal(pdf, tmp_path / pdf.stem, test="test_new_matches_legacy", flags=())
        for pdf in _pdfs()
    )


def test_no_metadata_rows_contain_only_text(tmp_path: Path) -> None:
    for pdf in _pdfs():
        legacy, new = sp.run_parity(
            pdf, tmp_path / pdf.stem, flags=["--no-metadata"], diffdir=ARTIFACTS
        )
        for path in (legacy, new):
            assert all(r.keys() == {"text"} for r in canonical_rows(path))
