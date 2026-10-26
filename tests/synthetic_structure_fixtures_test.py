"""Generated controls retain exact bytes, layout cases and invented content."""

from pathlib import Path

import fitz
import pytest

from scripts.generate_structure_fixtures import fixtures, structure_pages

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,pages", [("synthetic-structure.pdf", 55), ("synthetic-bullets.pdf", 1)]
)
def test_generated_fixture_bytes_and_source_structure(name, pages):
    generated = fixtures()[name]
    assert (ROOT / name).read_bytes() == generated
    with fitz.open(stream=generated, filetype="pdf") as document:
        assert len(document) == pages
        assert not document.embfile_names()
        assert not any(value for key, value in document.metadata.items() if key != "format")
        text = "\n".join(page.get_text() for page in document)
    assert "\ufffd" not in text
    if pages == 55:
        assert "CHAPTER 1" in text and "Figure 1-1" in text and "Table 1-1" in text
        assert "Synthetic cobalt markers" in text
        assert len(structure_pages()) == pages
    else:
        assert text.count("•") == 3
