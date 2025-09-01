from pathlib import Path

import fitz

from pdf_chunker.page_artifacts import remove_page_artifact_lines


def test_footer_strip_counts_sample():
    pdf = Path(__file__).resolve().parent.parent / "sample_book-footer.pdf"
    doc = fitz.open(pdf)
    counts = [
        len(page.get_text("text").splitlines())
        - len(remove_page_artifact_lines(page.get_text("text"), i + 1).splitlines())
        for i, page in enumerate(doc)
    ]
    assert counts == [3, 2]
