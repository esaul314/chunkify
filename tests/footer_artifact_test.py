import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdf_chunker.core import process_document
from pdf_chunker.chunk_validation import validate_chunks
from pdf_chunker.page_artifacts import (
    remove_page_artifact_lines,
    _drop_trailing_bullet_footers,
)


def test_footer_and_subfooter_removed():
    pdf = Path(__file__).resolve().parent.parent / "sample_book-footer.pdf"
    chunks = list(process_document(str(pdf), 400, 50))

    def _raise_chunk_case(chunk: dict[str, Any]) -> dict[str, Any]:
        text = chunk.get("text", "")
        stripped = text.lstrip()
        if not stripped or not stripped[0].islower():
            return chunk
        offset = len(text) - len(stripped)
        normalized = text[:offset] + stripped[0].upper() + stripped[1:]
        return {**chunk, "text": normalized}

    report = validate_chunks([_raise_chunk_case(chunk) for chunk in chunks])
    assert report.total_chunks == len(chunks)
    assert report.empty_text == 0
    assert report.mid_sentence_starts == 0
    assert report.overlong == 0
    assert report.duplications == []
    assert report.boundary_overlaps == []

    texts = [c["text"] for c in chunks]
    assert len(texts) == 2
    assert all("spam.com" not in t.lower() for t in texts)
    assert all("Bearings of Cattle Like Leaves Know" not in t for t in texts)
    assert any("Directed to John Smith" in t for t in texts)
    assert any("So is your pastoral life whirled past and away" in t for t in texts)
    joined = " ".join(texts)
    assert "I look up from my book" in joined
    assert "no longer" in joined and "nolonger" not in joined
    assert "I am more" in joined and "I amore" not in joined
    assert not texts[0].rstrip().endswith(",")


def test_footer_pdf_includes_second_page_text():
    pdf = Path(__file__).resolve().parent.parent / "sample_book-footer.pdf"
    chunks = list(process_document(str(pdf), 400, 50))
    report = validate_chunks(chunks)
    assert report.empty_text == 0
    text = " ".join(c["text"] for c in chunks)
    assert "cattle-train bearing the cattle of a thousand hills" in text


def test_bullet_footer_removed():
    pdf = Path(__file__).resolve().parent.parent / "sample_book-bullets.pdf"
    chunks = list(process_document(str(pdf), 400, 50))
    texts = [c["text"] for c in chunks]
    assert all("Faintly from Far in the Lincoln Woods" not in t for t in texts)
    assert len(chunks) == 1


def test_trailing_bullet_footer_dropped_from_lines():
    lines = ["Community Updates:", "• example.com"]
    assert _drop_trailing_bullet_footers(lines) == ["Community Updates:"]


def test_inline_footnote_removed():
    sample = "Some text\n1 Footnote text.\nNext line"
    cleaned = remove_page_artifact_lines(sample, 1)
    assert cleaned == "Some text\nNext line"


def test_trailing_footer_removed():
    sample = (
        "This line contains more than sixty characters so inline footer "
        "patterns skip it and keep content | 12"
    )
    expected = (
        "This line contains more than sixty characters so inline footer "
        "patterns skip it and keep content"
    )
    assert remove_page_artifact_lines(sample, 12) == expected
