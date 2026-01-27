import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdf_chunker.chunk_validation import validate_chunks
from pdf_chunker.core import (
    chunk_text,
    filter_blocks,
    parse_exclusions,
    process_document,
)
from pdf_chunker.page_artifacts import (
    _drop_trailing_bullet_footers,
    remove_page_artifact_lines,
)
from pdf_chunker.parsing import extract_structured_text


def _expected_chunk_count(pdf: Path, chunk_size: int, overlap: int) -> int:
    blocks = extract_structured_text(pdf, exclude_pages=None)
    filtered = filter_blocks(blocks, parse_exclusions(None))
    baseline_chunks = chunk_text(
        filtered,
        chunk_size,
        overlap,
        min_chunk_size=max(8, chunk_size // 10),
        enable_dialogue_detection=True,
    )
    return len(baseline_chunks)


@pytest.fixture(scope="module")
def sample_footer_pdf() -> Path:
    return Path(__file__).resolve().parent.parent / "sample_book-footer.pdf"


@pytest.fixture(scope="module")
def sample_footer_chunks(sample_footer_pdf: Path) -> list[dict[str, Any]]:
    return list(process_document(str(sample_footer_pdf), 400, 50))


@pytest.fixture(scope="module")
def sample_footer_texts(sample_footer_chunks: list[dict[str, Any]]) -> list[str]:
    return [chunk["text"] for chunk in sample_footer_chunks]


@pytest.fixture(scope="module")
def sample_bullet_pdf() -> Path:
    return Path(__file__).resolve().parent.parent / "sample_book-bullets.pdf"


@pytest.fixture(scope="module")
def sample_bullet_chunks(sample_bullet_pdf: Path) -> list[dict[str, Any]]:
    return list(process_document(str(sample_bullet_pdf), 400, 50))


@pytest.fixture
def book_club_footer_lines() -> list[str]:
    return [
        "BOOK CLUB QUESTIONS",
        "• Who guides the cattle-train?",
        "• What happens when the cattle arrive?",
    ]


@pytest.fixture
def shipping_footer_lines() -> list[str]:
    return [
        "SHIPPING NOTICE",
        "• Directed to John Smith",
        "• He expects the parcel at noon",
    ]


def test_footer_and_subfooter_removed(
    sample_footer_pdf: Path,
    sample_footer_chunks: list[dict[str, Any]],
    sample_footer_texts: list[str],
) -> None:
    def _raise_chunk_case(chunk: dict[str, Any]) -> dict[str, Any]:
        text = chunk.get("text", "")
        stripped = text.lstrip()
        if not stripped or not stripped[0].islower():
            return chunk
        offset = len(text) - len(stripped)
        normalized = text[:offset] + stripped[0].upper() + stripped[1:]
        return {**chunk, "text": normalized}

    report = validate_chunks([_raise_chunk_case(chunk) for chunk in sample_footer_chunks])
    assert report.total_chunks == len(sample_footer_chunks)
    assert report.empty_text == 0
    assert report.mid_sentence_starts == 0
    assert report.overlong == 0
    assert report.duplications == []
    assert report.boundary_overlaps == []

    expected_count = _expected_chunk_count(sample_footer_pdf, 400, 50)
    assert len(sample_footer_texts) == expected_count
    assert all("spam.com" not in t.lower() for t in sample_footer_texts)
    assert all("Bearings of Cattle Like Leaves Know" not in t for t in sample_footer_texts)
    assert any("Directed to John Smith" in t for t in sample_footer_texts)
    assert any("So is your pastoral life whirled past and away" in t for t in sample_footer_texts)
    joined = " ".join(sample_footer_texts)
    assert "I look up from my book" in joined
    assert "no longer" in joined and "nolonger" not in joined
    assert "I am more" in joined and "I amore" not in joined
    assert not sample_footer_texts[0].rstrip().endswith(",")


def test_footer_pdf_includes_second_page_text(sample_footer_chunks: list[dict[str, Any]]) -> None:
    report = validate_chunks(sample_footer_chunks)
    assert report.empty_text == 0
    text = " ".join(chunk["text"] for chunk in sample_footer_chunks)
    assert "cattle-train bearing the cattle of a thousand hills" in text


def test_bullet_footer_removed(
    sample_bullet_pdf: Path, sample_bullet_chunks: list[dict[str, Any]]
) -> None:
    texts = [chunk["text"] for chunk in sample_bullet_chunks]
    assert all("Faintly from Far in the Lincoln Woods" not in t for t in texts)
    expected_count = _expected_chunk_count(sample_bullet_pdf, 400, 50)
    assert len(sample_bullet_chunks) == expected_count


def test_trailing_bullet_footer_dropped_from_lines():
    lines = ["Community Updates:", "• example.com"]
    assert _drop_trailing_bullet_footers(lines) == ["Community Updates:"]


def test_trailing_bullet_footer_dropped_with_header_signal(
    book_club_footer_lines: list[str],
) -> None:
    pruned = _drop_trailing_bullet_footers(book_club_footer_lines)
    assert pruned == ["BOOK CLUB QUESTIONS"]


def test_trailing_bullet_footer_dropped_for_uppercase_header_without_partners() -> None:
    lines = ["PROGRAM OVERVIEW", "• Who leads the quarterly review?"]
    pruned = _drop_trailing_bullet_footers(lines)
    assert pruned == ["PROGRAM OVERVIEW"]


def test_trailing_bullet_footer_preserved_for_shipping_notice(
    shipping_footer_lines: list[str],
) -> None:
    # Shipping-related bullet content is now preserved as legitimate content
    pruned = _drop_trailing_bullet_footers(shipping_footer_lines)
    assert pruned == shipping_footer_lines


def test_trailing_bullet_footer_preserves_legitimate_list() -> None:
    lines = ["SHOPPING LIST:", "• apples", "• oranges"]
    assert _drop_trailing_bullet_footers(lines) == lines


def test_trailing_bullet_footer_preserves_colon_prompted_list() -> None:
    lines = [
        "There were two goals to this:",
        "• To stress to his team that it was not acceptable to resist a new application team ask just because it wasn't in their earlier plans",
        "• To remind his stakeholders that not telling his team early about their needs would result in higher costs, because the plan would have to change to accommodate the new work",
    ]
    assert _drop_trailing_bullet_footers(lines) == lines


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
