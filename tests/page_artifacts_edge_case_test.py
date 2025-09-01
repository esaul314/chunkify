import pytest
from pdf_chunker.page_artifacts import remove_page_artifact_lines, strip_page_artifact_suffix


@pytest.mark.parametrize(
    "text,page,expected",
    [
        ("content | IV", 4, "content"),
        ("content | iv", 4, "content"),
    ],
)
def test_strip_page_artifact_suffix_roman(text, page, expected):
    assert strip_page_artifact_suffix(text, page) == expected


def test_header_prefix_removed_and_sentence_preserved():
    line = "Person Name, PMP Alma, Quebec, Canada This is filled with the bleating of calves"
    cleaned = remove_page_artifact_lines(line, 1)
    assert cleaned.startswith("This is filled")
    assert "Person Name" not in cleaned
