import pytest
from pdf_chunker.page_artifacts import strip_page_artifact_suffix


@pytest.mark.parametrize(
    "text,page,expected",
    [
        ("content | IV", 4, "content"),
        ("content | iv", 4, "content"),
    ],
)
def test_strip_page_artifact_suffix_roman(text, page, expected):
    assert strip_page_artifact_suffix(text, page) == expected
