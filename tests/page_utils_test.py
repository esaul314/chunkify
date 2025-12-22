import pytest
from pdf_chunker.page_utils import parse_page_ranges


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("1", {1}),
        ("1,3-5", {1, 3, 4, 5}),
        ("2-4", {2, 3, 4}),
        (" 1 , 3-4 , 6 ", {1, 3, 4, 6}),
    ],
)
def test_parse_page_ranges_valid(spec: str, expected: set[int]) -> None:
    assert parse_page_ranges(spec) == expected


@pytest.mark.parametrize("spec", ["0", "1-0", "2-1", "a", "1,b"])
def test_parse_page_ranges_invalid(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_page_ranges(spec)
