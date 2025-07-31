import sys

sys.path.insert(0, ".")

import pytest
from pdf_chunker.page_utils import parse_page_ranges


def test_parse_individual_pages():
    assert parse_page_ranges("1,3,5") == {1, 3, 5}


def test_parse_page_ranges_function():
    assert parse_page_ranges("2-4") == {2, 3, 4}


def test_parse_mixed_pages_and_ranges():
    assert parse_page_ranges("1,3-4,6") == {1, 3, 4, 6}
