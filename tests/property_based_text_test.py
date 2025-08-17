from functools import reduce
from typing import Callable, TypeVar

from hypothesis import given, strategies as st
from pdf_chunker import splitter
from pdf_chunker.text_cleaning import clean_text


T = TypeVar("T")


def _apply_times(fn: Callable[[T], T], times: int, value: T) -> T:
    return reduce(lambda acc, _: fn(acc), range(times), value)


@given(st.text().filter(lambda s: "\x95" not in s), st.integers(min_value=1, max_value=5))
def test_clean_text_idempotent(sample: str, repeats: int) -> None:
    cleaned = clean_text(sample)
    assert _apply_times(clean_text, repeats, cleaned) == cleaned
    assert all(ord(ch) >= 32 or ch == "\n" for ch in cleaned)


@given(st.text(min_size=1, max_size=200))
def test_split_text_preserves_non_whitespace(sample: str) -> None:
    chunks = splitter._split_text_into_chunks(sample, chunk_size=50, overlap=0)
    joined = "".join(chunks)
    strip_ws = lambda s: "".join(ch for ch in s if not ch.isspace())
    assert set(strip_ws(joined)) <= set(strip_ws(sample))
