from functools import reduce
from typing import Callable, TypeVar

from hypothesis import given, strategies as st
from pdf_chunker import splitter
from pdf_chunker.page_artifacts import remove_page_artifact_lines
from pdf_chunker.text_cleaning import clean_text


T = TypeVar("T")


def compose(*funcs: Callable[[T], T]) -> Callable[[T], T]:
    return reduce(lambda f, g: lambda x: f(g(x)), funcs, lambda x: x)


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
    assert strip_ws(joined) == strip_ws(sample)


@given(st.text(min_size=1))
def test_split_roundtrip_cleaning(sample: str) -> None:
    pipeline = compose(
        clean_text,
        "".join,
        lambda s: splitter._split_text_into_chunks(s, 30, 0),
        clean_text,
    )
    assert pipeline(sample) == clean_text(sample)


def test_inline_footnote_continuation_preserved() -> None:
    sample = "Lead in.\n3 Footnote text. The continuation survives."
    cleaned = remove_page_artifact_lines(sample, 3)
    assert cleaned.endswith("The continuation survives.")
    assert "Lead in." in cleaned
