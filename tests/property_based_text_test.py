from hypothesis import given, strategies as st
from pdf_chunker.text_cleaning import clean_text
from pdf_chunker import splitter


@given(st.text().filter(lambda s: "\x95" not in s))
def test_clean_text_idempotent(sample: str) -> None:
    cleaned = clean_text(sample)
    assert clean_text(cleaned) == cleaned
    assert all(ord(ch) >= 32 or ch == "\n" for ch in cleaned)


@given(st.text(min_size=1, max_size=200))
def test_split_text_preserves_non_whitespace(sample: str) -> None:
    chunks = splitter._split_text_into_chunks(sample, chunk_size=50, overlap=0)
    joined = "".join(chunks)
    strip_ws = lambda s: "".join(ch for ch in s if not ch.isspace())
    assert set(strip_ws(joined)) <= set(strip_ws(sample))
