from hypothesis import given, strategies as st
from pdf_chunker.text_cleaning import clean_text
from pdf_chunker import splitter


ascii_text = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126))


@given(ascii_text)
def test_clean_text_idempotent(sample: str) -> None:
    cleaned = clean_text(sample)
    assert clean_text(cleaned) == cleaned
    assert all(ord(ch) >= 32 or ch == "\n" for ch in cleaned)


@given(ascii_text.filter(lambda s: s))
def test_split_text_preserves_non_whitespace(sample: str) -> None:
    chunks = splitter._split_text_into_chunks(sample, chunk_size=50, overlap=0)
    joined = "".join(chunks)
    strip_ws = lambda s: "".join(ch for ch in s if not ch.isspace())
    assert set(strip_ws(joined)) <= set(strip_ws(sample))
