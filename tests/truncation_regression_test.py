from pdf_chunker.passes.split_semantic import _get_split_fn

SENTINEL = "Alignment and trust are challenging"


def test_split_semantic_preserves_long_blocks() -> None:
    split, _ = _get_split_fn(chunk_size=400, overlap=0, min_chunk_size=40)
    long_text = "a" * 25_000 + SENTINEL
    chunks = split(long_text)
    assert any(SENTINEL in c for c in chunks)
