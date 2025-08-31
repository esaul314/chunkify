from pdf_chunker.splitter import _split_text_into_chunks


def test_splitter_respects_cleaning(pdf_case):
    raw, func, expected = pdf_case
    chunks = _split_text_into_chunks(func(raw).rstrip(), chunk_size=100, overlap=0)
    assert chunks == [expected]


def test_splitter_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(20))
    chunks = _split_text_into_chunks(text, chunk_size=10, overlap=2)
    assert [len(c.split()) for c in chunks] == [10, 10]
    assert chunks[1].split()[0] == "w8"
