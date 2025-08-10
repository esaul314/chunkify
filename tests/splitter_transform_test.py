from pdf_chunker.splitter import _split_text_into_chunks


def test_splitter_respects_cleaning(pdf_case):
    raw, func, expected = pdf_case
    chunks = _split_text_into_chunks(func(raw).rstrip(), chunk_size=100, overlap=0)
    assert chunks == [expected]
