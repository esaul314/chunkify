from pdf_chunker.chunk_validation import validate_chunks


def test_empty_chunks_flagged():
    report = validate_chunks([])
    assert report.is_empty()
    assert report.has_issues()
