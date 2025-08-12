from pdf_chunker.chunk_validation import validate_chunks


def test_validate_chunks_flags_empty_result():
    report = validate_chunks([])
    assert report.has_issues()
    assert report.total_chunks == 0
