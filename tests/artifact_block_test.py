from pdf_chunker.pdf_parsing import is_artifact_block


def test_is_artifact_block_numeric_only():
    page_height = 1000
    numeric_block = (0, 950, 100, 970, "123")
    assert is_artifact_block(numeric_block, page_height)

    mixed_block = (0, 10, 100, 30, "Section 1")
    assert not is_artifact_block(mixed_block, page_height)
