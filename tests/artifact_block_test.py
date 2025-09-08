from pdf_chunker.pdf_blocks import _filter_margin_artifacts


def test_filter_margin_artifacts_numeric_only() -> None:
    page_height = 1000
    numeric_block = (0, 950, 100, 970, "123")
    mixed_block = (0, 10, 100, 30, "Section 1")
    filtered = _filter_margin_artifacts([numeric_block, mixed_block], page_height)
    assert filtered == [mixed_block]

