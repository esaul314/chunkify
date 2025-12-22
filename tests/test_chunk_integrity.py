import logging

from pdf_chunker import splitter


def test_corrupted_chunk_retained(caplog):
    chunk = ", fragment with leading comma and sufficient length to avoid removal"
    original = chunk
    with caplog.at_level(logging.ERROR):
        out = splitter._validate_chunk_integrity([chunk], original)
    assert len(out) == 1
    assert "fragment with leading comma" in out[0]
    assert any("appears corrupted" in rec.message for rec in caplog.records)
