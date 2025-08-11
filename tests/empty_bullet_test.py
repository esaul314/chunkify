from pdf_chunker.splitter import _rebalance_bullet_chunks


def test_rebalance_removes_empty_bullets():
    chunks = ["Intro:\n• Alpha\n•\n• Beta", "• Gamma"]
    result = _rebalance_bullet_chunks(chunks)
    assert all(line.strip() != "•" for chunk in result for line in chunk.splitlines())
