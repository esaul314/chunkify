from pdf_chunker.splitter import _rebalance_bullet_chunks


def test_rebalance_cleans_wrapped_bullets():
    chunks = ["Intro:\n• Item one\n• continues here\n• Second item\n•\nAfter list."]
    assert _rebalance_bullet_chunks(chunks) == [
        "Intro:\n• Item one continues here\n• Second item\nAfter list."
    ]
