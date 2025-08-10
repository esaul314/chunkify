from pdf_chunker.splitter import _rebalance_bullet_chunks


def test_inline_bullet_block_rebalanced():
    chunks = [
        (
            "Some text at the beginning, then we have a bulleted list: "
            "• What are your users doing? (And, ideally, what do they think they're trying to do?) "
            "• How is the platform performing when they do that? Is it efficient? Responsive?\n\nCorrect?"
        ),
        (
            "• Do you need to make any of your products faster or cheaper or easier?\n"
            "• Is there an unmet need for a product or integration that you should provide?\n\n"
            "The text continues in the next paragraph..."
        ),
    ]
    assert _rebalance_bullet_chunks(chunks) == [
        "Some text at the beginning, then we have a bulleted list:\nCorrect?",
        (
            "• What are your users doing? (And, ideally, what do they think they're trying to do?)\n"
            "• How is the platform performing when they do that? Is it efficient? Responsive?\n"
            "• Do you need to make any of your products faster or cheaper or easier?\n"
            "• Is there an unmet need for a product or integration that you should provide?\n\n"
            "The text continues in the next paragraph..."
        ),
    ]
