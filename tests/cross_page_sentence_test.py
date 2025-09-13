#!/usr/bin/env python3

from pdf_chunker.pdf_blocks import merge_continuation_blocks, Block


def test_cross_page_sentence_with_proper_name():
    blocks = [
        Block(
            text="Economic inequality is usually measured by the",
            source={"page": 1},
        ),
        Block(text="Gini", source={"page": 2}),
        Block(text="coefficient extends across pages.", source={"page": 2}),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 1
    assert "Gini coefficient" in merged[0].text


def test_cross_page_sentence_with_min_word_context():
    blocks = [
        Block(text="Economic inequality is measured by the", source={"page": 1}),
        Block(text="Gini coefficient spans pages.", source={"page": 2}),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 1
    assert "Gini coefficient" in merged[0].text


def test_cross_page_sentence_without_page_numbers():
    blocks = [
        Block(text="Economic inequality is usually measured by the", source={}),
        Block(text="Gini coefficient carries on.", source={}),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 1
    assert "Gini coefficient" in merged[0].text


def test_cross_page_does_not_merge_entire_document():
    blocks = [
        Block(
            text="Economic inequality is usually measured by the",
            source={"page": 1},
        ),
        Block(
            text="Gini coefficient completes the sentence.",
            source={"page": 2},
        ),
        Block(
            text="New paragraph begins here with its own sentence.",
            source={"page": 3},
        ),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 2
    assert merged[1].text.startswith("New paragraph")


def test_non_consecutive_pages_do_not_merge():
    blocks = [
        Block(text="Ends without punctuation", source={"page": 1}),
        Block(text="Resume after gap", source={"page": 3}),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 2


def test_comma_same_page_continuation():
    blocks = [
        Block(text="Chapters may end with a teaser,", source={"page": 1}),
        Block(text="However more follows on the same page.", source={"page": 1}),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 1
    assert "teaser, However" in merged[0].text


def test_three_page_sentence_splits_after_second_page():
    blocks = [
        Block(text="Part one", source={"page": 1}),
        Block(text="continues on page two", source={"page": 2}),
        Block(text="and finally ends", source={"page": 3}),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 2
    assert merged[0].source.get("page_range") == (1, 2)
