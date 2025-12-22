#!/usr/bin/env python3

from pdf_chunker.pdf_blocks import merge_continuation_blocks


def test_cross_page_sentence_with_proper_name(block):
    blocks = [
        block("Economic inequality is usually measured by the", page=1),
        block("Gini", page=2),
        block("coefficient extends across pages.", page=2),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 1
    assert "Gini coefficient" in merged[0].text


def test_cross_page_sentence_lowercases_common_articles(block):
    blocks = [
        block("The analysis compared insights across the", page=1),
        block("Of sample cohorts across regions.", page=2),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 1
    assert "of sample cohorts across regions." in merged[0].text


def test_cross_page_sentence_with_min_word_context(block):
    blocks = [
        block("Economic inequality is measured by the", page=1),
        block("Gini coefficient spans pages.", page=2),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 1
    assert "Gini coefficient" in merged[0].text


def test_cross_page_sentence_without_page_numbers(block):
    blocks = [
        block("Economic inequality is usually measured by the"),
        block("Gini coefficient carries on."),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 2
    assert "Gini coefficient" in merged[1].text


def test_cross_page_does_not_merge_entire_document(block):
    blocks = [
        block("Economic inequality is usually measured by the", page=1),
        block("Gini coefficient completes the sentence.", page=2),
        block("New paragraph begins here with its own sentence.", page=3),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 2
    assert merged[1].text.startswith("New paragraph")


def test_non_consecutive_pages_do_not_merge(block):
    blocks = [
        block("Ends without punctuation", page=1),
        block("Resume after gap", page=3),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 2


def test_comma_same_page_continuation(block):
    blocks = [
        block("Chapters may end with a teaser,", page=1),
        block("However more follows on the same page.", page=1),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 1
    assert "teaser, However" in merged[0].text


def test_three_page_sentence_splits_after_second_page(block):
    blocks = [
        block("Part one", page=1),
        block("continues on page two", page=2),
        block("and finally ends", page=3),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 2
    assert merged[0].source.get("page_range") == (1, 2)


def test_footnote_block_appends_after_merge(block):
    blocks = [
        block("Whether or not they've called their", page=9),
        block(
            'We\u2019ll sometimes call these teams your "users".',
            page=9,
            footnote_block=True,
        ),
        block("efforts platform engineering, they embody the mindset.", page=9),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 1
    assert "called their efforts platform engineering" in merged[0].text
    assert merged[0].text.endswith('We\u2019ll sometimes call these teams your "users".')


def test_heading_like_block_does_not_merge_quote_continuation(block):
    blocks = [
        block('He said "Something unfinished', page=1),
        block("Next Section:", page=1),
    ]
    merged = list(merge_continuation_blocks(blocks))
    assert len(merged) == 2
