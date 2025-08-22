#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, ".")

from pdf_chunker.pdf_parsing import merge_continuation_blocks


def test_cross_page_sentence_with_proper_name():
    blocks = [
        {
            "text": "Economic inequality is usually measured by the",
            "source": {"page": 1},
        },
        {"text": "Gini", "source": {"page": 2}},
        {"text": "coefficient extends across pages.", "source": {"page": 2}},
    ]
    merged = merge_continuation_blocks(blocks)
    assert len(merged) == 1
    assert "Gini coefficient" in merged[0]["text"]


def test_cross_page_sentence_with_min_word_context():
    blocks = [
        {
            "text": "Economic inequality is measured by the",
            "source": {"page": 1},
        },
        {
            "text": "Gini coefficient spans pages.",
            "source": {"page": 2},
        },
    ]
    merged = merge_continuation_blocks(blocks)
    assert len(merged) == 1
    assert "Gini coefficient" in merged[0]["text"]


def test_cross_page_sentence_without_page_numbers():
    blocks = [
        {"text": "Economic inequality is usually measured by the"},
        {"text": "Gini coefficient carries on."},
    ]
    merged = merge_continuation_blocks(blocks)
    assert len(merged) == 1
    assert "Gini coefficient" in merged[0]["text"]


def test_cross_page_does_not_merge_entire_document():
    blocks = [
        {
            "text": "Economic inequality is usually measured by the",
            "source": {"page": 1},
        },
        {
            "text": "Gini coefficient completes the sentence.",
            "source": {"page": 2},
        },
        {
            "text": "New paragraph begins here with its own sentence.",
            "source": {"page": 3},
        },
    ]
    merged = merge_continuation_blocks(blocks)
    assert len(merged) == 2
    assert merged[1]["text"].startswith("New paragraph")


def test_comma_same_page_continuation():
    blocks = [
        {"text": "Chapters may end with a teaser,", "source": {"page": 1}},
        {
            "text": "However more follows on the same page.",
            "source": {"page": 1},
        },
    ]
    merged = merge_continuation_blocks(blocks)
    assert len(merged) == 1
    assert "teaser, However" in merged[0]["text"]
