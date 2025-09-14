import pytest
from pdf_chunker.passes.emit_jsonl import _merge_text, _rebalance_lists


@pytest.mark.parametrize(
    "raw, rest, expected",
    [
        (
            "Intro\n- a\n",
            "- b\nTail",
            ("Intro", "- a\n- b\nTail"),
        ),
        (
            "Intro\n1. one\n",
            "\n2. two\nTail",
            ("Intro", "1. one\n2. two\nTail"),
        ),
        (
            "Intro\n1. one\n",
            "\n\n2. two\nTail",
            ("Intro", "1. one\n2. two\nTail"),
        ),
    ],
)
def test_rebalance_lists(raw, rest, expected):
    assert _rebalance_lists(raw, rest) == expected

    
def test_merge_text_collapses_list_gap():
    assert _merge_text("1. one", "2. two") == "1. one\n2. two"
