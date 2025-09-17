import pytest

from pdf_chunker.passes.emit_jsonl import (
    _first_non_empty_line,
    _is_list_line,
    _merge_text,
    _rebalance_lists,
    _rows_from_item,
    _split,
)


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
        (
            "Lead\n\nIntro",
            "- a\n- b",
            ("Lead", "Intro\n- a\n- b"),
        ),
    ],
)
def test_rebalance_lists(raw, rest, expected):
    assert _rebalance_lists(raw, rest) == expected


def test_merge_text_collapses_list_gap():
    assert _merge_text("1. one", "2. two") == "1. one\n2. two"


def test_split_reserves_intro_for_list():
    text = "Lead\n\nIntro\n- a\n- b"
    limit = len("Lead\n\nIntro")
    assert _split(text, limit) == ["Lead", "Intro\n- a\n- b"]


def test_split_moves_list_after_long_prefix():
    prefix = "x" * 8000
    text = f"{prefix}\nIntro\n- a\n- b"
    chunks = _split(text, 8000)
    assert chunks == [prefix, "Intro\n- a\n- b"]
    assert not _is_list_line(_first_non_empty_line(chunks[1]))


def test_split_preserves_intro_label_with_colon():
    text = "Intro summary\n\nIntro:\n- a\n- b"
    limit = len("Intro summary")
    chunks = _split(text, limit)
    assert chunks == ["Intro summary", "Intro:\n- a\n- b"]


def test_rows_from_item_keeps_list_kind_metadata():
    item = {
        "text": "Intro\n- a\n- b",
        "meta": {
            "list_kind": "bullet",
        },
    }
    rows = _rows_from_item(item)
    assert rows
    assert {row["metadata"]["list_kind"] for row in rows} == {"bullet"}
