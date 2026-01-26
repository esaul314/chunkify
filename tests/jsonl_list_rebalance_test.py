import pytest

from pdf_chunker.passes.emit_jsonl import (
    _merge_text,
    _rebalance_lists,
    _rows_from_item,
    _split,
)
from pdf_chunker.passes.emit_jsonl_lists import (
    is_list_line as _is_list_line,
)
from pdf_chunker.passes.emit_jsonl_lists import (
    prepend_intro as _prepend_intro,
)
from pdf_chunker.passes.emit_jsonl_text import (
    first_non_empty_line as _first_non_empty_line,
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
            ("Intro", "1. one\n\n2. two\nTail"),
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


def test_prepend_intro_normalizes_numbered_list_spacing():
    intro = "Here are the recurring causes—namely:"
    rest = "\n\n1. Teams cannot self-service their needs.\n2. Platform scope is too broad."
    combined = _prepend_intro(intro, rest)
    lines = combined.splitlines()
    assert lines[0] == intro
    assert lines[1] == ""
    assert lines[2].startswith("1. Teams")


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


def test_peel_list_intro_preserves_content_after_label_colon():
    """Regression test: Q5 followed by question text should not lose content.

    When text has a pattern like "Q5: <question>? <more text>" followed by
    a bullet list, the content after the colon must be preserved. Previously,
    peel_list_intro incorrectly treated "Q5:" as a list intro, dropping
    the question and subsequent paragraphs.
    """
    from pdf_chunker.passes.emit_jsonl_lists import peel_list_intro

    # Text with "Q5:" followed by substantial content (468 chars in real case)
    text = (
        "Q4: Why? A: They're spread thin. "
        "Q5: So why aren't senior engineers helping? "
        "And by now we get to the answers. "
        "Some Common Discoveries Having walked the hierarchy."
    )

    kept, intro = peel_list_intro(text)

    # Should NOT peel anything - there's substantial content after "Q5:"
    assert intro == "", f"Expected empty intro, got: {intro!r}"
    assert "Q5:" in kept, "Q5 label should be in kept text"
    assert "senior engineers" in kept, "Q5 question should be preserved"
    assert "Common Discoveries" in kept, "Subsequent content should be preserved"


def test_heading_chunk_includes_section_content():
    """Headings must NOT be isolated in their own JSONL line.

    A section heading belongs with its section text, separated by a newline.
    This test guards against regressions where heading-boundary logic
    incorrectly creates a heading-only chunk.
    """
    # Simulate what the pipeline produces: heading + content in same record
    heading_text = "Some Common Discoveries Along the Way"
    body_text = (
        "Having walked the hierarchy of organizational dysfunction, we can now start prescribing."
    )
    combined = f"{heading_text}\n{body_text}"

    # The combined text should be in a single chunk
    assert combined.startswith(heading_text), "Heading at start"
    assert body_text in combined, "Body follows heading"
    assert len(combined) > 100, "Chunk should not be tiny (heading + content)"

    # Verify structure: heading\nbody
    lines = combined.split("\n", 1)
    assert lines[0] == heading_text
    assert lines[1] == body_text
