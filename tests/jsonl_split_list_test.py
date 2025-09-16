import pytest

from pdf_chunker.passes.emit_jsonl import _split


@pytest.mark.parametrize(
    "items, limit",
    [
        (["- a", "- b"], len("Intro\n- a\n")),
        (["1. one", "2. two"], len("Intro\n1. one\n")),
    ],
)
def test_split_preserves_lists(items, limit):
    text = "Intro\n" + "\n".join(items) + "\nTail"
    expected = ["Intro", "\n".join(items) + "\nTail"]
    assert _split(text, limit) == expected


def test_split_numbered_list_with_blank_line():
    text = "Intro\n1. one\n\n2. two\n3. three\nTail"
    limit = len("Intro\n1. one\n")
    expected = ["Intro", "1. one\n\n2. two\n3. three\nTail"]
    assert _split(text, limit) == expected


def test_split_bullet_list_with_blank_line():
    text = "Intro\n- a\n\n- b\n- c\nTail"
    limit = len("Intro\n- a\n")
    expected = ["Intro", "- a\n\n- b\n- c\nTail"]
    assert _split(text, limit) == expected


def test_split_numbered_list_small_limit():
    text = "Intro\n1. one\n2. two"
    assert _split(text, 5) == ["Intro", "1. one\n2. two"]
    

def test_split_keeps_intro_with_list_and_no_empty_chunks():
    prefix = "Lead paragraph describing the upcoming bullets."
    text = f"{prefix}\n\nIntro\n1. one\n2. two\nTail"
    limit = len(prefix)
    chunks = _split(text, limit)
    assert chunks[0] == prefix
    assert chunks[1].startswith("Intro\n1. one")
    assert "" not in chunks
