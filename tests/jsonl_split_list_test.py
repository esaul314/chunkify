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
