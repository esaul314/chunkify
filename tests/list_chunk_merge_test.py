import sys

sys.path.insert(0, ".")

from pdf_chunker.splitter import _merge_standalone_lists


def test_merge_bullet_and_numbered_lists():
    cases = [
        (["Intro", "• a", "• b", "End"], ["Intro\n• a\n• b", "End"]),
        (["Intro", "1. a", "2. b", "End"], ["Intro\n1. a\n2. b", "End"]),
    ]
    for chunks, expected in cases:
        assert _merge_standalone_lists(chunks) == expected
