import sys

sys.path.insert(0, ".")

from pdf_chunker.list_detection import (
    is_bullet_fragment,
    split_bullet_fragment,
)


def test_bullet_fragment_with_leading_marker():
    curr = "\u2022 What are your users doing? (And, ideally, what do they think they're trying"
    nxt = "\u2022 to do?) \u2022 How is the platform performing when they do that?"
    assert is_bullet_fragment(curr, nxt)
    frag, rest = split_bullet_fragment(nxt)
    assert frag == "to do?)"
    assert rest.startswith("\u2022 How is the platform performing")
