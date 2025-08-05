import pytest
from pdf_chunker.list_detection import is_bullet_list_pair


@pytest.mark.parametrize("bullet", ["\u2022", "*"])
def test_is_bullet_list_pair_with_colon(bullet):
    curr = f"Intro: {bullet} first"
    nxt = f"{bullet} second"
    assert is_bullet_list_pair(curr, nxt)
