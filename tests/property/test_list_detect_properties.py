from hypothesis import given, settings, strategies as st

from pdf_chunker.framework import Artifact
from pdf_chunker.passes.list_detect import BULLET_CHARS, list_detect


alpha = st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs"))
text = st.text(alphabet=alpha, min_size=1)

bullet_marker = st.sampled_from(tuple(BULLET_CHARS) + ("-",))
bullet_items = st.builds(lambda b, t: f"{b} {t}", bullet_marker, text)

number_prefix = st.integers(min_value=1, max_value=9999)
number_suffix = st.sampled_from([".", ")"])
number_items = st.builds(lambda n, s, t: f"{n}{s} {t}", number_prefix, number_suffix, text)


def _first_block(content: str) -> dict:
    doc = {"type": "page_blocks", "pages": [{"page": 1, "blocks": [{"text": content}]}]}
    return list_detect(Artifact(payload=doc)).payload["pages"][0]["blocks"][0]


@given(bullet_items)
@settings(deadline=None)
def test_bullet_items_classified(sample: str) -> None:
    block = _first_block(sample)
    assert block["type"] == "list_item"
    assert block["list_kind"] == "bullet"


@given(number_items)
@settings(deadline=None)
def test_numbered_items_classified(sample: str) -> None:
    block = _first_block(sample)
    assert block["type"] == "list_item"
    assert block["list_kind"] == "numbered"
