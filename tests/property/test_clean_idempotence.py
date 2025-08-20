from functools import reduce
from typing import Callable, TypeVar

from hypothesis import given, settings, strategies as st
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.text_clean import text_clean

T = TypeVar("T")


def _apply(times: int, fn: Callable[[T], T], value: T) -> T:
    return reduce(lambda acc, _: fn(acc), range(times), value)


alphabet = st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs"))


@given(st.text(alphabet=alphabet, max_size=200))
@settings(deadline=None)
def test_text_clean_idempotent(sample: str) -> None:
    artifact = Artifact(payload=sample)
    once = text_clean(artifact)
    twice = _apply(2, text_clean, artifact)
    assert twice.payload == once.payload
    assert twice.meta == once.meta
