import pytest
from hypothesis import given, strategies as st
from pdf_chunker.list_detection import (
    BULLET_CHARS,
    is_bullet_list_pair,
    is_numbered_list_pair,
)


@pytest.mark.parametrize("bullet", ["\u2022", "*"])
def test_is_bullet_list_pair_with_colon(bullet):
    curr = f"Intro: {bullet} first"
    nxt = f"{bullet} second"
    assert is_bullet_list_pair(curr, nxt)


text_strategy = st.text(
    st.characters(blacklist_categories=("Cs",), blacklist_characters="\n"),
    min_size=1,
)
bullet_char_strategy = st.sampled_from(list(BULLET_CHARS) + ["-"])


@st.composite
def bullet_pairs(draw):
    bullet = draw(bullet_char_strategy)
    prefix, first, second = draw(st.tuples(text_strategy, text_strategy, text_strategy))
    bullet_line = lambda b, txt: f"{('- ' if b == '-' else b + ' ')}{txt}"
    curr = draw(
        st.one_of(
            st.just(bullet_line(bullet, first)),
            st.just(f"{prefix}: {bullet_line(bullet, first)}"),
            st.just(f"{prefix}:\n{bullet_line(bullet, first)}"),
        )
    )
    nxt = bullet_line(bullet, second)
    return curr, nxt


@given(bullet_pairs())
def test_is_bullet_list_pair_property(pair):
    curr, nxt = pair
    assert is_bullet_list_pair(curr, nxt)


number_strategy = st.integers(min_value=1, max_value=999)
delim_strategy = st.sampled_from([".", ")"])


@st.composite
def numbered_pairs(draw):
    n1, n2 = draw(st.tuples(number_strategy, number_strategy))
    delim = draw(delim_strategy)
    prefix, first, second = draw(st.tuples(text_strategy, text_strategy, text_strategy))
    num_line = lambda n, txt: f"{n}{delim} {txt}"
    curr = draw(
        st.one_of(
            st.just(num_line(n1, first)),
            st.just(f"{prefix}\n{num_line(n1, first)}"),
        )
    )
    nxt = num_line(n2, second)
    return curr, nxt


@given(numbered_pairs())
def test_is_numbered_list_pair_property(pair):
    curr, nxt = pair
    assert is_numbered_list_pair(curr, nxt)
