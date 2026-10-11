"""Guard sentence-fusion budgets and the pure coalescing fold."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker.passes.sentence_fusion import (
    _BudgetView,
    _coalesce_sentence_runs,
    _MergeBudget,
)


@pytest.mark.parametrize(
    ("target", "hard", "words", "effective", "dense", "pending", "expected"),
    (
        (None, None, 2, 20, True, True, False),
        (10, 20, 10, 10, False, False, False),
        (10, 20, 11, 11, False, False, True),
        (10, 20, 11, 11, True, False, True),
        (10, 20, 20, 20, True, False, True),
        (10, 20, 21, 21, False, False, False),
        (10, 20, 2, 25, True, False, False),
        (10, 20, 2, 25, True, True, True),
        (10, 20, 21, 25, True, True, False),
        (10, None, 2, 25, True, False, False),
        (10, None, 2, 25, True, True, True),
        (10, None, 12, 25, False, False, True),
    ),
)
def test_overflow_permission_preserves_budget_boundaries(
    target: int | None,
    hard: int | None,
    words: int,
    effective: int,
    dense: bool,
    pending: bool,
    expected: bool,
) -> None:
    budget = _MergeBudget(target, hard, words, effective, effective)
    view = _BudgetView(budget, target, dense, pending)

    assert view.allow_overflow() is expected
    assert view.budget == budget


@pytest.mark.parametrize(
    ("chunks", "expected"),
    (
        ((), []),
        (("Only fragment",), ["Only fragment"]),
        (("First", "unfinished"), ["First", "unfinished"]),
        (("First", "completed."), ["First completed."]),
        (("First.", "Second."), ["First.", "Second."]),
        ((" First ", "\ncompleted! ", "Tail"), ["First completed!", "Tail"]),
        (
            ("First", "completed?", "Trailing", "fragments"),
            ["First completed?", "Trailing", "fragments"],
        ),
        (("First", 'completed."', "Tail"), ['First completed."', "Tail"]),
    ),
)
def test_coalescing_flushes_completed_runs_and_keeps_unfinished_tail(
    chunks: tuple[str, ...], expected: list[str]
) -> None:
    assert _coalesce_sentence_runs(iter(chunks)) == expected


@given(st.lists(st.text(alphabet="abc .!?\n", max_size=40), max_size=12))
def test_coalescing_preserves_word_order_and_does_not_mutate_input(chunks: list[str]) -> None:
    original = tuple(chunks)
    result = _coalesce_sentence_runs(iter(chunks))

    assert " ".join(result).split() == " ".join(original).split()
    assert tuple(chunks) == original
    assert _coalesce_sentence_runs(iter(original)) == result
