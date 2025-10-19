"""Predicate coverage for :mod:`pdf_chunker.strategies.bullets`.

Expected marker inventory:
* ``bullet_chars`` => "*•◦▪‣·●◉○‧"
* ``hyphen_bullet_prefix`` => "- "
* ``numbered_pattern`` => ``(?P<number>\\d+)[.)]\\s+(?P<body>.+)``
"""

from __future__ import annotations

import pytest

from pdf_chunker.strategies.bullets import default_bullet_strategy


_STRATEGY = default_bullet_strategy()


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        pytest.param("• Point", True, id="solid-bullet"),
        pytest.param("- Hyphenated", True, id="hyphen-prefix"),
        pytest.param("   ▪ Nested", True, id="indented"),
        pytest.param("Paragraph", False, id="plain"),
        pytest.param("1) Numbered", False, id="numbered-not-bullet"),
    ),
)
def test_starts_with_bullet_respects_inventory(text: str, expected: bool) -> None:
    """Default bullet characters and hyphen prefix drive leading detection."""

    assert _STRATEGY.starts_with_bullet(text) is expected


@pytest.mark.parametrize(
    ("text", "head", "tail"),
    (
        pytest.param("• Item\n continuation", "• Item", "continuation", id="multiline"),
        pytest.param("• Solo", "• Solo", "", id="single-line"),
        pytest.param("• Marker\n\n  body", "• Marker", "body", id="blank-line-gap"),
    ),
)
def test_split_bullet_fragment_preserves_marker_head(
    text: str, head: str, tail: str
) -> None:
    """Fragment splitting keeps bullet heads intact for every marker variant."""

    assert _STRATEGY.split_bullet_fragment(text) == (head, tail)


@pytest.mark.parametrize(
    ("current", "following", "fragment", "continuation"),
    (
        pytest.param("• Combine ingredients", "mix thoroughly", True, False, id="inline-body"),
        pytest.param("Support options:\n•", "contact support", True, True, id="dangling-marker"),
        pytest.param("• Complete item.", "Next entry", False, False, id="terminal-period"),
        pytest.param("Paragraph", "standalone", False, False, id="plain-text"),
    ),
)
def test_fragment_and_continuation_detections_are_consistent(
    current: str, following: str, fragment: bool, continuation: bool
) -> None:
    """Bullet fragment and continuation predicates align with last-line heuristics."""

    assert _STRATEGY.is_bullet_fragment(current, following) is fragment
    assert _STRATEGY.is_bullet_continuation(current, following) is continuation


@pytest.mark.parametrize(
    ("leader", "next_line", "expected"),
    (
        pytest.param("Agenda:\n• One", "• Two", True, id="colon-leads"),
        pytest.param("Agenda:", "• Two", False, id="colon-alone"),
        pytest.param("Paragraph", "• Two", False, id="no-bullet-context"),
        pytest.param("• One", "Paragraph", False, id="next-not-bullet"),
    ),
)
def test_bullet_list_pair_detection_matches_context(
    leader: str, next_line: str, expected: bool
) -> None:
    """Bullet list pairing looks for inline markers and colon led sequences."""

    assert _STRATEGY.is_bullet_list_pair(leader, next_line) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        pytest.param("1. Start", True, id="dot"),
        pytest.param("2) Continue", True, id="paren"),
        pytest.param("1 Step", False, id="missing-marker"),
        pytest.param("Step 1", False, id="trailing-digit"),
    ),
)
def test_starts_with_number_matches_numbered_pattern(text: str, expected: bool) -> None:
    """Numbered predicate mirrors the configured numbering regular expression."""

    assert _STRATEGY.starts_with_number(text) is expected


@pytest.mark.parametrize(
    ("current", "following", "pair", "continuation"),
    (
        pytest.param("1. Start", "2. Next", True, False, id="adjacent-numbers"),
        pytest.param("Topic\n1. Start", "2. Next", True, False, id="inline-number"),
        pytest.param("Paragraph", "2. Next", False, False, id="no-leading-number"),
        pytest.param("1. First part", "continues detail", False, True, id="continuation"),
    ),
)
def test_numbered_pair_and_continuation_logic(
    current: str, following: str, pair: bool, continuation: bool
) -> None:
    """Numbered pairing and continuation reuse the shared numbering inventory."""

    assert _STRATEGY.is_numbered_list_pair(current, following) is pair
    assert _STRATEGY.is_numbered_continuation(current, following) is continuation
