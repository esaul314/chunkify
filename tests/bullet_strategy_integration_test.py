"""Regression tests ensuring bullet strategy wiring across modules."""

from pdf_chunker import page_artifacts, text_cleaning
from pdf_chunker.strategies import bullets


def test_trailing_footer_respects_shared_bullet_strategy() -> None:
    lines = [
        "Support options:",
        "\u25e6 Need help?",
        "\u25e6 Contact us?",
    ]

    pruned = page_artifacts._drop_trailing_bullet_footers(lines)

    assert pruned == ["Support options:"]
    assert page_artifacts._BULLET_STRATEGY is bullets.default_bullet_strategy()


def test_remove_stray_bullet_lines_uses_shared_bullet_inventory() -> None:
    text = "Intro\n\u25e6 stray bullet\nNext"

    cleaned_once = text_cleaning.remove_stray_bullet_lines(text)
    cleaned_twice = text_cleaning.remove_stray_bullet_lines(text)

    assert "\u25e6" not in cleaned_once
    assert cleaned_once == cleaned_twice
    assert cleaned_once.endswith("Next")
