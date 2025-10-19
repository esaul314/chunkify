"""Regression tests ensuring bullet strategy wiring across modules."""

from pdf_chunker import page_artifacts, text_cleaning
from pdf_chunker.strategies import bullets


def test_trailing_footer_respects_shared_bullet_strategy() -> None:
    strategy = page_artifacts.bullet_strategy()
    marker = next(iter(strategy.bullet_chars))
    lines = [
        "Support options:",
        f"{marker}Need help?",
        f"{marker} Contact us?",
    ]

    pruned = page_artifacts._drop_trailing_bullet_footers(lines)

    assert pruned == ["Support options:"]
    assert page_artifacts.bullet_strategy() is bullets.default_bullet_strategy()


def test_remove_stray_bullet_lines_uses_shared_bullet_inventory() -> None:
    text = "Intro\n\u25e6 stray bullet\nNext"

    cleaned_once = text_cleaning.remove_stray_bullet_lines(text)
    cleaned_twice = text_cleaning.remove_stray_bullet_lines(text)

    assert "\u25e6" not in cleaned_once
    assert cleaned_once == cleaned_twice
    assert cleaned_once.endswith("Next")


def test_footer_context_strips_all_strategy_bullet_markers() -> None:
    strategy = page_artifacts.bullet_strategy()
    exotic_marker = next(
        marker for marker in strategy.bullet_chars if marker not in " -\u2022*"
    )
    footer_like = f"{exotic_marker} 2024"

    assert page_artifacts._looks_like_footer_context(footer_like)
