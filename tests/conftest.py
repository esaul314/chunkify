from __future__ import annotations

from typing import Callable, Dict

import pytest

_COLOR_CODES: Dict[str, str] = {
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "bold": "1",
}


def _ansi(code: str) -> str:
    return f"\033[{code}m"


def _colorize(text: str, code: str) -> str:
    return f"{_ansi(code)}{text}\033[0m"


@pytest.fixture
def color() -> Callable[[str, str], str]:
    return lambda text, shade: (
        _colorize(text, _COLOR_CODES.get(shade, "")) if shade in _COLOR_CODES else text
    )


@pytest.fixture
def print_test_header(color: Callable[[str, str], str]) -> Callable[[str], None]:
    return lambda name: print(color(f"Testing {name}...", "blue"))


@pytest.fixture
def print_test_result(color: Callable[[str, str], str]) -> Callable[[str, str], None]:
    return lambda name, status: print(
        color(
            f"{name} testing {'completed successfully!' if status == 'success' else 'completed with issues.'}",
            "green" if status == "success" else "red",
        )
    )
