from typing import Callable, Dict
from pathlib import Path
import sys
import base64

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pdf_chunker.text_cleaning import (
    fix_hyphenated_linebreaks,
    normalize_ligatures,
    remove_underscore_emphasis,
)

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


_TEST_DATA = Path("test_data")

_CASES = (
    ("ligature.b64", normalize_ligatures, "Ligature fi and fl test"),
    ("underscore.b64", remove_underscore_emphasis, "leading bold trailing"),
    (
        "hyphenation.b64",
        fix_hyphenated_linebreaks,
        "This is a container with soft hyphen hy-phen",
    ),
)


@pytest.fixture(params=_CASES, ids=("ligature", "underscore", "hyphenation"))
def pdf_case(request):
    fitz = pytest.importorskip("fitz")
    filename, func, expected = request.param
    pdf_bytes = base64.b64decode((_TEST_DATA / filename).read_text())
    raw = "".join(page.get_text() for page in fitz.open(stream=pdf_bytes, filetype="pdf"))
    return raw, func, expected
