from typing import Any, Callable, Dict
from pathlib import Path
import sys
import base64
import ssl

import nltk
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _download_if_missing(resource: str, name: str) -> None:
    try:
        nltk.data.find(resource)
    except LookupError:
        ssl._create_default_https_context = ssl._create_unverified_context
        nltk.download(name, quiet=True)


def _ensure_nltk_resources() -> None:
    tuple(
        _download_if_missing(res, name)
        for res, name in (("corpora/cmudict", "cmudict"), ("tokenizers/punkt", "punkt"))
    )


@pytest.fixture(scope="session", autouse=True)
def _nltk_data() -> None:
    _ensure_nltk_resources()

from pdf_chunker.text_cleaning import (
    fix_hyphenated_linebreaks,
    normalize_ligatures,
    remove_underscore_emphasis,
)
from pdf_chunker.pdf_blocks import Block

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
def block() -> Callable[[str, Dict[str, Any]], Block]:
    return lambda text, **source: Block(text=text, source=source)


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
