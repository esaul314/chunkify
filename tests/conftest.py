from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence
from pathlib import Path
import sys
import base64
import ssl
import warnings

import nltk
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _pymupdf_deprecation_messages() -> tuple[str, ...]:
    return (
        r"builtin type SwigPyPacked has no __module__ attribute",
        r"builtin type SwigPyObject has no __module__ attribute",
        r"builtin type swigvarlink has no __module__ attribute",
    )


def _suppress_deprecations(messages: Iterable[str]) -> None:
    tuple(
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=DeprecationWarning,
        )
        for message in messages
    )


_suppress_deprecations(_pymupdf_deprecation_messages())


try:  # Optional regression plugin used by a subset of golden tests.
    import pytest_regressions  # type: ignore  # noqa: F401
except ImportError:
    _PYTEST_REGRESSIONS_AVAILABLE = False
else:
    _PYTEST_REGRESSIONS_AVAILABLE = True


def _skip_optional_regression(request_path: Path) -> Optional[str]:
    """Return a skip reason for optional regression tests missing dependencies."""

    optional_targets: Iterable[tuple[Path, str]] = (
        (
            Path("tests/golden/test_conversion.py"),
            (
                "pytest-regressions is required for golden conversion checks. "
                "Install it via `pip install .[dev]` to enable the file_regression fixture."
            ),
        ),
    )
    return next(
        (
            reason
            for target, reason in optional_targets
            if request_path.as_posix().endswith(target.as_posix())
        ),
        None,
    )


if not _PYTEST_REGRESSIONS_AVAILABLE:

    @pytest.fixture
    def file_regression(request: pytest.FixtureRequest):  # type: ignore[name-defined]
        reason = _skip_optional_regression(Path(str(request.node.fspath)))
        if reason is not None:
            pytest.skip(reason)
        pytest.skip(
            "pytest-regressions is not installed; install the optional dependency to use file_regression."
        )


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


@dataclass(frozen=True)
class _KnownIssue:
    node_suffix: str
    reason: str


def _known_issue_catalog() -> Sequence[_KnownIssue]:
    return ()


def _known_issue_lookup(catalog: Sequence[_KnownIssue]) -> Mapping[str, str]:
    return {issue.node_suffix: issue.reason for issue in catalog}


def _xfail_known_issue(item: pytest.Item, lookup: Mapping[str, str]) -> None:
    reason = next(
        (
            lookup[suffix]
            for suffix in lookup
            if item.nodeid.endswith(suffix)
        ),
        None,
    )
    if reason is not None:
        item.add_marker(
            pytest.mark.xfail(
                reason=f"{reason} (see tests/KNOWN_ISSUES.md)",
                strict=False,
            )
        )


def pytest_collection_modifyitems(config: pytest.Config, items: Sequence[pytest.Item]) -> None:
    catalog = _known_issue_catalog()
    if not catalog:
        return
    lookup = _known_issue_lookup(catalog)
    tuple(_xfail_known_issue(item, lookup) for item in items)

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
