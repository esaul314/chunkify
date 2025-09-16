"""Nox sessions for linting, type checking, and testing."""

from collections.abc import Iterable
from pathlib import Path
from typing import Final

import nox

Command = tuple[str, ...]

ROOT: Final = Path(__file__).parent
SESSION_DEPENDENCIES: Final = ("-e", ".[dev]")
TYPECHECK_EMPTY_MESSAGE: Final = "No typecheck targets yet."

LINT_TARGETS: Final = (
    "noxfile.py",
    "pdf_chunker/__init__.py",
    "pdf_chunker/passes/emit_jsonl.py",
    "pdf_chunker/passes/split_semantic.py",
    "tests/bootstrap",
)

TYPECHECK_TARGETS: Final = (
    "pdf_chunker/__init__.py",
    "pdf_chunker/passes/__init__.py",
    "pdf_chunker/passes/emit_jsonl.py",
    "pdf_chunker/passes/extraction_fallback.py",
    "pdf_chunker/passes/heading_detect.py",
    "pdf_chunker/passes/list_detect.py",
    "pdf_chunker/passes/pdf_parse.py",
    "pdf_chunker/passes/split_semantic.py",
    "pdf_chunker/passes/text_clean.py",
)

TEST_TARGETS: Final = ("tests",)


def _existing(paths: Iterable[str]) -> tuple[str, ...]:
    """Return the subset of ``paths`` that exist, preserving declaration order."""

    return tuple(path for path in dict.fromkeys(paths) if (ROOT / path).exists())


def _lint_commands() -> tuple[Command, ...]:
    targets = _existing(LINT_TARGETS)
    return (
        ("ruff", "check", "--fix", *targets),
        ("black", "--check", *targets),
    )


def _typecheck_commands() -> tuple[Command, ...]:
    targets = _existing(TYPECHECK_TARGETS)
    return (("mypy", "--allow-untyped-globals", *targets),) if targets else ()


def _test_commands() -> tuple[Command, ...]:
    targets = _existing(TEST_TARGETS)
    return (("pytest", "-q", *targets),) if targets else ()


def _run(
    session: nox.Session,
    commands: tuple[Command, ...],
    *,
    empty_message: str | None = None,
) -> None:
    """Install shared dependencies and execute ``commands`` within ``session``."""

    session.install(*SESSION_DEPENDENCIES)
    if not commands:
        if empty_message is not None:
            session.log(empty_message)
        return
    for command in commands:
        session.run(*command)


@nox.session
def lint(session: nox.Session) -> None:
    """Run static analysis via Ruff and Black."""

    _run(session, _lint_commands())


@nox.session
def typecheck(session: nox.Session) -> None:
    """Run mypy for the modules that currently have coverage."""

    _run(session, _typecheck_commands(), empty_message=TYPECHECK_EMPTY_MESSAGE)


@nox.session
def tests(session: nox.Session) -> None:
    """Execute the pytest suite."""

    _run(session, _test_commands())
