"""Nox sessions for linting, type checking, and testing."""

from pathlib import Path

import nox

PY_TARGETS = ["pdf_chunker/__init__.py", "noxfile.py", "tests/bootstrap"]


@nox.session
def lint(session):
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "--fix", *PY_TARGETS)
    session.run("black", "--check", *PY_TARGETS)


@nox.session
def typecheck(session):
    session.install("-e", ".[dev]")
    targets = [t for t in ["pdf_chunker/__init__.py"] if Path(t).exists()]
    if targets:
        session.run("mypy", "--allow-untyped-globals", *targets)
    else:
        session.log("No typecheck targets yet.")


@nox.session
def tests(session):
    session.install("-e", ".[dev]")
    paths = (p.as_posix() for p in [Path("tests")] if p.exists())
    session.run("pytest", "-q", *paths)
