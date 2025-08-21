"""Nox sessions scoped to bootstrap code for Story A.

This temporary narrowing avoids legacy modules until they are
refactored in later stories.
"""

import os

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
    targets = [t for t in ["pdf_chunker/__init__.py"] if os.path.exists(t)]
    if targets:
        session.run("mypy", "--allow-untyped-globals", *targets)
    else:
        session.log("No typecheck targets yet.")


@nox.session
def tests(session):
    session.install("-e", ".[dev]")
    paths = (
        f"tests/{suite}"
        for suite in ("bootstrap", "golden", "parity")
        if os.path.exists(f"tests/{suite}")
    )
    session.run("pytest", "-q", *paths)
