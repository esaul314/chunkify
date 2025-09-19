"""Nox automation sessions for pdf_chunker."""

from __future__ import annotations

from pathlib import Path

import nox

nox.options.sessions = ("lint", "typecheck", "tests")
nox.options.reuse_existing_virtualenvs = True

PROJECT_ROOT = Path(__file__).parent


def _install_requirements(session: nox.Session) -> None:
    requirements = PROJECT_ROOT / "requirements.txt"
    if requirements.exists():
        session.install("-r", str(requirements))


@nox.session()
def lint(session: nox.Session) -> None:
    session.install("black", "flake8")
    session.run(
        "black",
        "--check",
        "--extend-exclude",
        "pdf_chunker/text_cleaning.py",
        "pdf_chunker",
        "scripts",
        "tests",
    )
    session.run("flake8", "pdf_chunker", "scripts", "tests")


@nox.session()
def typecheck(session: nox.Session) -> None:
    session.install("mypy")
    _install_requirements(session)
    session.run("mypy", "pdf_chunker")


@nox.session()
def tests(session: nox.Session) -> None:
    _install_requirements(session)
    session.install("pytest")
    session.run("pytest", "tests")
