"""Nox sessions for linting, type checking, and testing."""

from pathlib import Path

import nox

LINT_TARGETS = [
    "noxfile.py",
    "pdf_chunker/__init__.py",
    "pdf_chunker/passes/emit_jsonl.py",
    "pdf_chunker/passes/split_semantic.py",
    "tests/bootstrap",
]

TYPECHECK_TARGETS = [
    "pdf_chunker/__init__.py",
    "pdf_chunker/passes/__init__.py",
    "pdf_chunker/passes/emit_jsonl.py",
    "pdf_chunker/passes/extraction_fallback.py",
    "pdf_chunker/passes/heading_detect.py",
    "pdf_chunker/passes/list_detect.py",
    "pdf_chunker/passes/pdf_parse.py",
    "pdf_chunker/passes/split_semantic.py",
    "pdf_chunker/passes/text_clean.py",
]


@nox.session
def lint(session):
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "--fix", *LINT_TARGETS)
    session.run("black", "--check", *LINT_TARGETS)


@nox.session
def typecheck(session):
    session.install("-e", ".[dev]")
    targets = [t for t in TYPECHECK_TARGETS if Path(t).exists()]
    if targets:
        session.run("mypy", "--allow-untyped-globals", *targets)
    else:
        session.log("No typecheck targets yet.")


@nox.session
def tests(session):
    session.install("-e", ".[dev]")
    paths = (p.as_posix() for p in [Path("tests")] if p.exists())
    session.run("pytest", "-q", *paths)
