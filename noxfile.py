"""Nox sessions for linting, type checking, and testing."""

from collections.abc import Iterable
from pathlib import Path
from typing import Final

import nox

Command = tuple[str, ...]

ROOT: Final = Path(__file__).parent
SUPPORTED_PYTHON: Final = "3.11"
SESSION_DEPENDENCIES: Final = ("-e", ".[dev]")
TEST_SESSION_DEPENDENCIES: Final = ("-e", ".[dev,enrichment]")
QUALITY_PATHS: Final = (
    "pdf_chunker/__init__.py",
    "pdf_chunker/contracts.py",
    "pdf_chunker/framework.py",
    "pdf_chunker/core_new.py",
    "pdf_chunker/env_utils.py",
    "pdf_chunker/page_utils.py",
    "pdf_chunker/fallbacks.py",
    "pdf_chunker/evaluation.py",
    "pdf_chunker/language.py",
    "pdf_chunker/source_matchers.py",
    "pdf_chunker/patterns.py",
    "pdf_chunker/list_detection.py",
    "pdf_chunker/epub_parsing.py",
    "pdf_chunker/adapters/io_epub.py",
    "pdf_chunker/adapters/io_text.py",
    "pdf_chunker/passes/ai_enrich.py",
    "pdf_chunker/passes/split_modules/segments.py",
    "pdf_chunker/passes/pdf_parse.py",
    "pdf_chunker/passes/extraction_fallback.py",
    "pdf_chunker/passes/epub_parse.py",
    "pdf_chunker/passes/text_parse.py",
    "pdf_chunker/passes/semantic_dedupe.py",
    "pdf_chunker/passes/transform_log.py",
    "pdf_chunker/passes/chunk_options.py",
    "pdf_chunker/passes/chunk_pipeline.py",
    "pdf_chunker/passes/sentence_fusion.py",
    "pdf_chunker/passes/metrics.py",
    "pdf_chunker/passes/heading_detect.py",
    "pdf_chunker/passes/list_detect.py",
    "pdf_chunker/passes/merge_footers.py",
    "pdf_chunker/passes/detect_doc_end.py",
    "pdf_chunker/passes/emit_jsonl.py",
    "pdf_chunker/passes/split_semantic.py",
)
LINT_EXTRA_PATHS: Final = (
    "noxfile.py",
    "tests/bootstrap",
    "tests/io_epub_adapter_test.py",
    "tests/pdf_parse_pass_test.py",
    "tests/utils_test.py",
    "tests/chunk_pipeline_test.py",
    "tests/sentence_fusion_test.py",
    "tests/structure_pass_purity_test.py",
    "tests/numbered_list_cross_page_chunk_test.py",
)
TEST_TARGETS: Final = ("tests",)
TYPECHECK_EMPTY_MESSAGE: Final = "No typecheck targets yet."

# Missing runtimes must not turn skipped checks into a successful baseline.
nox.options.error_on_missing_interpreters = True


def _existing(paths: Iterable[str]) -> tuple[str, ...]:
    """Return the subset of ``paths`` that exist, preserving declaration order."""

    return tuple(path for path in dict.fromkeys(paths) if (ROOT / path).exists())


def _lint_commands() -> tuple[Command, ...]:
    targets = _existing((*QUALITY_PATHS, *LINT_EXTRA_PATHS))
    return (
        ("ruff", "check", "--no-fix", *targets),
        ("black", "--check", *targets),
    )


def _typecheck_commands() -> tuple[Command, ...]:
    targets = _existing(QUALITY_PATHS)
    return (("mypy", *targets),) if targets else ()


def _test_commands() -> tuple[Command, ...]:
    targets = _existing(TEST_TARGETS)
    return (("pytest", "-q", "-m", "not legacy_parity", *targets),) if targets else ()


def _run(
    session: nox.Session,
    commands: tuple[Command, ...],
    *,
    dependencies: tuple[str, ...] = SESSION_DEPENDENCIES,
    empty_message: str | None = None,
) -> None:
    """Install shared dependencies and execute ``commands`` within ``session``."""

    session.install(*dependencies)
    if not commands:
        if empty_message is not None:
            session.log(empty_message)
        return
    for command in commands:
        session.run(*command)


@nox.session(python=SUPPORTED_PYTHON)
def lint(session: nox.Session) -> None:
    """Run static analysis via Ruff and Black."""

    _run(session, _lint_commands())


@nox.session(python=SUPPORTED_PYTHON)
def typecheck(session: nox.Session) -> None:
    """Run mypy for the declared production quality paths."""

    _run(session, _typecheck_commands(), empty_message=TYPECHECK_EMPTY_MESSAGE)


@nox.session(python=SUPPORTED_PYTHON)
def tests(session: nox.Session) -> None:
    """Execute the pytest suite."""

    _run(session, _test_commands(), dependencies=TEST_SESSION_DEPENDENCIES)
