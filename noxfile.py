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
    "pdf_chunker/conversion_descriptor.py",
    "pdf_chunker/article_conversion.py",
    "pdf_chunker/adapters/article_conversion.py",
    "pdf_chunker/framework.py",
    "pdf_chunker/core_new.py",
    "pdf_chunker/env_utils.py",
    "pdf_chunker/page_utils.py",
    "pdf_chunker/fallbacks.py",
    "pdf_chunker/evaluation.py",
    "pdf_chunker/language.py",
    "pdf_chunker/geometry.py",
    "pdf_chunker/source_matchers.py",
    "pdf_chunker/patterns.py",
    "pdf_chunker/list_detection.py",
    "pdf_chunker/epub_parsing.py",
    "pdf_chunker/adapters/io_epub.py",
    "pdf_chunker/adapters/io_joplin_epub.py",
    "pdf_chunker/passes/joplin_markdown.py",
    "pdf_chunker/adapters/io_text.py",
    "pdf_chunker/adapters/request_budget.py",
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
    "pdf_chunker/passes/detect_page_artifacts.py",
    "pdf_chunker/passes/text_clean.py",
    "pdf_chunker/passes/emit_jsonl.py",
    "pdf_chunker/passes/split_semantic.py",
)
LINT_EXTRA_PATHS: Final = (
    "noxfile.py",
    "tests/bootstrap",
    "tests/io_epub_adapter_test.py",
    "tests/joplin_markdown_preservation_test.py",
    "tests/verified_markdown_ingress_test.py",
    "tests/conversion_descriptor_test.py",
    "tests/article_conversion_test.py",
    "tests/epub_body_markup_test.py",
    "tests/pdf_parse_pass_test.py",
    "tests/utils_test.py",
    "tests/chunk_pipeline_test.py",
    "tests/sentence_fusion_test.py",
    "tests/structure_pass_purity_test.py",
    "tests/page_artifact_pass_test.py",
    "tests/text_clean_purity_test.py",
    "tests/geometry_contract_test.py",
    "tests/numbered_list_cross_page_chunk_test.py",
    "tests/ai_enrich_request_budget_test.py",
    "tests/ai_enrich_context_fixtures_test.py",
)
TEST_TARGETS: Final = ("tests",)

# Missing runtimes must not turn skipped checks into a successful baseline.
nox.options.error_on_missing_interpreters = True


def _existing(paths: Iterable[str], *, scope: str = "quality") -> tuple[str, ...]:
    """Require every declared target instead of silently narrowing the gate.

    Consume the input once and deduplicate in first-seen order. Empty declarations
    and blank paths are invalid; nonblank path spelling is preserved. Report all
    missing paths in the group before any command is built. Filesystem checks stay
    in this tooling boundary, not in document-processing passes.
    """

    targets = tuple(dict.fromkeys(paths))
    if not targets:
        raise ValueError(f"No targets declared for {scope}.")
    if any(not path.strip() for path in targets):
        raise ValueError(f"Blank targets declared for {scope}.")
    missing = tuple(path for path in targets if not (ROOT / path).exists())
    if missing:
        raise FileNotFoundError(f"Missing declared {scope} targets: {', '.join(missing)}.")
    return targets


def _lint_commands() -> tuple[Command, ...]:
    production = _existing(QUALITY_PATHS, scope="production quality")
    extras = _existing(LINT_EXTRA_PATHS, scope="lint-only")
    targets = tuple(dict.fromkeys((*production, *extras)))
    return (
        ("ruff", "check", "--no-fix", *targets),
        ("black", "--check", *targets),
    )


def _typecheck_commands() -> tuple[Command, ...]:
    targets = _existing(QUALITY_PATHS, scope="production quality")
    return (("mypy", *targets),)


def _test_commands() -> tuple[Command, ...]:
    targets = _existing(TEST_TARGETS, scope="tests")
    return (("pytest", "-q", "-m", "not legacy_parity", *targets),)


def _run(
    session: nox.Session,
    commands: tuple[Command, ...],
    *,
    dependencies: tuple[str, ...] = SESSION_DEPENDENCIES,
) -> None:
    """Install dependencies and execute commands after target validation."""

    session.install(*dependencies)
    for command in commands:
        session.run(*command)


@nox.session(python=SUPPORTED_PYTHON)
def lint(session: nox.Session) -> None:
    """Run static analysis via Ruff and Black."""

    _run(session, _lint_commands())


@nox.session(python=SUPPORTED_PYTHON)
def typecheck(session: nox.Session) -> None:
    """Run mypy for the declared production quality paths."""

    _run(session, _typecheck_commands())


@nox.session(python=SUPPORTED_PYTHON)
def tests(session: nox.Session) -> None:
    """Execute the pytest suite."""

    _run(session, _test_commands(), dependencies=TEST_SESSION_DEPENDENCIES)
