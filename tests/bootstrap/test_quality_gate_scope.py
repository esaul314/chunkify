from noxfile import (
    LINT_EXTRA_PATHS,
    QUALITY_PATHS,
    _lint_commands,
    _typecheck_commands,
)


def test_quality_paths_include_canonical_boundary_modules() -> None:
    assert {
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
    } <= set(QUALITY_PATHS)


def test_lint_commands_are_derived_from_declared_paths() -> None:
    expected_paths = (*QUALITY_PATHS, *LINT_EXTRA_PATHS)
    ruff, black = _lint_commands()

    assert ruff == ("ruff", "check", "--no-fix", *expected_paths)
    assert black == ("black", "--check", *expected_paths)


def test_typecheck_command_is_derived_from_quality_paths() -> None:
    assert _typecheck_commands() == (("mypy", *QUALITY_PATHS),)
