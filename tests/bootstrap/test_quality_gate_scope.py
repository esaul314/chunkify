"""Guard the quality scope, interpreter policy, and production syntax."""

import ast
from pathlib import Path

import nox
import pytest
import yaml

import noxfile
from noxfile import (
    LINT_EXTRA_PATHS,
    QUALITY_PATHS,
    _lint_commands,
    _typecheck_commands,
)

ROOT = Path(__file__).parents[2]


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
        "pdf_chunker/adapters/io_epub.py",
        "pdf_chunker/adapters/io_text.py",
        "pdf_chunker/passes/pdf_parse.py",
        "pdf_chunker/passes/extraction_fallback.py",
        "pdf_chunker/passes/epub_parse.py",
        "pdf_chunker/passes/text_parse.py",
        "pdf_chunker/passes/semantic_dedupe.py",
        "pdf_chunker/passes/transform_log.py",
        "pdf_chunker/passes/chunk_options.py",
        "pdf_chunker/passes/chunk_pipeline.py",
        "pdf_chunker/passes/sentence_fusion.py",
        "pdf_chunker/passes/heading_detect.py",
        "pdf_chunker/passes/list_detect.py",
        "pdf_chunker/passes/merge_footers.py",
        "pdf_chunker/passes/detect_doc_end.py",
        "pdf_chunker/passes/detect_page_artifacts.py",
        "pdf_chunker/passes/metrics.py",
    } <= set(QUALITY_PATHS)


def test_lint_commands_are_derived_from_declared_paths() -> None:
    expected_paths = (*QUALITY_PATHS, *LINT_EXTRA_PATHS)
    ruff, black = _lint_commands()

    assert ruff == ("ruff", "check", "--no-fix", *expected_paths)
    assert black == ("black", "--check", *expected_paths)


def test_typecheck_command_is_derived_from_quality_paths() -> None:
    assert _typecheck_commands() == (("mypy", *QUALITY_PATHS),)


def test_semantic_helper_tests_are_lint_checked() -> None:
    assert {
        "tests/chunk_pipeline_test.py",
        "tests/sentence_fusion_test.py",
    } <= set(LINT_EXTRA_PATHS)


def test_structure_pass_tests_are_lint_checked() -> None:
    assert "tests/structure_pass_purity_test.py" in LINT_EXTRA_PATHS


def test_page_artifact_pass_tests_are_lint_checked() -> None:
    assert "tests/page_artifact_pass_test.py" in LINT_EXTRA_PATHS


@pytest.mark.parametrize("session_name", ("lint", "typecheck", "tests"))
def test_quality_sessions_request_supported_python(session_name: str) -> None:
    assert getattr(noxfile, session_name).python == "3.11"


def test_quality_sessions_configure_missing_interpreter_as_error() -> None:
    assert nox.options.error_on_missing_interpreters is True


def test_quality_interpreter_matches_ci_jobs() -> None:
    workflow_path = ROOT / ".github/workflows/ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    versions = {
        step["with"]["python-version"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    }

    assert versions == {noxfile.SUPPORTED_PYTHON}


@pytest.mark.parametrize(
    "source_path",
    sorted((ROOT / "pdf_chunker").rglob("*.py")),
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_production_sources_parse_on_session_interpreter(source_path: Path) -> None:
    ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
