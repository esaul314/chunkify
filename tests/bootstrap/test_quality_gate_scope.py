"""Guard the quality scope, interpreter policy, and production syntax."""

import ast
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import nox
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

import noxfile
from noxfile import (
    LINT_EXTRA_PATHS,
    QUALITY_PATHS,
    _lint_commands,
    _typecheck_commands,
)

ROOT = Path(__file__).parents[2]
TARGET_SCOPES = (
    ("_lint_commands", "lint", "QUALITY_PATHS", "production quality"),
    ("_lint_commands", "lint", "LINT_EXTRA_PATHS", "lint-only"),
    ("_typecheck_commands", "typecheck", "QUALITY_PATHS", "production quality"),
    ("_test_commands", "tests", "TEST_TARGETS", "tests"),
)


@pytest.fixture
def gate_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give command planners an isolated, entirely present target declaration."""

    (tmp_path / "module.py").touch()
    (tmp_path / "helper.py").touch()
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(noxfile, "ROOT", tmp_path)
    monkeypatch.setattr(noxfile, "QUALITY_PATHS", ("module.py",))
    monkeypatch.setattr(noxfile, "LINT_EXTRA_PATHS", ("helper.py",))
    monkeypatch.setattr(noxfile, "TEST_TARGETS", ("tests",))
    return tmp_path


@dataclass
class _RecordingSession:
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def install(self, *args: str) -> None:
        self.calls.append(("install", args))

    def run(self, *args: str) -> None:
        self.calls.append(("run", args))

    def log(self, message: str) -> None:
        self.calls.append(("log", (message,)))


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
        "pdf_chunker/geometry.py",
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
        "pdf_chunker/passes/text_clean.py",
        "pdf_chunker/passes/metrics.py",
    } <= set(QUALITY_PATHS)


def test_lint_commands_are_derived_from_declared_paths() -> None:
    expected_paths = tuple(dict.fromkeys((*QUALITY_PATHS, *LINT_EXTRA_PATHS)))
    ruff, black = _lint_commands()

    assert ruff == ("ruff", "check", "--no-fix", *expected_paths)
    assert black == ("black", "--check", *expected_paths)


def test_typecheck_command_is_derived_from_quality_paths() -> None:
    assert _typecheck_commands() == (("mypy", *dict.fromkeys(QUALITY_PATHS)),)


def test_test_command_is_derived_from_declared_targets() -> None:
    assert noxfile._test_commands() == (
        ("pytest", "-q", "-m", "not legacy_parity", *dict.fromkeys(noxfile.TEST_TARGETS)),
    )


@pytest.mark.parametrize("planner,session_name,declaration,scope", TARGET_SCOPES)
def test_missing_targets_fail_with_all_missing_paths_in_order(
    gate_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner: str,
    session_name: str,
    declaration: str,
    scope: str,
) -> None:
    targets = ("module.py", "second_missing.py", "first_missing.py", "second_missing.py")
    monkeypatch.setattr(noxfile, declaration, targets)

    with pytest.raises(FileNotFoundError) as error:
        getattr(noxfile, planner)()

    assert str(error.value) == (
        f"Missing declared {scope} targets: second_missing.py, first_missing.py."
    )
    assert getattr(noxfile, declaration) == targets


@pytest.mark.parametrize("planner,session_name,declaration,scope", TARGET_SCOPES)
def test_empty_target_declarations_fail(
    gate_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner: str,
    session_name: str,
    declaration: str,
    scope: str,
) -> None:
    monkeypatch.setattr(noxfile, declaration, ())

    with pytest.raises(ValueError, match=f"^No targets declared for {scope}\\.$"):
        getattr(noxfile, planner)()


@pytest.mark.parametrize("blank", ("", " \t"))
@pytest.mark.parametrize("planner,session_name,declaration,scope", TARGET_SCOPES)
def test_blank_targets_do_not_expand_or_shrink_the_gate(
    gate_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner: str,
    session_name: str,
    declaration: str,
    scope: str,
    blank: str,
) -> None:
    monkeypatch.setattr(noxfile, declaration, ("module.py", blank))

    with pytest.raises(ValueError, match=f"^Blank targets declared for {scope}\\.$"):
        getattr(noxfile, planner)()


@pytest.mark.parametrize("invalid", (("missing.py",), (), ("",)))
@pytest.mark.parametrize("planner,session_name,declaration,scope", TARGET_SCOPES)
def test_invalid_targets_fail_before_session_effects(
    gate_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner: str,
    session_name: str,
    declaration: str,
    scope: str,
    invalid: tuple[str, ...],
) -> None:
    monkeypatch.setattr(noxfile, declaration, invalid)
    session = _RecordingSession()
    expected_error = FileNotFoundError if invalid and invalid[0] else ValueError

    with pytest.raises(expected_error):
        getattr(noxfile, session_name)(session)

    assert session.calls == []


def test_target_resolution_consumes_a_one_shot_input_in_order(gate_root: Path) -> None:
    targets = ("tests", "module.py", "tests", "helper.py", "module.py")
    stream = iter(targets)
    resolve = partial(noxfile._existing, scope="fixture")

    assert resolve(stream) == ("tests", "module.py", "helper.py")
    assert tuple(stream) == ()
    assert targets == ("tests", "module.py", "tests", "helper.py", "module.py")


def test_nonblank_target_spelling_is_not_normalized(gate_root: Path) -> None:
    name = " spaced name.py "
    (gate_root / name).touch()

    assert noxfile._existing((name,)) == (name,)


@given(st.lists(st.sampled_from(QUALITY_PATHS), min_size=1, max_size=30))
def test_target_resolution_is_ordered_idempotent_and_preserves_inputs(paths: list[str]) -> None:
    before = paths.copy()
    resolved = noxfile._existing(iter(paths))

    assert resolved == tuple(dict.fromkeys(paths))
    assert noxfile._existing(resolved) == resolved
    assert paths == before


def test_lint_target_duplicates_are_removed_across_declarations(
    gate_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(noxfile, "QUALITY_PATHS", ("module.py", "module.py"))
    monkeypatch.setattr(
        noxfile, "LINT_EXTRA_PATHS", ("helper.py", "module.py", "tests", "helper.py")
    )

    assert noxfile._lint_commands() == (
        ("ruff", "check", "--no-fix", "module.py", "helper.py", "tests"),
        ("black", "--check", "module.py", "helper.py", "tests"),
    )


@pytest.mark.parametrize(
    "session_name,dependencies,commands",
    (
        (
            "lint",
            noxfile.SESSION_DEPENDENCIES,
            (
                ("ruff", "check", "--no-fix", "module.py", "helper.py"),
                ("black", "--check", "module.py", "helper.py"),
            ),
        ),
        ("typecheck", noxfile.SESSION_DEPENDENCIES, (("mypy", "module.py"),)),
        (
            "tests",
            noxfile.TEST_SESSION_DEPENDENCIES,
            (("pytest", "-q", "-m", "not legacy_parity", "tests"),),
        ),
    ),
)
def test_valid_sessions_keep_dependency_profiles_and_command_order(
    gate_root: Path,
    session_name: str,
    dependencies: tuple[str, ...],
    commands: tuple[tuple[str, ...], ...],
) -> None:
    session = _RecordingSession()

    getattr(noxfile, session_name)(session)

    assert session.calls == [("install", dependencies), *(("run", command) for command in commands)]


def test_semantic_helper_tests_are_lint_checked() -> None:
    assert {
        "tests/chunk_pipeline_test.py",
        "tests/sentence_fusion_test.py",
    } <= set(LINT_EXTRA_PATHS)


def test_structure_pass_tests_are_lint_checked() -> None:
    assert "tests/structure_pass_purity_test.py" in LINT_EXTRA_PATHS


def test_page_artifact_pass_tests_are_lint_checked() -> None:
    assert "tests/page_artifact_pass_test.py" in LINT_EXTRA_PATHS


def test_text_clean_purity_tests_are_lint_checked() -> None:
    assert "tests/text_clean_purity_test.py" in LINT_EXTRA_PATHS


def test_geometry_contract_tests_are_lint_checked() -> None:
    assert "tests/geometry_contract_test.py" in LINT_EXTRA_PATHS


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
