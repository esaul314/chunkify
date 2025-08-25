import pytest

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _enforce_invariants


def _check(spec: PipelineSpec, path: str) -> None:
    """Helper to trigger invariant validation."""
    _enforce_invariants(spec, input_path=path)


@pytest.mark.parametrize(
    "spec",
    [
        PipelineSpec(pipeline=["split_semantic", "text_clean"]),
        PipelineSpec(pipeline=["pdf_parse", "epub_parse"]),
    ],
)
def test_invalid_pipelines_rejected(spec: PipelineSpec) -> None:
    with pytest.raises(ValueError):
        _check(spec, "dummy.pdf")


def test_valid_pipeline_passes() -> None:
    spec = PipelineSpec(pipeline=["pdf_parse", "text_clean", "split_semantic"])
    _check(spec, "dummy.pdf")


def test_parse_step_replaced_for_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = PipelineSpec(pipeline=["pdf_parse", "text_clean"])
    monkeypatch.setattr(
        "pdf_chunker.core_new.registry",
        lambda: {"epub_parse": object(), "text_clean": object()},
        raising=False,
    )
    steps = _enforce_invariants(spec, input_path="dummy.epub")
    assert steps == ["epub_parse", "text_clean"]
