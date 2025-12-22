from collections.abc import Callable
from functools import reduce

import pytest

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _enforce_invariants


def _add(step: str) -> Callable[[PipelineSpec], PipelineSpec]:
    return lambda spec: PipelineSpec(pipeline=[*spec.pipeline, step])


def _build_pipeline(*steps: str) -> PipelineSpec:
    return reduce(lambda spec, s: _add(s)(spec), steps, PipelineSpec(pipeline=[]))


def test_valid_pipeline() -> None:
    spec = _build_pipeline("pdf_parse", "text_clean", "split_semantic")
    _enforce_invariants(spec, input_path="dummy.pdf")


def test_split_requires_clean() -> None:
    spec = _build_pipeline("split_semantic", "text_clean")
    with pytest.raises(ValueError):
        _enforce_invariants(spec, input_path="dummy.pdf")


def test_media_mixing_rejected() -> None:
    spec = _build_pipeline("pdf_parse", "epub_parse")
    with pytest.raises(ValueError):
        _enforce_invariants(spec, input_path="dummy.pdf")
