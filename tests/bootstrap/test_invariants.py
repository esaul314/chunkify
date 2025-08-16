import pytest

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _enforce_invariants


def test_clean_before_split_enforced() -> None:
    spec = PipelineSpec(pipeline=["split_semantic", "text_clean"])
    with pytest.raises(ValueError):
        _enforce_invariants(spec, input_path="dummy.pdf")


def test_pdf_epub_separation_enforced() -> None:
    spec = PipelineSpec(pipeline=["pdf_parse", "epub_parse"])
    with pytest.raises(ValueError):
        _enforce_invariants(spec, input_path="dummy.pdf")


def test_valid_pipeline_passes() -> None:
    spec = PipelineSpec(pipeline=["pdf_parse", "text_clean", "split_semantic"])
    _enforce_invariants(spec, input_path="dummy.pdf")
