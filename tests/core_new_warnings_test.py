from pdf_chunker.core_new import _collect_warnings
from pdf_chunker.config import PipelineSpec
from pdf_chunker.framework import Artifact


def test_collect_warnings_flags_known_issues() -> None:
    payload = [
        {"text": "Body footnote 1", "metadata": {"chunk_id": "c1", "source": {}}},
        {"text": "Some text", "metadata": {"chunk_id": "c2"}},
    ]
    spec = PipelineSpec(
        pipeline=[],
        options={"pdf_parse": {"exclude_pages": "1", "engine": "pymupdf4llm"}},
    )
    warnings = _collect_warnings(Artifact(payload=payload, meta={}), spec, generate_metadata=True)
    assert set(warnings) == {
        "footnote_anchors",
        "page_exclusion_noop",
        "metadata_gaps",
        "underscore_loss",
    }


def test_collect_warnings_ignores_metadata_when_disabled() -> None:
    payload = [{"text": "plain"}]
    spec = PipelineSpec(options={"split_semantic": {"generate_metadata": False}})
    warnings = _collect_warnings(Artifact(payload=payload, meta={}), spec, generate_metadata=False)
    assert "metadata_gaps" not in warnings
