from pdf_chunker import core_new
from pdf_chunker.config import PipelineSpec
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.ai_enrich import _classify_chunk


class _ProbePass:
    name = "ai_enrich"
    input_type = dict
    output_type = dict

    def __init__(self, expected_client):
        self.expected_client = expected_client

    def __call__(self, artifact):
        options = artifact.meta["options"]["ai_enrich"]
        assert options["client"] is self.expected_client
        return artifact


def test_run_convert_passes_prepared_client_to_enrichment(monkeypatch):
    client = object()
    probe = _ProbePass(client)
    monkeypatch.setattr(core_new, "registry", lambda: {"ai_enrich": probe})
    monkeypatch.setattr(
        "pdf_chunker.adapters.ai_enrich.prepare_options",
        lambda options: {**options, "client": client},
    )
    monkeypatch.setattr(core_new, "write_run_report", lambda *_args: None)
    spec = PipelineSpec(pipeline=["ai_enrich"], options={"ai_enrich": {"enabled": True}})
    artifact = Artifact(payload={"type": "chunks", "items": []}, meta={"input": "sample.pdf"})

    result, _ = core_new.run_convert(artifact, spec)

    assert result.payload["type"] == "chunks"


def test_ai_enrichment_marks_successful_chunks_complete():
    class Client:
        def classify_chunk_utterance(self, _text, *, tag_configs):
            return {"classification": "explanation", "tags": []}

    result = _classify_chunk(
        {"text": "A useful explanation", "metadata": {}}, client=Client(), tag_configs={}
    )

    assert result["metadata"]["enrichment_status"] == "complete"
    assert result["meta"]["enrichment_status"] == "complete"


def test_ai_enrichment_fails_on_completion_error():
    import pytest

    class Client:
        def classify_chunk_utterance(self, _text, *, tag_configs):
            return {"classification": "error", "tags": []}

    with pytest.raises(RuntimeError, match="completion failed"):
        _classify_chunk(
            {"text": "A useful explanation", "metadata": {}}, client=Client(), tag_configs={}
        )
