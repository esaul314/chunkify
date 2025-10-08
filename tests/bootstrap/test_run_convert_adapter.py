import json

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import assemble_report, run_convert, write_run_report
from pdf_chunker.framework import Artifact, registry


class _StubClient:
    def classify_chunk_utterance(self, text_chunk: str, *, tag_configs: dict) -> dict:
        return {"classification": "question", "tags": ["technical"]}


def test_run_convert_writes_jsonl(tmp_path):
    spec = PipelineSpec(
        pipeline=["emit_jsonl"],
        options={
            "emit_jsonl": {"output_path": str(tmp_path / "out.jsonl")},
            "run_report": {"output_path": str(tmp_path / "run_report.json")},
        },
    )
    doc = {"type": "chunks", "items": [{"text": "a"}, {"text": "b"}]}
    artifact = Artifact(payload=doc, meta={"metrics": {}, "input": "doc.pdf"})
    artifact, timings = run_convert(artifact, spec)
    passes = [registry()[s] for s in spec.pipeline]
    report = assemble_report(timings, artifact.meta or {}, passes)
    write_run_report(spec, report)

    out_file = tmp_path / "out.jsonl"
    lines = out_file.read_text(encoding="utf-8").splitlines()
    expected = [json.dumps(r, ensure_ascii=False) for r in artifact.payload]
    assert lines == expected

    report_file = tmp_path / "run_report.json"
    assert report_file.exists()


def test_run_convert_with_ai_enrich_preserves_items(tmp_path) -> None:
    out_path = tmp_path / "out.jsonl"
    spec = PipelineSpec(
        pipeline=["ai_enrich", "emit_jsonl"],
        options={
            "ai_enrich": {
                "enabled": True,
                "client": _StubClient(),
                "tag_configs": {"generic": ["technical"]},
            },
            "emit_jsonl": {"output_path": str(out_path)},
            "run_report": {"output_path": str(tmp_path / "run_report.json")},
        },
    )
    doc = {"type": "chunks", "items": [{"text": "What is AI?"}]}
    artifact = Artifact(payload=doc, meta={"metrics": {}, "input": "doc.pdf"})
    result, _ = run_convert(artifact, spec)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert rows and rows[0]["metadata"]["utterance_type"] == "question"
    assert result.payload[0]["metadata"]["tags"] == ["technical"]
