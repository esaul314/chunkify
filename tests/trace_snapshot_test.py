from __future__ import annotations

import json
from pathlib import Path

from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import run_convert
from pdf_chunker.framework import Artifact


def _doc(blocks: list[str]) -> dict:
    return {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [{"page": 1, "blocks": [{"text": b} for b in blocks]}],
    }


def test_trace_snapshots(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_chunker(text: str, chunk_size: int, overlap: int, *, min_chunk_size: int):
        return ["foo part", "rest"]

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", fake_chunker)
    art = Artifact(payload=_doc(["foo one", "two"]), meta={"input": "doc.pdf"})
    spec = PipelineSpec(
        pipeline=["text_clean", "split_semantic"],
        options={"run_report": {"output_path": str(tmp_path / "r.json")}},
    )
    run_convert(art, spec, trace="foo")

    trace_root = Path("artifacts/trace")
    run_dir = next(trace_root.iterdir())
    steps = {p.stem for p in run_dir.iterdir()}
    assert steps == {"text_clean", "split_semantic"}

    clean = json.loads((run_dir / "text_clean.json").read_text())
    assert clean["pages"][0]["blocks"] == [{"text": "foo one"}]

    chunks = json.loads((run_dir / "split_semantic.json").read_text())
    assert chunks and all("foo" in c["text"] for c in chunks)
