from __future__ import annotations

import json

from pdf_chunker.config import PipelineSpec

from scripts import replay_from_snapshot as replay


def test_run_passes_from_snapshot(tmp_path) -> None:
    snap = {"type": "page_blocks", "pages": [{"page_number": 1, "blocks": [{"text": "ﬁsh"}]}]}
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps(snap), encoding="utf-8")
    spec = PipelineSpec(pipeline=["pdf_parse", "text_clean"], options={})
    art = replay.artifact_from_snapshot(str(snap_path))
    tail = replay.passes_after(spec, "pdf_parse")
    result = replay.run_passes(spec, art, tail)
    text = result.payload["pages"][0]["blocks"][0]["text"]
    assert text == "fish"
