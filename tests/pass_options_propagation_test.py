from pdf_chunker.core_new import _run_passes
from pdf_chunker.framework import Artifact
from pdf_chunker.config import PipelineSpec


def _doc(text: str) -> dict:
    return {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [{"page": 1, "blocks": [{"text": text}]}],
    }


def test_run_passes_respects_spec_options(monkeypatch) -> None:
    captured: dict[str, tuple[int, int, int]] = {}

    def fake_semantic_chunker(
        text: str, chunk_size: int, overlap: int, *, min_chunk_size: int
    ) -> list[str]:
        captured["args"] = (chunk_size, overlap, min_chunk_size)
        return [text[:5], text[5:]]

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", fake_semantic_chunker)

    opts = {"split_semantic": {"chunk_size": 5, "overlap": 0, "generate_metadata": False}}
    art = Artifact(payload=_doc("hello world"))
    spec = PipelineSpec(pipeline=["split_semantic"], options=opts)
    out, _ = _run_passes(spec, art)
    items = out.payload["items"]

    assert captured["args"] == (5, 0, 8)
    assert len(items) >= 1
    assert all("meta" not in item for item in items)
    assert out.meta["options"]["split_semantic"] == {
        "chunk_size": 5,
        "overlap": 0,
        "generate_metadata": False,
    }


def test_run_passes_respects_min_size_override(monkeypatch) -> None:
    captured: dict[str, tuple[int, int, int]] = {}

    def fake_semantic_chunker(
        text: str, chunk_size: int, overlap: int, *, min_chunk_size: int
    ) -> list[str]:
        captured["args"] = (chunk_size, overlap, min_chunk_size)
        return [text]

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", fake_semantic_chunker)

    opts = {
        "split_semantic": {
            "chunk_size": 12,
            "overlap": 2,
            "min_chunk_size": 6,
        }
    }
    art = Artifact(payload=_doc("hello world"))
    spec = PipelineSpec(pipeline=["split_semantic"], options=opts)
    out, _ = _run_passes(spec, art)

    assert captured["args"] == (12, 2, 6)
    assert out.meta["options"]["split_semantic"]["min_chunk_size"] == 6
