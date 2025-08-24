from pdf_chunker.cli import _cli_overrides
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _run_passes
from pdf_chunker.framework import Artifact


def _doc(text: str) -> dict:
    return {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [{"page": 1, "blocks": [{"text": text}]}],
    }


def test_cli_flags_affect_split_semantic(monkeypatch) -> None:
    captured: dict[str, tuple[int, int, int]] = {}

    def fake_semantic_chunker(text: str, chunk_size: int, overlap: int, *, min_chunk_size: int) -> list[str]:
        captured["args"] = (chunk_size, overlap, min_chunk_size)
        return [text[:5], text[5:]]

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", fake_semantic_chunker)
    overrides = _cli_overrides(
        out=None,
        chunk_size=5,
        overlap=0,
        enrich=False,
        exclude_pages=None,
        no_metadata=True,
    )
    spec = PipelineSpec(pipeline=["split_semantic"], options=overrides)
    art = Artifact(payload=_doc("hello world"))
    out, _ = _run_passes(spec, art)
    items = out.payload["items"]
    assert captured["args"] == (5, 0, 8)
    assert len(items) == 2 and all("meta" not in item for item in items)
