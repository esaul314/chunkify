import pytest

from pdf_chunker.cli import _cli_overrides
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import _run_passes, run_convert
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_semantic import split_semantic


def _doc(text: str) -> dict:
    return {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [{"page": 1, "blocks": [{"text": text}]}],
    }


def test_cli_flags_affect_split_semantic(monkeypatch) -> None:
    captured: dict[str, tuple[int, int, int]] = {}

    def fake_semantic_chunker(
        text: str, chunk_size: int, overlap: int, *, min_chunk_size: int
    ) -> list[str]:
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


@pytest.mark.parametrize(
    "overrides, relation",
    [
        ({"split_semantic": {"chunk_size": 200}}, "gt"),
        ({"split_semantic": {"overlap": 10}}, "lt"),
    ],
)
def test_split_counts_change_with_overrides(tmp_path, monkeypatch, overrides, relation) -> None:
    def fake_semantic_chunker(
        text: str, chunk_size: int, overlap: int, *, min_chunk_size: int
    ) -> list[str]:
        step = chunk_size - overlap
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", fake_semantic_chunker)

    def _run(opts: dict | None = None) -> int:
        spec_opts = {"run_report": {"output_path": str(tmp_path / "r.json")}}
        spec = PipelineSpec(options={**spec_opts, **(opts or {})})
        art = Artifact(payload=_doc("x" * 1150), meta={"metrics": {}, "input": "doc.pdf"})
        seeded, _ = run_convert(art, spec)
        return len(split_semantic(seeded).payload["items"])

    base = _run()
    new = _run(overrides)
    assert (new > base) if relation == "gt" else (new < base)
