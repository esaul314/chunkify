import pytest

from pdf_chunker.cli import _cli_overrides
from pdf_chunker.config import PipelineSpec
from pdf_chunker.core_new import run_convert
from pdf_chunker.framework import Artifact
from pdf_chunker.passes.split_semantic import (
    _collapse_records,
    _soft_segments,
    split_semantic,
)


def _observed_overlap(first: list[str], second: list[str]) -> int:
    limit = min(len(first), len(second))
    return max(
        (size for size in range(limit, -1, -1) if first[-size:] == second[:size]),
        default=0,
    )


def _doc(text: str) -> dict:
    return {
        "type": "page_blocks",
        "source_path": "src.pdf",
        "pages": [{"page": 1, "blocks": [{"text": text}]}],
    }


def test_cli_flags_affect_split_semantic(tmp_path, monkeypatch) -> None:
    captured: dict[str, tuple[int, int, int]] = {}

    def fake_semantic_chunker(
        text: str, chunk_size: int, overlap: int, *, min_chunk_size: int
    ) -> list[str]:
        captured["args"] = (chunk_size, overlap, min_chunk_size)
        words = text.split()
        step = max(chunk_size - overlap, 1)
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), step)
            if words[i : i + chunk_size]
        ]

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", fake_semantic_chunker)
    overrides = _cli_overrides(
        out=None,
        chunk_size=5,
        overlap=0,
        enrich=False,
        exclude_pages=None,
        no_metadata=True,
    )
    opts = {**overrides, "run_report": {"output_path": str(tmp_path / "r.json")}}
    spec = PipelineSpec(pipeline=["text_clean", "split_semantic"], options=opts)
    source = " ".join(f"w{i}" for i in range(7))
    art = Artifact(payload=_doc(source), meta={"input": "doc.pdf"})
    out, _ = run_convert(art, spec)
    items = out.payload["items"]
    words = [item["text"].split() for item in items]

    assert captured["args"] == (5, 0, 8)
    assert len(items) == 2 and all("meta" not in item for item in items)
    assert len(words[0]) == 5
    assert _observed_overlap(words[0], words[1]) == 0


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


def test_dense_fragments_respect_override_limits(tmp_path, monkeypatch) -> None:
    calls: list[int] = []

    def fake_semantic_chunker(
        text: str, chunk_size: int, overlap: int, *, min_chunk_size: int
    ) -> list[str]:
        calls.append(chunk_size)
        if chunk_size <= 80:
            step = max(chunk_size - overlap, 1)
            return [text[i : i + chunk_size] for i in range(0, len(text), step)]
        return [text]

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", fake_semantic_chunker)

    def _run(opts: dict | None = None) -> list[str]:
        spec_opts = {"run_report": {"output_path": str(tmp_path / "r.json")}}
        spec = PipelineSpec(options={**spec_opts, **(opts or {})})
        text = "x" * 600 + "y" * 600
        art = Artifact(payload=_doc(text), meta={"metrics": {}, "input": "doc.pdf"})
        seeded, _ = run_convert(art, spec)
        return [item["text"] for item in split_semantic(seeded).payload["items"]]

    base = _run()
    override = _run({"split_semantic": {"chunk_size": 80, "overlap": 0}})

    assert 400 in calls
    assert 80 in calls
    assert len(base) == 1
    assert len(override) > len(base)
    assert any(" " not in text for text in override)


def test_run_convert_overrides_existing_meta_options(tmp_path, monkeypatch) -> None:
    captured: dict[str, tuple[int, int, int]] = {}

    def fake_semantic_chunker(
        text: str, chunk_size: int, overlap: int, *, min_chunk_size: int
    ) -> list[str]:
        captured["args"] = (chunk_size, overlap, min_chunk_size)
        return [text]

    monkeypatch.setattr("pdf_chunker.splitter.semantic_chunker", fake_semantic_chunker)

    art = Artifact(
        payload=_doc("hello"),
        meta={"input": "doc.pdf", "options": {"split_semantic": {"chunk_size": 99, "overlap": 9}}},
    )
    opts = {
        "split_semantic": {"chunk_size": 5, "overlap": 1},
        "run_report": {"output_path": str(tmp_path / "r.json")},
    }
    spec = PipelineSpec(options=opts)
    seeded, _ = run_convert(art, spec)
    split_semantic(seeded)

    assert captured["args"] == (5, 1, 8)
    assert seeded.meta["options"]["split_semantic"] == {"chunk_size": 5, "overlap": 1}


def test_collapse_records_does_not_span_pages() -> None:
    records = [
        (
            page,
            {"text": text, "type": "paragraph", "source": {"page": page}},
            text,
        )
        for page, text in ((1, "alpha"), (1, "beta"), (2, "gamma"))
    ]

    collapsed = list(_collapse_records(records, limit=100))

    pages = [page for page, *_ in collapsed]
    assert pages == [1, 2]
    assert collapsed[0][2].split() == ["alpha", "beta"]
    assert collapsed[1][2] == "gamma"
    assert collapsed[1][1]["source"]["page"] == 2


def test_soft_segments_retains_internal_newlines_under_word_budget() -> None:
    text = "1. First item\n2. Second item\n3. Third item"

    segments = _soft_segments(text, max_chars=40, max_words=4)

    assert "\n2." in segments[0]
