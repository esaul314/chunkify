import logging
from collections import Counter

from pdf_chunker import core
from pdf_chunker.framework import Artifact


def test_parse_exclusions_success():
    assert core.parse_exclusions("1, 3-4") == {1, 3, 4}


def test_parse_exclusions_logs_error_on_failure(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=core.logger.name)

    def boom(_spec: str) -> set[int]:
        raise ValueError("boom")

    monkeypatch.setattr("pdf_chunker.page_utils.parse_page_ranges", boom)

    assert core.parse_exclusions("1-2") == set()
    assert any("Error parsing" in rec.message for rec in caplog.records)


def test_filter_blocks_removes_excluded_pages(caplog):
    caplog.set_level(logging.DEBUG, logger=core.logger.name)
    blocks = (
        {"source": {"page": 1}, "text": "keep"},
        {"source": {"page": 2}, "text": "drop"},
        {"source": {}, "text": "missing"},
    )

    remaining = core.filter_blocks(blocks, {2})

    assert [block["text"] for block in remaining] == ["keep", "missing"]
    assert any("After filtering excluded pages" in rec.message for rec in caplog.records)


def test_chunk_text_invokes_splitter(monkeypatch):
    captured = []

    class DummySplit:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def __call__(self, artifact: Artifact) -> Artifact:
            captured.append(artifact.payload)
            return Artifact(payload={"items": [{"text": "first"}, {"text": ""}, {"text": "second"}]})

    monkeypatch.setattr(core, "_SplitSemanticPass", DummySplit)

    blocks = (
        {"text": "alpha", "source": {"page": 2, "index": 1}},
        {"text": "beta", "source": {"page": 1, "index": 0}},
    )

    chunks = core.chunk_text(blocks, 1000, 50, min_chunk_size=12, enable_dialogue_detection=False)

    assert chunks == ["first", "second"]
    params, payload = captured
    assert params["chunk_size"] == 1000
    assert params["overlap"] == 50
    assert params["min_chunk_size"] == 12
    assert payload["pages"][0]["page"] == 1
    assert [page["page"] for page in payload["pages"]] == [1, 2]


def test_log_chunk_stats_emits_warning_for_tiny_chunks(caplog):
    caplog.set_level(logging.WARNING, logger=core.logger.name)

    core.log_chunk_stats(["This chunk has many words", "tiny"], label="Snippet")

    messages = Counter(rec.levelname for rec in caplog.records)
    assert messages["WARNING"] == 1
    assert any("Very short chunks" in rec.message for rec in caplog.records)
