"""Shared parser normalization preserves block order and extraction provenance."""

from copy import deepcopy

import pytest

import pdf_chunker.passes.pdf_parse as pdf_parse_mod
import pdf_chunker.pdf_parsing as pdf_parsing
from pdf_chunker.framework import Artifact, Pass
from pdf_chunker.passes.epub_parse import epub_parse
from pdf_chunker.passes.text_parse import text_parse

PARSERS = (pdf_parse_mod.pdf_parse, epub_parse, text_parse)


def test_pdf_parse_normalizes_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy extractor called")

    monkeypatch.setattr(pdf_parsing, "_legacy_extract_text_blocks_from_pdf", boom)

    blocks = [
        {"source": {"page": 2}, "text": "two"},
        {"source": {"page": None}, "text": "none"},
        {"source": {"page": 1}, "text": "one"},
    ]

    artifact = Artifact(payload=blocks, meta={"input": "dummy.pdf"})
    result = pdf_parse_mod.pdf_parse(artifact)
    assert result.payload == {
        "type": "page_blocks",
        "source_path": "dummy.pdf",
        "pages": [
            {"page": 1, "blocks": [blocks[2]]},
            {"page": 2, "blocks": [blocks[0]]},
            {"page": None, "blocks": [blocks[1]]},
        ],
    }
    assert result.meta["metrics"]["pdf_parse"]["pages"] == 3


@pytest.mark.parametrize("parser", PARSERS, ids=lambda parser: parser.name)
def test_parser_groups_one_shot_blocks_stably_without_mutation(parser: Pass) -> None:
    blocks = [
        {"source": {"page": 2}, "text": "Second page, first block."},
        {"source": {}, "text": "Unknown page, first block."},
        {"source": {"page": 1}, "text": "First page."},
        {"source": {"page": 2}, "text": "Second page, second block."},
        {"source": {"page": None}, "text": "Unknown page, second block."},
    ]
    original = deepcopy(blocks)
    meta = {"input": "memory", "metrics": {"upstream": {"blocks": 5}}}
    artifact = Artifact(payload=iter(blocks), meta=meta)

    result = parser(artifact)

    assert result.payload == {
        "type": "page_blocks",
        "source_path": "memory",
        "pages": [
            {"page": 1, "blocks": [blocks[2]]},
            {"page": 2, "blocks": [blocks[0], blocks[3]]},
            {"page": None, "blocks": [blocks[1], blocks[4]]},
        ],
    }
    assert result.meta["metrics"] == {
        "upstream": {"blocks": 5},
        "pdf_parse": {"pages": 3},
    }
    assert blocks == original
    assert artifact.meta == meta


@pytest.mark.parametrize("parser", PARSERS, ids=lambda parser: parser.name)
def test_parser_moves_extraction_report_to_metrics_without_mutation(parser: Pass) -> None:
    quality = {"selected_engine": "pymupdf", "quality_score": 0.9}
    payload = {
        "type": "page_blocks",
        "source_path": "memory",
        "pages": [{"page": 1, "blocks": [{"text": "Preserved text."}]}],
        "extraction_quality": quality,
    }
    original = deepcopy(payload)
    meta = {"metrics": {"pdf_parse": {"engine": "pymupdf"}, "upstream": {"blocks": 1}}}
    artifact = Artifact(payload=payload, meta=meta)

    result = parser(artifact)

    assert result.payload == {
        "type": "page_blocks",
        "source_path": "memory",
        "pages": payload["pages"],
    }
    assert result.meta["metrics"] == {
        "pdf_parse": {"engine": "pymupdf", "pages": 1, "extraction_quality": quality},
        "upstream": {"blocks": 1},
    }
    assert payload == original
    assert artifact.meta == meta
