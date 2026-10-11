"""Structure passes preserve their inputs, including nested metric containers."""

from copy import deepcopy
from functools import partial
from types import MappingProxyType
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker.framework import Artifact, Pass
from pdf_chunker.passes.detect_doc_end import _truncate_pages, detect_doc_end
from pdf_chunker.passes.heading_detect import annotate_headings, heading_detect
from pdf_chunker.passes.list_detect import list_detect
from pdf_chunker.passes.merge_footers import _merge_trailing_short_lines, merge_footers
from pdf_chunker.passes.metrics import with_metrics

STRUCTURE_PASSES = (heading_detect, list_detect, merge_footers, detect_doc_end)


def _payload(step: Pass) -> Any:
    blocks = [{"text": "Chapter 1"}, {"text": "Some ordinary paragraph text."}]
    return (
        blocks
        if step is heading_detect
        else {"type": "page_blocks", "pages": [{"page": 1, "blocks": blocks}]}
    )


@pytest.mark.parametrize("step", STRUCTURE_PASSES, ids=lambda step: step.name)
def test_pass_metrics_do_not_mutate_input_artifact(step: Pass) -> None:
    meta: dict[str, object] = {
        "source": "sample.pdf",
        "options": {"untouched": [1]},
        "metrics": {step.name: {"custom": 7}, "other": {"pages": 2}},
    }
    artifact = Artifact(_payload(step), meta)
    before = deepcopy(artifact)

    result = step(artifact)
    repeated = step(artifact)

    assert artifact == before
    assert meta == before.meta
    assert result == repeated
    assert result.meta is not None
    metrics = result.meta["metrics"]
    assert isinstance(metrics, dict)
    assert metrics[step.name]["custom"] == 7
    assert metrics["other"] == {"pages": 2}
    assert result.meta["options"] == {"untouched": [1]}


@pytest.mark.parametrize("step", STRUCTURE_PASSES, ids=lambda step: step.name)
@pytest.mark.parametrize("payload", (None, "text", {"type": "chunks", "items": []}))
def test_unsupported_payload_is_identity(step: Pass, payload: object) -> None:
    artifact = Artifact(payload, {"metrics": {step.name: {"custom": 7}}})
    assert step(artifact) is artifact


@pytest.mark.parametrize("step", STRUCTURE_PASSES, ids=lambda step: step.name)
@pytest.mark.parametrize("meta", (None, {}))
def test_pass_creates_missing_metrics(step: Pass, meta: dict[str, object] | None) -> None:
    result = step(Artifact(_payload(step), meta))
    assert result.meta is not None
    metrics = result.meta["metrics"]
    assert isinstance(metrics, dict)
    assert step.name in metrics


def test_metric_update_can_be_partially_applied_to_read_only_mappings() -> None:
    existing = MappingProxyType({"count": 3, "custom": 7})
    all_metrics = MappingProxyType({"example": existing, "other": {"pages": 2}})
    meta = MappingProxyType({"metrics": all_metrics, "source": "sample.pdf"})
    updates = MappingProxyType({"count": 5})
    update = partial(with_metrics, pass_name="example", updates=updates)

    result = update(meta)

    assert result == {
        "metrics": {"example": {"count": 5, "custom": 7}, "other": {"pages": 2}},
        "source": "sample.pdf",
    }
    assert existing["count"] == 3
    assert updates["count"] == 5
    assert result["metrics"] is not all_metrics
    assert result["metrics"]["example"] is not existing
    assert update(result) == result


@given(
    st.dictionaries(st.sampled_from(("count", "pages", "custom")), st.integers()),
    st.dictionaries(st.sampled_from(("count", "pages", "custom")), st.integers()),
)
def test_metric_updates_compose_by_right_biased_merge(
    first: dict[str, int], second: dict[str, int]
) -> None:
    meta: dict[str, object] = {"metrics": {"example": {"keep": 7}, "other": {"keep": 2}}}
    before = deepcopy(meta)
    sequential = with_metrics(
        with_metrics(meta, pass_name="example", updates=first),
        pass_name="example",
        updates=second,
    )
    combined = with_metrics(meta, pass_name="example", updates={**first, **second})

    assert sequential == combined
    assert with_metrics(combined, pass_name="example", updates={}) == combined
    assert meta == before


@pytest.mark.parametrize("invalid", (None, [], "bad", 7))
@pytest.mark.parametrize("nested", (False, True))
def test_malformed_metrics_fail_explicitly(invalid: object, nested: bool) -> None:
    meta = {"metrics": {"example": invalid} if nested else invalid}
    before = deepcopy(meta)
    with pytest.raises(TypeError, match="must be a mapping"):
        with_metrics(meta, pass_name="example", updates={"count": 1})
    assert meta == before


@pytest.mark.parametrize(
    ("texts", "merged"),
    (
        ([], 0),
        (["short", "tail"], 0),
        (["x" * 41], 0),
        (["x" * 41, "one"], 0),
        (["x" * 41, "one", "two"], 1),
        (["x" * 41, "one", "two", "three"], 2),
        (["x" * 41, "one", "two", "three", "four"], 0),
        (["x" * 41, "y" * 40, "tail"], 1),
        (["x" * 41, "y" * 41, "tail"], 0),
    ),
)
def test_footer_boundaries_preserve_words_and_first_tail_provenance(
    texts: list[str], merged: int
) -> None:
    blocks = [{"text": text, "source": {"line": index}} for index, text in enumerate(texts)]
    before = deepcopy(blocks)

    result, count = _merge_trailing_short_lines(blocks)

    assert count == merged
    assert len(result) == len(blocks) - merged
    assert " ".join(block["text"] for block in result).split() == " ".join(texts).split()
    assert blocks == before
    if merged:
        assert result[-1]["source"] == blocks[-merged - 1]["source"]


@pytest.mark.parametrize(
    ("texts", "remaining"),
    (
        ([], 0),
        (["THE END", "tail", "tail", "tail"], 4),
        (["THE END", "tail", "tail"], 1),
        (["body", "END", "tail"], 2),
        (["END", "THE END", "tail"], 2),
        (["body", "end", "tail"], 3),
    ),
)
def test_document_end_search_window_and_latest_marker(texts: list[str], remaining: int) -> None:
    pages = [
        {"blocks": [{"text": text}, {"text": ". . ."}, {"text": str(index + 1)}]}
        for index, text in enumerate(texts)
    ]
    before = deepcopy(pages)

    result, truncated = _truncate_pages(pages)

    assert result == before[:remaining]
    assert truncated == len(pages) - remaining
    assert pages == before


@pytest.mark.parametrize(
    ("text", "is_heading"),
    (("", False), ("  Chapter 1  ", True), ("Body", False), ("Ordinary body text.", False)),
)
def test_heading_annotations_consume_one_shot_input_without_mutation(
    text: str, is_heading: bool
) -> None:
    block = {"text": text, "type": "heading", "source": {"page": 2}}
    before = deepcopy(block)
    result = annotate_headings(iter((block,)))

    assert result[0]["text"] == text.strip()
    assert result[0]["is_heading"] is is_heading
    assert result[0]["type"] == ("heading" if is_heading else "paragraph")
    assert result[0]["source"] == block["source"]
    assert block == before


def test_list_continuation_state_resets_at_page_boundaries() -> None:
    doc = {
        "type": "page_blocks",
        "pages": [
            {"page": 1, "blocks": [{"text": "1. unfinished"}, {"text": "continuation."}]},
            {"page": 2, "blocks": [{"text": "ordinary"}, {"text": "•"}, {"text": "item"}]},
        ],
    }
    before = deepcopy(doc)
    result = list_detect(Artifact(doc))
    pages = result.payload["pages"]

    assert [block.get("list_kind") for block in pages[0]["blocks"]] == ["numbered", "numbered"]
    assert [block.get("list_kind") for block in pages[1]["blocks"]] == [None, "bullet", "bullet"]
    assert doc == before
    assert result.meta["metrics"]["list_detect"] == {"bullet_items": 2, "numbered_items": 2}
