"""Characterize pure page cleanup, option precedence, and ordered aggregation."""

import re
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker.framework import Artifact
from pdf_chunker.interactive import FooterConfig, FooterDecisionCache
from pdf_chunker.passes.detect_page_artifacts import (
    _clean_block,
    _clean_doc,
    _DetectPageArtifactsPass,
)

TABLE = "|Chapter 1|Col2|\n|---|---|\n|Chapter 1||\nBody text."
CONFIG = FooterConfig.from_dict({"known_footer_patterns": [r"Footer \d+"]})


def _doc(texts: list[str]) -> dict[str, Any]:
    return {
        "type": "page_blocks",
        "source_path": "book.pdf",
        "pages": [
            {
                "page": index,
                "location": f"page-{index}",
                "blocks": [{"text": text, "source": {"page": index}}],
            }
            for index, text in enumerate(texts, 1)
        ],
    }


@pytest.mark.parametrize(
    ("text", "expected", "changed"),
    (
        ("Body text.", "Body text.", False),
        ("", "", False),
        (" \n ", " \n ", False),
        ("|Not a table|", "|Not a table|", False),
        (TABLE, "Chapter 1\nBody text.", True),
        ("Footer 9", "", True),
        ("Body with Footer 9 inside.", "", True),
    ),
)
def test_block_cleanup_preserves_provenance(text: str, expected: str, changed: bool) -> None:
    block = {"text": text, "type": "paragraph", "source": {"page": 9}}
    before = deepcopy(block)

    cleaned, actual_changed, decisions = _clean_block(block, 9, CONFIG)

    assert cleaned == {**block, "text": expected}
    assert actual_changed is changed
    assert decisions == ([r"block matched footer pattern: Footer \d+"] if "Footer" in text else [])
    assert block == before


def test_patterns_match_flattened_text_and_first_match_wins() -> None:
    config = FooterConfig.from_dict({"known_footer_patterns": [r"^Chapter 1\nBody", "Chapter"]})
    cleaned, changed, decisions = _clean_block({"text": TABLE}, 1, config)
    assert cleaned == {"text": ""}
    assert changed is True
    assert decisions == [r"block matched footer pattern: ^Chapter 1\nBody"]


def test_document_cleanup_accepts_one_shot_pages_and_partial_application() -> None:
    doc = _doc([TABLE, "Footer 2", "Body text.", " \n "])
    before = deepcopy(doc)
    visited: list[int] = []

    def pages() -> Iterator[dict[str, Any]]:
        for page in doc["pages"]:
            visited.append(page["page"])
            yield page

    clean = partial(_clean_doc, footer_config=CONFIG)
    cleaned, count, decisions = clean({**doc, "pages": pages()})

    assert visited == [1, 2, 3, 4]
    assert cleaned == {
        **doc,
        "pages": [
            {**page, "blocks": [{**page["blocks"][0], "text": text}] if text else []}
            for page, text in zip(
                doc["pages"], ("Chapter 1\nBody text.", "", "Body text.", ""), strict=True
            )
        ],
    }
    assert count == 2  # Empty input blocks are filtered, not counted as changed.
    assert decisions == [r"block matched footer pattern: Footer \d+"]
    assert doc == before
    again, changes, _ = clean(cleaned)
    assert again == cleaned
    assert changes == 0


def test_decisions_follow_page_and_block_order() -> None:
    config = FooterConfig.from_dict({"known_footer_patterns": ["Footer", "Runtime"]})
    doc = _doc(["Footer 1", "Body text.", "Footer 3"])
    doc["pages"][0]["blocks"].append({"text": "Runtime 1"})

    cleaned, count, decisions = _clean_doc(doc, config)

    assert count == 3
    assert decisions == [
        "block matched footer pattern: Footer",
        "block matched footer pattern: Runtime",
        "block matched footer pattern: Footer",
    ]
    assert [page["page"] for page in cleaned["pages"]] == [1, 2, 3]
    assert [len(page["blocks"]) for page in cleaned["pages"]] == [0, 1, 0]


@pytest.mark.parametrize("doc", ({}, {"pages": []}, {"pages": [{"page": 1}]}))
def test_missing_pages_or_blocks_have_empty_identity(doc: dict[str, Any]) -> None:
    before = deepcopy(doc)
    cleaned, count, decisions = _clean_doc(doc)
    assert cleaned == {
        **doc,
        "pages": [{**page, "blocks": []} for page in doc.get("pages", [])],
    }
    assert count == 0
    assert decisions == []
    assert doc == before


@given(
    st.lists(st.sampled_from(("Body text.", TABLE, "Footer 9", " \n ")), max_size=8),
    st.lists(st.sampled_from(("Body text.", TABLE, "Footer 9", " \n ")), max_size=8),
)
def test_document_cleanup_composes_by_page_concatenation(
    first: list[str], second: list[str]
) -> None:
    left = _doc(first)
    right = _doc(second)
    combined = {**left, "pages": [*left["pages"], *right["pages"]]}
    before = deepcopy(combined)
    clean = partial(_clean_doc, footer_config=CONFIG)
    left_doc, left_count, left_decisions = clean(left)
    right_doc, right_count, right_decisions = clean(right)

    actual, count, decisions = clean(combined)

    assert actual == {**combined, "pages": [*left_doc["pages"], *right_doc["pages"]]}
    assert count == left_count + right_count
    assert decisions == [*left_decisions, *right_decisions]
    assert combined == before


@pytest.mark.parametrize(
    ("options", "kept"),
    (
        ({"known_footer_patterns": [r"Runtime \d+"]}, ["Footer 9", "Body text."]),
        ({"known_footer_patterns": []}, ["Runtime 9", "Body text."]),
        ({"never_footer_patterns": [r"Footer \d+"]}, ["Runtime 9", "Body text."]),
        ({"known_footer_patterns": [None, 7, r"Runtime \d+"]}, ["Footer 9", "Body text."]),
    ),
)
def test_runtime_pattern_override_precedence(options: dict[str, Any], kept: list[str]) -> None:
    step = _DetectPageArtifactsPass(known_footer_patterns=(r"Footer \d+",))
    artifact = Artifact(
        _doc(["Footer 9", "Runtime 9", "Body text."]),
        {"options": {step.name: options}},
    )
    before = deepcopy(artifact)
    result = step(artifact)
    assert [block["text"] for page in result.payload["pages"] for block in page["blocks"]] == kept
    assert artifact == before


def test_cleanup_does_not_invoke_callbacks_or_mutate_decision_cache() -> None:
    def forbidden_callback(*args: object) -> bool:
        pytest.fail("This cleanup pass must not prompt or invoke footer callbacks")

    step = _DetectPageArtifactsPass(
        known_footer_patterns=(r"Footer \d+",), footer_callback=forbidden_callback
    )
    cache = FooterDecisionCache()
    cache.set("Footer 9", False)
    config = FooterConfig(known_patterns=CONFIG.known_patterns, callback=forbidden_callback)
    assert _clean_doc(_doc(["Footer 9"]), config, cache)[1] == 1
    assert cache.get("Footer 9") is False

    # A callback-only runtime override replaces the instance config, as before.
    doc = _doc(["Footer 9"])
    result = step(Artifact(doc, {"options": {step.name: {"footer_callback": forbidden_callback}}}))
    assert result.payload == doc


def test_metric_update_retains_unrelated_and_prior_decisions() -> None:
    step = _DetectPageArtifactsPass(known_footer_patterns=(r"Footer \d+",))
    meta = {"metrics": {step.name: {"custom": 7, "footer_decisions": ["prior"]}, "other": {"n": 2}}}
    first = step(Artifact(_doc(["Footer 9"]), meta))
    second = step(first)
    assert first.meta["metrics"][step.name] == {
        "custom": 7,
        "blocks_cleaned": 1,
        "footer_decisions": [r"block matched footer pattern: Footer \d+"],
    }
    assert second.meta["metrics"][step.name] == {
        **first.meta["metrics"][step.name],
        "blocks_cleaned": 0,
    }
    assert second.meta["metrics"]["other"] == {"n": 2}
    assert meta["metrics"][step.name]["footer_decisions"] == ["prior"]


@pytest.mark.parametrize("invalid", (None, [], "bad", 7))
@pytest.mark.parametrize("container", ("options", "metrics"))
@pytest.mark.parametrize("nested", (False, True))
def test_malformed_metadata_containers_fail_without_mutation(
    invalid: object, container: str, nested: bool
) -> None:
    step = _DetectPageArtifactsPass()
    artifact = Artifact(_doc([TABLE]), {container: {step.name: invalid} if nested else invalid})
    before = deepcopy(artifact)
    with pytest.raises(TypeError, match="must be a mapping"):
        step(artifact)
    assert artifact == before


def test_invalid_runtime_regex_fails_without_mutating_inputs() -> None:
    step = _DetectPageArtifactsPass()
    artifact = Artifact(_doc([TABLE]), {"options": {step.name: {"known_footer_patterns": ["["]}}})
    before = deepcopy(artifact)
    with pytest.raises(re.error):
        step(artifact)
    assert artifact == before
