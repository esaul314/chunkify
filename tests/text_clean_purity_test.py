"""Text cleanup preserves artifacts and composes ordered page transformations."""

import re
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pdf_chunker import splitter, text_cleaning
from pdf_chunker.framework import Artifact
from pdf_chunker.inline_styles import InlineStyleSpan
from pdf_chunker.interactive import compile_footer_patterns
from pdf_chunker.passes.text_clean import (
    _clean_doc,
    _inline_style_metrics,
    _normalize_nbsp_like,
    _patch_clean_paragraph,
    _patch_split_text_into_chunks,
    _TextCleanPass,
    text_clean,
)


def _doc(texts: list[str]) -> dict[str, Any]:
    return {
        "type": "page_blocks",
        "source_path": "book.pdf",
        "pages": [
            {"page": index, "blocks": [{"text": text, "source": {"line": index}}]}
            for index, text in enumerate(texts, 1)
        ],
    }


@pytest.mark.parametrize("payload", ("Foo\u00a0bar", _doc(["Foo\u00a0bar", " "])))
def test_cleanup_preserves_nested_input_metrics(payload: object) -> None:
    artifact = Artifact(
        payload,
        {
            "source": "book.pdf",
            "options": {"other": [1]},
            "metrics": {"normalized": False, "text_clean": {"custom": 7}, "other": {"n": 2}},
        },
    )
    before = deepcopy(artifact)

    result = text_clean(artifact)

    assert artifact == before
    assert result == text_clean(artifact)
    assert result.meta["metrics"]["normalized"] is True
    assert result.meta["metrics"]["text_clean"]["custom"] == 7
    assert result.meta["metrics"]["other"] == {"n": 2}
    assert result.meta["options"] == {"other": [1]}


@pytest.mark.parametrize("meta", (None, {}))
@pytest.mark.parametrize("payload", ("Foo\u00a0bar", _doc(["Foo\u00a0bar"])))
def test_cleanup_creates_missing_metrics(payload: object, meta: dict[str, object] | None) -> None:
    result = text_clean(Artifact(payload, meta))
    assert result.meta["metrics"]["normalized"] is True
    assert ("text_clean" in result.meta["metrics"]) is isinstance(payload, dict)


@pytest.mark.parametrize("payload", (None, 7, {"type": "chunks", "items": []}))
def test_unsupported_payload_is_identity(payload: object) -> None:
    artifact = Artifact(payload, {"metrics": {"normalized": False}})
    assert text_clean(artifact) is artifact


@pytest.mark.parametrize(
    "doc", ({"type": "page_blocks"}, _doc([]), {"type": "page_blocks", "pages": [{"page": 3}]})
)
def test_empty_documents_report_zero_surviving_blocks(doc: dict[str, Any]) -> None:
    before = deepcopy(doc)
    result = text_clean(Artifact(doc))
    assert result.meta["metrics"] == {
        "normalized": True,
        "text_clean": {"blocks": 0, "inline_style_block_ratio": 0.0, "inline_style_tag_counts": {}},
    }
    assert doc == before


def test_string_cleanup_retains_document_metrics_without_creating_them() -> None:
    artifact = Artifact("Foo\u00a0bar", {"metrics": {"text_clean": None, "other": 7}})
    before = deepcopy(artifact)
    result = _TextCleanPass(footer_patterns=("Foo.*",))(artifact)
    assert result.payload == "Foo bar"
    assert result.meta == {"metrics": {"text_clean": None, "other": 7, "normalized": True}}
    assert artifact == before


def test_document_updates_do_not_overwrite_previous_result_metrics() -> None:
    step = _TextCleanPass(footer_patterns=(r"Footer",), interactive_footers=True)
    first = step(
        Artifact(_doc(["Body.\n\nFooter 9 Next."]), {"metrics": {"text_clean": {"custom": 7}}})
    )
    before = deepcopy(first)
    later_input = Artifact(_doc([]), first.meta)
    later_before = deepcopy(later_input)
    second = text_clean(later_input)

    assert first == before
    assert later_input == later_before
    assert first.meta["metrics"]["text_clean"]["blocks"] == 1
    assert second.meta["metrics"]["text_clean"] == {
        **first.meta["metrics"]["text_clean"],
        "blocks": 0,
        "inline_style_block_ratio": 0.0,
        "inline_style_tag_counts": {},
    }
    assert second.meta["metrics"]["text_clean"]["inline_footers_stripped"] == 1
    assert second.meta["metrics"]["text_clean"]["interactive_heuristic"] is True


def test_partial_cleanup_consumes_pages_and_blocks_once_in_order() -> None:
    pages = [
        {
            "page": 2,
            "label": "first",
            "blocks": [{"text": "Body.\n\nFooter 2 Next."}, {"text": " "}],
        },
        {"page": 1, "label": "second", "blocks": [{"text": "More.\n\nFooter 1 End.", "page": 99}]},
    ]
    before = deepcopy(pages)
    visited: list[object] = []

    def blocks(page: dict[str, Any]) -> Iterator[dict[str, Any]]:
        for block in page["blocks"]:
            visited.append(block["text"])
            yield block

    def page_stream() -> Iterator[dict[str, Any]]:
        for page in pages:
            visited.append(page["label"])
            yield {**page, "blocks": blocks(page)}

    clean = partial(
        _clean_doc,
        footer_patterns=(re.compile(r"Footer \d+"),),
        inline_patterns=compile_footer_patterns(("Footer",), inline=True, midtext=True),
    )
    result, count, styles, stripped = clean({"type": "page_blocks", "pages": page_stream()})

    assert visited == [
        "first",
        pages[0]["blocks"][0]["text"],
        " ",
        "second",
        pages[1]["blocks"][0]["text"],
    ]
    assert [page["page"] for page in result["pages"]] == [2, 1]
    assert [page["blocks"][0]["page"] for page in result["pages"]] == [2, 1]
    assert [page["blocks"][0]["text"] for page in result["pages"]] == ["Body. Next.", "More. End."]
    assert count == 2
    assert styles == {"inline_style_block_ratio": 0.0, "inline_style_tag_counts": {}}
    assert stripped == ["Footer 2", "Footer 1"]
    assert pages == before


@given(
    st.lists(st.sampled_from(("Body.", "Foo\u00a0bar", " ", "Footer 9")), max_size=5),
    st.lists(st.sampled_from(("Body.", "Foo\u00a0bar", " ", "Footer 9")), max_size=5),
)
def test_cleanup_commutes_with_ordered_page_concatenation(
    left: list[str], right: list[str]
) -> None:
    left_doc, right_doc = _doc(left), _doc(right)
    clean = partial(_clean_doc, footer_patterns=(re.compile(r"Footer \d+"),))
    left_clean, left_count, _, left_stripped = clean(left_doc)
    right_clean, right_count, _, right_stripped = clean(right_doc)
    combined = {"type": "page_blocks", "pages": left_doc["pages"] + right_doc["pages"]}
    before = deepcopy(combined)

    result, count, metrics, stripped = clean(combined)

    assert result["pages"] == left_clean["pages"] + right_clean["pages"]
    assert count == left_count + right_count
    assert stripped == left_stripped + right_stripped
    assert metrics == _inline_style_metrics(result["pages"])
    assert combined == before


def test_style_metrics_use_surviving_blocks_and_preserve_spans() -> None:
    doc = _doc(["Bold-\ntext", "Body.", "Footer 9", " "])
    span = InlineStyleSpan(0, 10, "bold")
    doc["pages"][0]["blocks"][0]["inline_styles"] = [span]
    before = deepcopy(doc)
    cleaned, count, metrics, _ = _clean_doc(doc, footer_patterns=(re.compile(r"Footer \d+"),))

    assert count == 2
    assert metrics == {"inline_style_block_ratio": 0.5, "inline_style_tag_counts": {"bold": 1}}
    assert cleaned["pages"][0]["blocks"][0]["inline_styles"] == [InlineStyleSpan(0, 9, "bold")]
    assert doc == before


@pytest.mark.parametrize(
    ("patterns", "remaining"),
    (((), "Runtime 9"), (("Runtime.*",), "Footer 9"), ((None, 7, "Runtime.*"), "Footer 9")),
)
def test_runtime_patterns_replace_only_when_nonempty(
    patterns: tuple[object, ...], remaining: str
) -> None:
    artifact = Artifact(
        _doc(["Footer 9", "Runtime 9"]), {"options": {"text_clean": {"footer_patterns": patterns}}}
    )
    before = deepcopy(artifact)
    result = _TextCleanPass(footer_patterns=("Footer.*",))(artifact)
    assert [block["text"] for page in result.payload["pages"] for block in page["blocks"]] == [
        remaining
    ]
    assert result.meta["metrics"]["text_clean"]["footer_patterns_applied"] == 1
    assert artifact == before


@pytest.mark.parametrize("invalid", (None, [], "bad", 7))
@pytest.mark.parametrize("container", ("options", "metrics"))
@pytest.mark.parametrize("nested", (False, True))
def test_malformed_containers_fail_explicitly(
    invalid: object, container: str, nested: bool
) -> None:
    artifact = Artifact(
        _doc(["Body."]), {container: {"text_clean": invalid} if nested else invalid}
    )
    before = deepcopy(artifact)
    with pytest.raises(TypeError, match="must be a mapping"):
        text_clean(artifact)
    assert artifact == before


def test_invalid_runtime_regex_propagates_without_mutating_input() -> None:
    artifact = Artifact(_doc(["Body."]), {"options": {"text_clean": {"footer_patterns": ("[",)}}})
    before = deepcopy(artifact)
    with pytest.raises(re.error):
        text_clean(artifact)
    assert artifact == before


@pytest.mark.parametrize("invalid", (None, [], "bad", 7))
def test_string_cleanup_rejects_malformed_outer_metrics(invalid: object) -> None:
    artifact = Artifact("Body.", {"metrics": invalid})
    before = deepcopy(artifact)
    with pytest.raises(TypeError, match="artifact metrics must be a mapping"):
        text_clean(artifact)
    assert artifact == before


@pytest.mark.parametrize("source_path", ("book.pdf", "book.epub", "episode.txt"))
def test_injected_callback_order_and_format_policy_are_preserved(source_path: str) -> None:
    calls: list[tuple[str, int]] = []

    def callback(text: str, page: int, context: dict[str, Any]) -> bool:
        calls.append((text, page))
        assert context["heuristic"] is True
        return True

    doc = _doc(["Chapter One 2", "Chapter Two 1"])
    doc["source_path"] = source_path
    artifact = Artifact(
        doc, {"options": {"text_clean": {"interactive_footers": True, "footer_callback": callback}}}
    )
    before = deepcopy(artifact)
    result = text_clean(artifact)
    assert calls == (
        [("Chapter One 2", 1), ("Chapter Two 1", 2)] if source_path.endswith(".pdf") else []
    )
    assert result.meta["metrics"]["text_clean"]["blocks"] == (0 if calls else 2)
    assert artifact == before


def test_compatibility_wrappers_are_marked_and_installed_once() -> None:
    cleaner, split = text_cleaning.clean_paragraph, splitter._split_text_into_chunks
    _patch_clean_paragraph()
    _patch_split_text_into_chunks()
    assert text_cleaning.clean_paragraph is cleaner
    assert splitter._split_text_into_chunks is split
    assert cleaner.__dict__["_normalizes_nbsp"] is True
    assert split.__dict__["_preserves_raw_fragment"] is True
    assert cleaner("Foo\u00a0bar") == "Foo bar"
    assert split("Body text.", 0, 0) == []
    assert split("", 2, 0) == []


def test_compatibility_wrapper_installation_preserves_markers_and_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cleaner(text: str) -> str:
        return text

    def split(text: str, chunk_size: int, overlap: int) -> list[str]:
        return ["unused"]

    monkeypatch.setattr(text_cleaning, "clean_paragraph", cleaner)
    monkeypatch.setattr(splitter, "_split_text_into_chunks", split)
    # Rendering has separate coverage; isolate wrapper installation/windowing.
    monkeypatch.setattr(splitter, "_detokenize_with_newlines", lambda tokens: " ".join(tokens))
    _patch_clean_paragraph()
    _patch_split_text_into_chunks()
    wrapped_cleaner = text_cleaning.clean_paragraph
    wrapped_split = splitter._split_text_into_chunks

    assert wrapped_cleaner.__wrapped__ is cleaner
    assert wrapped_split.__wrapped__ is split
    assert wrapped_cleaner.__dict__["_normalizes_nbsp"] is True
    assert wrapped_split.__dict__["_preserves_raw_fragment"] is True
    assert wrapped_cleaner("Foo\u00a0bar") == "Foo bar"
    assert wrapped_split("Body text.", 10, 0) == ["Body text."]
    monkeypatch.setattr(splitter, "_dedupe_overlapping_chunks", lambda chunks: [])
    assert wrapped_split("Body text.", 10, 0) == ["Body text."]
    _patch_clean_paragraph()
    _patch_split_text_into_chunks()
    assert text_cleaning.clean_paragraph is wrapped_cleaner
    assert splitter._split_text_into_chunks is wrapped_split


@given(st.text(alphabet="abc \u00a0\u2003\u200b\u2060", max_size=80))
def test_whitespace_normalization_is_idempotent(text: str) -> None:
    normalized = _normalize_nbsp_like(text)
    assert _normalize_nbsp_like(normalized) == normalized
