"""Characterize option resolution and the composable semantic iterators."""

from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import Any

import pytest

from pdf_chunker.passes.chunk_options import SplitMetrics, SplitOptions, derive_min_chunk_size
from pdf_chunker.passes.chunk_pipeline import (
    Block,
    Record,
    attach_headings,
    chunk_records,
    iter_blocks,
    merge_adjacent_blocks,
)


@pytest.mark.parametrize(
    ("chunk_size", "minimum", "expected"),
    ((400, None, 40), (40, None, 8), (0, None, 8), (400, 0, 0), (400, 12, 12)),
)
def test_minimum_size_is_derived_only_when_absent(
    chunk_size: int, minimum: int | None, expected: int
) -> None:
    assert derive_min_chunk_size(chunk_size, minimum) == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({}, SplitOptions(400, 100, 40)),
        ({"overlap": 25}, SplitOptions(400, 25, 40)),
        ({"chunk_size": 60}, SplitOptions(60, 100, 8)),
        ({"min_chunk_size": 0}, SplitOptions(400, 100, 0)),
        (
            {"chunk_size": "120", "overlap": "20", "min_chunk_size": "15"},
            SplitOptions(120, 20, 15),
        ),
    ),
)
def test_option_overrides_leave_inputs_unchanged(
    overrides: dict[str, Any], expected: SplitOptions
) -> None:
    options = SplitOptions.from_base(400, 100, None)
    meta = {"options": {"split_semantic": overrides, "text_clean": {"enabled": True}}}
    original = deepcopy(meta)

    assert options.with_meta(meta) == expected
    assert options == SplitOptions(400, 100, 40)
    assert meta == original


def test_split_metrics_merge_without_mutating_nested_inputs() -> None:
    meta: dict[str, Any] = {
        "metrics": {
            "pdf_parse": {"pages": 2},
            "split_semantic": {"chunks": 1, "soft_limit_hits": 3},
        },
        "source": "sample.pdf",
    }
    extra = {"overlap_trimmed": 2}
    original = deepcopy(meta)

    result = SplitMetrics(4, extra).apply(meta)

    assert result == {
        **original,
        "metrics": {
            "pdf_parse": {"pages": 2},
            "split_semantic": {"chunks": 4, "soft_limit_hits": 3, "overlap_trimmed": 2},
        },
    }
    assert meta == original
    assert extra == {"overlap_trimmed": 2}
    assert result["metrics"] is not meta["metrics"]
    assert result["metrics"]["split_semantic"] is not meta["metrics"]["split_semantic"]


@pytest.mark.parametrize(
    ("block_source", "page_source", "page_number", "expected"),
    (
        ({"page": 9}, {"page": 5}, 3, 9),
        ({"page": "invalid"}, {"page": 5}, 3, 5),
        (None, None, 3, 3),
        ({"page": None}, {"page": "invalid"}, "invalid", 1),
        ({"page": 0}, None, 3, 0),
    ),
)
def test_block_page_resolution_preserves_precedence(
    block_source: Any, page_source: Any, page_number: Any, expected: int
) -> None:
    block = {"text": "Body", "source": block_source}
    page = {"page": page_number, "source": page_source, "blocks": iter((block,))}
    records = iter_blocks({"pages": iter((page,))})

    assert list(records) == [(expected, block)]


def test_block_iteration_keeps_order_and_consumes_one_shot_inputs_lazily() -> None:
    first, second = {"text": "First"}, {"text": "Second"}
    blocks = [first, second]
    original = deepcopy(blocks)
    pages = iter(({"blocks": iter((first,))}, {"blocks": ()}, {"blocks": iter((second,))}))
    records = iter_blocks({"pages": pages})

    assert next(records) == (1, first)
    assert list(records) == [(3, second)]
    assert list(pages) == []
    assert blocks == original
    assert list(iter_blocks({})) == []


def _append_record(records: list[Record], record: Record) -> list[Record]:
    return [*records, record]


def _join_heading_texts(headings: Sequence[str], body: str) -> str:
    return "\n".join((*headings, body))


def _plain_chunk(text: str) -> dict[str, Any]:
    return {"text": text}


def _metadata_chunk(
    text: str, block: Block, page: int, *, index: int, source: str
) -> dict[str, Any]:
    return {
        "text": text,
        "meta": {"page": page, "index": index, "source": source, "kind": block.get("type")},
    }


def test_iterator_stages_compose_with_partial_application_without_mutating_blocks() -> None:
    blocks = [
        {"type": "heading", "text": "Chapter"},
        {"type": "heading", "text": "Section"},
        {"type": "paragraph", "text": "Body|"},
        {"type": "paragraph", "text": "Tail"},
    ]
    original = deepcopy(blocks)
    merge = partial(
        merge_adjacent_blocks,
        text_of=lambda block: str(block["text"]),
        fold=_append_record,
        split_fn=lambda text: iter(text.split("|")),
    )
    headings = partial(
        attach_headings,
        is_heading=lambda block: block.get("type") == "heading",
        merge_block_text=_join_heading_texts,
    )
    records = iter((page, block) for page, block in zip((1, 2, 2, 3), blocks, strict=True))

    result = list(headings(merge(records)))

    assert result == [
        (1, {**blocks[2], "has_heading_prefix": True}, "Chapter\nSection\nBody"),
        (3, blocks[3], "Tail"),
    ]
    assert blocks == original
    assert result[0][1] is not blocks[2]
    assert list(merge(iter(()))) == []


@pytest.mark.parametrize("generate_metadata", (False, True))
def test_chunk_builders_are_lazy_and_support_keyword_index_and_partial_application(
    generate_metadata: bool,
) -> None:
    block = {"type": "paragraph"}
    records = iter(((7, block, "First"), (9, block, "Second")))
    build = partial(
        chunk_records,
        generate_metadata=generate_metadata,
        build_plain=_plain_chunk,
        build_with_meta=partial(_metadata_chunk, source="sample.pdf"),
    )
    chunks = build(records)

    assert next(records) == (7, block, "First")  # Construction has not consumed input.
    expected: dict[str, Any] = {"id": "0", "text": "Second"}
    if generate_metadata:
        expected["meta"] = {"page": 9, "index": 0, "source": "sample.pdf", "kind": "paragraph"}
    assert list(chunks) == [expected]
    assert list(build(iter(()))) == []


def test_chunk_ids_and_metadata_indices_follow_record_order() -> None:
    block = {"type": "paragraph"}
    records = iter(((7, block, "First"), (9, block, "Second")))

    assert list(
        chunk_records(
            records,
            generate_metadata=True,
            build_plain=_plain_chunk,
            build_with_meta=partial(_metadata_chunk, source="sample.pdf"),
        )
    ) == [
        {
            "id": "0",
            "text": "First",
            "meta": {"page": 7, "index": 0, "source": "sample.pdf", "kind": "paragraph"},
        },
        {
            "id": "1",
            "text": "Second",
            "meta": {"page": 9, "index": 1, "source": "sample.pdf", "kind": "paragraph"},
        },
    ]
