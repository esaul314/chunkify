"""Composable iterators used to construct semantic chunk pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import reduce
from typing import Any, cast

Block = Doc = Mapping[str, Any]
Record = tuple[int, Block, str]


def _page_from_source(value: Any) -> int | None:
    return (
        candidate
        if isinstance(value, Mapping) and isinstance((candidate := value.get("page")), int)
        else None
    )


def _resolve_page(block: Block, page: Mapping[str, Any], fallback: int) -> int:
    raw_page = page.get("page") if hasattr(page, "get") else None
    page_number = raw_page if isinstance(raw_page, int) else None
    return next(
        (
            candidate
            for candidate in (
                _page_from_source(block.get("source")),
                _page_from_source(page.get("source")),
                page_number,
            )
            if candidate is not None
        ),
        fallback,
    )


def iter_blocks(doc: Doc) -> Iterator[tuple[int, Block]]:
    pages = doc.get("pages", cast(Iterable[Mapping[str, Any]], ()))
    return (
        (_resolve_page(block, page, index + 1), block)
        for index, page in enumerate(pages)
        for block in page.get("blocks", ())
    )


def merge_adjacent_blocks(
    blocks: Iterable[tuple[int, Block]],
    *,
    text_of: Callable[[Block], str],
    fold: Callable[[list[Record], Record], list[Record]],
    split_fn: Callable[[str], Iterable[str]],
) -> Iterator[Record]:
    merged = reduce(
        fold,
        ((page, block, text_of(block)) for page, block in blocks),
        cast(list[Record], []),
    )
    return ((page, block, text) for page, block, raw in merged for text in split_fn(raw) if text)


def attach_headings(
    records: Iterable[Record],
    *,
    is_heading: Callable[[Block], bool],
    merge_block_text: Callable[[Sequence[str], str], str],
) -> Iterator[Record]:
    pending_page: int | None = None
    pending_texts: list[str] = []
    for page, block, text in records:
        if is_heading(block):
            pending_page = pending_page or page
            pending_texts.append(text)
            continue
        if not pending_texts:
            yield page, block, text
            continue
        heading_text = merge_block_text(pending_texts, text)
        # Mark block as containing a heading at its start
        merged_block = dict(block)
        merged_block["has_heading_prefix"] = True
        yield (pending_page or page), merged_block, heading_text
        pending_page, pending_texts = None, []


def chunk_records(
    records: Iterable[Record],
    *,
    generate_metadata: bool,
    build_plain: Callable[[str], dict[str, Any]],
    build_with_meta: Callable[[str, Block, int, int], dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    return (
        {
            "id": str(index),
            **(
                build_with_meta(text, block, page, index=index)
                if generate_metadata
                else build_plain(text)
            ),
        }
        for index, (page, block, text) in enumerate(records)
    )
