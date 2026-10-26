"""Composable iterators used to construct semantic chunk pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol, TypeAlias, cast

Block: TypeAlias = dict[str, Any]
Doc: TypeAlias = Mapping[str, Any]
Record: TypeAlias = tuple[int, Block, str]


class _MetadataBuilder(Protocol):
    """A chunk builder accepting the enumerated index by keyword."""

    def __call__(self, text: str, block: Block, page: int, /, *, index: int) -> dict[str, Any]: ...


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
    """Yield blocks lazily in input order, without changing their mappings.

    Page provenance prefers the block source, then the page source, then the
    page number; absent integer provenance falls back to the one-based position.
    Pages and blocks may each be one-shot iterables.
    """

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
    """Fold blocks eagerly, then split nonempty text lazily in record order.

    The injected fold owns merging; this iterator does not mutate input blocks.
    Keyword-only collaborators allow partially applying the stage to a stream.
    """

    merged: list[Record] = []
    for page, block in blocks:
        merged = fold(merged, (page, block, text_of(block)))
    return ((page, block, text) for page, block, raw in merged for text in split_fn(raw) if text)


def attach_headings(
    records: Iterable[Record],
    *,
    is_heading: Callable[[Block], bool],
    merge_block_text: Callable[[Sequence[str], str], str],
    same_page_only: bool = False,
) -> Iterator[Record]:
    """Attach consecutive headings to the next body using a copied body block.

    The first pending heading supplies page provenance. Unheaded records pass
    through unchanged. Trailing headings retain their own blocks and provenance.
    With same_page_only, page boundaries flush unresolved headings unchanged.
    """

    pending_page: int | None = None
    pending_texts: list[str] = []
    pending_records: list[Record] = []
    for page, block, text in records:
        if same_page_only and pending_records and pending_page != page:
            yield from pending_records
            pending_page, pending_texts, pending_records = None, [], []
        if is_heading(block):
            pending_page = pending_page or page
            pending_texts.append(text)
            pending_records.append((page, block, text))
            continue
        if not pending_texts:
            yield page, block, text
            continue
        heading_text = merge_block_text(pending_texts, text)
        # Mark block as containing a heading at its start
        merged_block = dict(block)
        merged_block["has_heading_prefix"] = True
        merged_block["source_blocks"] = tuple(
            source
            for _, contributor, _ in (*pending_records, (page, block, text))
            for source in contributor.get("source_blocks", (contributor,))
        )
        yield (pending_page or page), merged_block, heading_text
        pending_page, pending_texts, pending_records = None, [], []
    if same_page_only:
        yield from pending_records


def chunk_records(
    records: Iterable[Record],
    *,
    generate_metadata: bool,
    build_plain: Callable[[str], dict[str, Any]],
    build_with_meta: _MetadataBuilder,
) -> Iterator[dict[str, Any]]:
    """Build chunks lazily with stable zero-based IDs and optional metadata.

    Only the selected builder is called. Metadata builders receive ``index``
    by keyword, including when their other settings are partially applied.
    """

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
