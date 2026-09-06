"""Small, JSON-compatible contracts shared at pipeline boundaries.

The implementation still accepts legacy dictionaries with additional fields. These
types document the fields the canonical runner requires without changing the
serialized JSON shape.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class SourceRef(TypedDict, total=False):
    """Location information attached to an extracted block."""

    page: int
    index: int
    bbox: list[float]


class Block(TypedDict, total=False):
    """Minimal extracted block contract."""

    text: str
    type: str
    source: SourceRef
    inline_styles: list[object]


class Page(TypedDict):
    """Page containing ordered extracted blocks."""

    page: int | None
    blocks: list[Block]


class PageBlocks(TypedDict, total=False):
    """Canonical parser output."""

    type: Literal["page_blocks"]
    pages: list[Page]
    source_path: str
    document_metadata: dict[str, object]


class Chunk(TypedDict, total=False):
    """Canonical semantic chunk row before final emission."""

    text: str
    metadata: dict[str, object]
    meta: dict[str, object]


class Chunks(TypedDict):
    """Canonical semantic splitter output."""

    type: Literal["chunks"]
    items: list[Chunk]


class RunMetrics(TypedDict, total=False):
    """Stable top-level run-report metrics contract."""

    page_count: int
    chunk_count: int
    env: dict[str, object]


class ArtifactMeta(TypedDict, total=False):
    """Common artifact metadata; passes may preserve additional legacy keys."""

    input: str
    metrics: dict[str, object]
    options: dict[str, dict[str, object]]
    warnings: list[str]
