"""List detection pass.

Annotates ``page_blocks`` documents with list metadata using a configurable
bullet strategy. Blocks that start with bullets or numbered markers, or
continue such items, are marked as ``list_item`` with a corresponding
``list_kind``.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import reduce
from itertools import chain
from typing import Any

from pdf_chunker.framework import Artifact, register
from pdf_chunker.passes.metrics import with_metrics
from pdf_chunker.strategies.bullets import (
    BULLET_CHARS,
    BULLET_CHARS_ESC,
    HYPHEN_BULLET_PREFIX,
    NUMBERED_RE,
    BulletHeuristicStrategy,
    default_bullet_strategy,
)

Block = dict[str, Any]


def _list_kind(strategy: BulletHeuristicStrategy, curr: str, prev: str | None) -> str | None:
    if strategy.starts_with_bullet(curr) or (prev and strategy.is_bullet_continuation(prev, curr)):
        return "bullet"
    if strategy.starts_with_number(curr) or (
        prev and strategy.is_numbered_continuation(prev, curr)
    ):
        return "numbered"
    return None


def _annotate(strategy: BulletHeuristicStrategy, prev: Block | None, block: Block) -> Block:
    prev_text = prev.get("text") if prev else None
    kind = _list_kind(strategy, block.get("text", ""), prev_text)
    return {**block, "type": "list_item", "list_kind": kind} if kind else block


def _annotate_blocks(blocks: Iterable[Block], strategy: BulletHeuristicStrategy) -> list[Block]:
    """Fold one block stream with page-local continuation state, without mutation."""

    def step(
        state: tuple[Block | None, list[Block]], block: Block
    ) -> tuple[Block | None, list[Block]]:
        prev, acc = state
        annotated = _annotate(strategy, prev, block)
        return annotated, [*acc, annotated]

    initial: tuple[Block | None, list[Block]] = (None, [])
    return reduce(step, blocks, initial)[1]


def _annotate_page(page: dict[str, Any], strategy: BulletHeuristicStrategy) -> dict[str, Any]:
    return {**page, "blocks": _annotate_blocks(page.get("blocks", []), strategy)}


def _count_kinds(blocks: Iterable[Block]) -> dict[str, int]:
    return {
        "bullet_items": sum(b.get("list_kind") == "bullet" for b in blocks),
        "numbered_items": sum(b.get("list_kind") == "numbered" for b in blocks),
    }


def _annotate_doc(
    doc: dict[str, Any], strategy: BulletHeuristicStrategy
) -> tuple[dict[str, Any], dict[str, int]]:
    pages = [_annotate_page(p, strategy) for p in doc.get("pages", [])]
    blocks = list(chain.from_iterable(p.get("blocks", []) for p in pages))
    return {**doc, "pages": pages}, _count_kinds(blocks)


class _ListDetectPass:
    name = "list_detect"
    input_type = dict
    output_type = dict

    def __init__(self, strategy: BulletHeuristicStrategy | None = None) -> None:
        self._strategy = strategy or default_bullet_strategy()

    def __call__(self, a: Artifact[object]) -> Artifact[object]:
        doc = a.payload
        if not isinstance(doc, dict) or doc.get("type") != "page_blocks":
            return a

        updated, counts = _annotate_doc(doc, self._strategy)
        meta = with_metrics(a.meta, pass_name=self.name, updates=counts)
        return Artifact(payload=updated, meta=meta)


list_detect = register(_ListDetectPass())


__all__ = [
    "BULLET_CHARS",
    "BULLET_CHARS_ESC",
    "HYPHEN_BULLET_PREFIX",
    "NUMBERED_RE",
    "list_detect",
]
