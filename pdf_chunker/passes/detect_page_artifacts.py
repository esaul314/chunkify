"""Cleanup page artifacts (e.g., flatten markdown-like tables).

This pass flattens leading markdown tables, then removes entire blocks matching
known footer patterns. Callback/cache settings remain accepted for compatibility;
this cleanup path does not classify ambiguous footers or invoke callbacks.
Configure via pipeline options:

    detect_page_artifacts:
      known_footer_patterns:
        - "Collective Wisdom.*\\d+"
        - "Book Title.*\\d{1,3}$"
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pdf_chunker import page_artifacts
from pdf_chunker.framework import Artifact, register
from pdf_chunker.interactive import (
    FooterCallback,
    FooterConfig,
    FooterDecisionCache,
)
from pdf_chunker.passes.metrics import with_metrics

Block = dict[str, Any]


def _clean_block(
    block: Block,
    page_num: int,
    footer_config: FooterConfig | None = None,
    footer_cache: FooterDecisionCache | None = None,
) -> tuple[Block, bool, list[str]]:
    """Clean a single block, returning (cleaned_block, changed, decisions)."""
    text = block.get("text", "")
    cleaned = page_artifacts._flatten_markdown_table(text)
    changed = cleaned != text
    decisions: list[str] = []

    # If we have footer config with known patterns, check if entire block is a footer
    if footer_config and footer_config.known_patterns:
        for pat in footer_config.known_patterns:
            if pat.search(cleaned):
                decisions.append(f"block matched footer pattern: {pat.pattern}")
                return {**block, "text": ""}, True, decisions

    return {**block, "text": cleaned}, changed, decisions


def _clean_page(
    page: dict[str, Any],
    footer_config: FooterConfig | None = None,
    footer_cache: FooterDecisionCache | None = None,
) -> tuple[dict[str, Any], int, list[str]]:
    """Clean all blocks on a page."""
    page_num = page.get("page", 0)
    blocks = page.get("blocks", [])
    if not blocks:
        return {**page, "blocks": []}, 0, []

    results = [_clean_block(b, page_num, footer_config, footer_cache) for b in blocks]
    cleaned_blocks = [r[0] for r in results]
    changed_count = sum(1 for r in results if r[1])
    all_decisions = [d for r in results for d in r[2]]

    # Filter out empty blocks (footer-matched)
    non_empty = [b for b in cleaned_blocks if b.get("text", "").strip()]
    return {**page, "blocks": non_empty}, changed_count, all_decisions


def _clean_doc(
    doc: dict[str, Any],
    footer_config: FooterConfig | None = None,
    footer_cache: FooterDecisionCache | None = None,
) -> tuple[dict[str, Any], int, list[str]]:
    """Map pages once, preserving order, then aggregate counts and decisions.

    Data is first so callers can partially apply footer settings. Input pages
    may be a one-shot iterable; the result is eager, like the canonical payload.
    Neither the document nor the supplied configuration/cache is mutated.
    """
    results = tuple(_clean_page(page, footer_config, footer_cache) for page in doc.get("pages", []))
    return (
        {**doc, "pages": [page for page, _, _ in results]},
        sum(changed for _, changed, _ in results),
        [decision for _, _, decisions in results for decision in decisions],
    )


@dataclass
class _DetectPageArtifactsPass:
    """Flatten tables and remove known footers without mutating input artifacts.

    Runtime known patterns or a callback replace the instance footer config.
    Empty patterns and never-pattern-only overrides retain that config. This
    path uses only known patterns; other footer settings are compatibility fields.
    Options and metrics must be mappings; malformed containers raise TypeError.
    """

    name: str = field(default="detect_page_artifacts", init=False)
    input_type: type = field(default=object, init=False)
    output_type: type = field(default=object, init=False)

    known_footer_patterns: tuple[str, ...] = ()
    never_footer_patterns: tuple[str, ...] = ()
    interactive: bool = False
    footer_callback: FooterCallback | None = None

    def __post_init__(self) -> None:
        """Compile regex patterns after initialization."""
        self._footer_config: FooterConfig | None = None
        self._footer_cache: FooterDecisionCache | None = None

        if self.known_footer_patterns or self.never_footer_patterns or self.footer_callback:
            callback = self.footer_callback

            self._footer_config = FooterConfig(
                known_patterns=tuple(
                    re.compile(p, re.IGNORECASE)
                    for p in self.known_footer_patterns
                    if isinstance(p, str)
                ),
                never_patterns=tuple(
                    re.compile(p, re.IGNORECASE)
                    for p in self.never_footer_patterns
                    if isinstance(p, str)
                ),
                callback=callback,
            )
            self._footer_cache = FooterDecisionCache()

    def __call__(self, a: Artifact[object]) -> Artifact[object]:
        payload = a.payload
        if not isinstance(payload, dict) or payload.get("type") != "page_blocks":
            return a

        # Check for runtime options that might override instance config
        options = (a.meta or {}).get("options", {})
        if not isinstance(options, Mapping):
            raise TypeError("artifact options must be a mapping")
        opts = options.get(self.name, {})
        if not isinstance(opts, Mapping):
            raise TypeError(f"options for pass {self.name!r} must be a mapping")
        footer_config = self._footer_config
        footer_cache = self._footer_cache

        # Allow runtime override of patterns
        if opts.get("known_footer_patterns") or opts.get("footer_callback"):
            footer_config = FooterConfig.from_dict(opts)
            footer_cache = FooterDecisionCache()

        cleaned_doc, changed, decisions = _clean_doc(payload, footer_config, footer_cache)

        meta = with_metrics(
            a.meta,
            pass_name=self.name,
            updates={
                "blocks_cleaned": changed,
                **({"footer_decisions": decisions} if decisions else {}),
            },
        )

        return Artifact(payload=cleaned_doc, meta=meta)


detect_page_artifacts = register(_DetectPageArtifactsPass())
