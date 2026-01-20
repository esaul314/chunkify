"""Cleanup page artifacts (e.g., flatten markdown-like tables).

This pass provides optional interactive footer confirmation for ambiguous cases.
Configure via pipeline options:

    detect_page_artifacts:
      known_footer_patterns:
        - "Collective Wisdom.*\\d+"
        - "Book Title.*\\d{1,3}$"
      never_footer_patterns:
        - "^Chapter \\d+"
      interactive: true  # Enable interactive prompts (CLI only)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable, Dict, Mapping, Tuple

from pdf_chunker.framework import Artifact, register
from pdf_chunker import page_artifacts
from pdf_chunker.interactive import (
    FooterCallback,
    FooterConfig,
    FooterDecisionCache,
    classify_footer,
)

Block = Dict[str, Any]


def _clean_block(
    block: Block,
    page_num: int,
    footer_config: FooterConfig | None = None,
    footer_cache: FooterDecisionCache | None = None,
) -> Tuple[Block, bool, list[str]]:
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
    page: Dict[str, Any],
    footer_config: FooterConfig | None = None,
    footer_cache: FooterDecisionCache | None = None,
) -> Tuple[Dict[str, Any], int, list[str]]:
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
    doc: Dict[str, Any],
    footer_config: FooterConfig | None = None,
    footer_cache: FooterDecisionCache | None = None,
) -> Tuple[Dict[str, Any], int, list[str]]:
    """Clean all pages in a document."""
    all_decisions: list[str] = []

    def step(
        acc: Tuple[list[Dict[str, Any]], int],
        page: Dict[str, Any],
    ) -> Tuple[list[Dict[str, Any]], int]:
        pages, total = acc
        cleaned, changed, decisions = _clean_page(page, footer_config, footer_cache)
        all_decisions.extend(decisions)
        return [*pages, cleaned], total + changed

    pages, total = reduce(step, doc.get("pages", []), ([], 0))
    return {**doc, "pages": pages}, total, all_decisions


@dataclass
class _DetectPageArtifactsPass:
    """Pass to detect and remove page artifacts (headers, footers, etc.).

    Attributes:
        known_footer_patterns: Regex patterns that always identify footers
        never_footer_patterns: Regex patterns for text that is never a footer
        interactive: Enable interactive prompts for ambiguous cases
        footer_callback: Optional callback for programmatic confirmation
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
            if self.interactive and callback is None:
                # Import here to avoid circular deps; CLI sets up the prompt
                from pdf_chunker.interactive import make_cli_footer_prompt

                callback = make_cli_footer_prompt()

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

    def __call__(self, a: Artifact) -> Artifact:
        payload = a.payload
        if not isinstance(payload, dict) or payload.get("type") != "page_blocks":
            return a

        # Check for runtime options that might override instance config
        opts = (a.meta or {}).get("options", {}).get("detect_page_artifacts", {})
        footer_config = self._footer_config
        footer_cache = self._footer_cache

        # Allow runtime override of patterns
        if opts.get("known_footer_patterns") or opts.get("footer_callback"):
            footer_config = FooterConfig.from_dict(opts)
            footer_cache = FooterDecisionCache()

        cleaned_doc, changed, decisions = _clean_doc(payload, footer_config, footer_cache)

        meta = dict(a.meta or {})
        metrics = meta.setdefault("metrics", {})
        pass_metrics = metrics.setdefault("detect_page_artifacts", {})
        pass_metrics["blocks_cleaned"] = changed
        if decisions:
            pass_metrics["footer_decisions"] = decisions

        return Artifact(payload=cleaned_doc, meta=meta)


detect_page_artifacts = register(_DetectPageArtifactsPass())
