from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, cast

from pdf_chunker.framework import Artifact, register


def _apply_fixpoint(transform: Callable[[str], str], text: str) -> str:
    """Return ``text`` after repeatedly applying ``transform`` until stable."""

    updated = transform(text)
    return text if updated == text else _apply_fixpoint(transform, updated)


_NBSP_TRANSLATION = {
    ord("\u00a0"): " ",  # non-breaking space
    ord("\u2000"): " ",  # en quad
    ord("\u2001"): " ",  # em quad
    ord("\u2002"): " ",  # en space
    ord("\u2003"): " ",  # em space
    ord("\u2004"): " ",  # three-per-em space
    ord("\u2005"): " ",  # four-per-em space
    ord("\u2006"): " ",  # six-per-em space
    ord("\u2007"): " ",  # figure space
    ord("\u2008"): " ",  # punctuation space
    ord("\u2009"): " ",  # thin space
    ord("\u200a"): " ",  # hair space
    ord("\u202f"): " ",  # narrow no-break space
    ord("\ufeff"): "",  # zero-width no-break space
    ord("\u200b"): "",  # zero-width space
    ord("\u200c"): "",  # zero-width non-joiner
    ord("\u200d"): "",  # zero-width joiner
    ord("\u2060"): "",  # word joiner
}

_COLLAPSE_SPACES = re.compile(r" {2,}")


def _normalize_nbsp_like(text: str) -> str:
    """Convert NBSP-like characters to plain spaces and collapse multiples."""

    return _apply_fixpoint(
        lambda value: _COLLAPSE_SPACES.sub(" ", value.translate(_NBSP_TRANSLATION)), text
    )


def _patch_clean_paragraph() -> None:
    """Wrap ``clean_paragraph`` to normalize NBSP-like whitespace."""

    from pdf_chunker import text_cleaning as _text_cleaning

    clean_paragraph = getattr(_text_cleaning, "clean_paragraph", None)
    if clean_paragraph is None or getattr(clean_paragraph, "_normalizes_nbsp", False):
        return

    @wraps(clean_paragraph)
    def _wrapped(paragraph: str) -> str:
        cleaned = clean_paragraph(paragraph)
        return _normalize_nbsp_like(cleaned)

    _wrapped._normalizes_nbsp = True
    _text_cleaning.clean_paragraph = _wrapped


def _render_chunk(tokens: Sequence[str]) -> str:
    """Render ``tokens`` to text without post-processing side effects."""

    from pdf_chunker import splitter as _splitter

    return _splitter._detokenize_with_newlines(tuple(tokens))


def _token_windows(
    tokens: tuple[str, ...], chunk_size: int, overlap: int
) -> tuple[tuple[str, ...], ...]:
    """Return token windows for ``tokens`` respecting ``chunk_size`` and ``overlap``."""

    if len(tokens) <= chunk_size:
        return (tokens,)

    step = max(1, chunk_size - overlap)
    return tuple(tokens[index : index + chunk_size] for index in range(0, len(tokens), step))


def _patch_split_text_into_chunks() -> None:
    """Ensure detokenized fragments never collapse to empty strings."""

    from pdf_chunker import splitter as _splitter

    split_fn = getattr(_splitter, "_split_text_into_chunks", None)
    if split_fn is None or getattr(split_fn, "_preserves_raw_fragment", False):
        return

    @wraps(split_fn)
    def _wrapped(text: str, chunk_size: int, overlap: int) -> list[str]:
        if chunk_size <= 0:
            return []

        materialized = tuple(_splitter._tokenize_with_newlines(text))
        if not materialized:
            return []

        windows = _token_windows(materialized, chunk_size, overlap)
        chunks = [_render_chunk(window) for window in windows]
        return _splitter._dedupe_overlapping_chunks(chunks) or [_render_chunk(materialized)]

    _wrapped._preserves_raw_fragment = True
    _splitter._split_text_into_chunks = _wrapped


_patch_clean_paragraph()
_patch_split_text_into_chunks()


def _style_of(span: Any) -> str | None:
    """Return the style tag for ``span`` regardless of representation."""

    if hasattr(span, "style"):
        return span.style
    if isinstance(span, Mapping):
        return span.get("style")
    return None


def _inline_style_metrics(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate inline-style coverage and tag counts for ``pages``."""

    blocks = [block for page in pages for block in page.get("blocks", [])]
    total_blocks = len(blocks)
    if total_blocks == 0:
        return {
            "inline_style_block_ratio": 0.0,
            "inline_style_tag_counts": {},
        }

    styled_blocks = [block for block in blocks if block.get("inline_styles")]
    tag_counts = Counter(
        style
        for block in styled_blocks
        for style in (_style_of(span) for span in block.get("inline_styles") or ())
        if style is not None
    )
    return {
        "inline_style_block_ratio": len(styled_blocks) / total_blocks,
        "inline_style_tag_counts": dict(tag_counts),
    }


def _clean_block(
    block: dict[str, Any],
    footer_patterns: tuple[re.Pattern[str], ...] = (),
    inline_patterns: tuple[re.Pattern[str], ...] = (),
    footer_callback: Any = None,
    footer_cache: Any = None,
    *,
    interactive_heuristic: bool = False,
    source_path: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return a new block with normalized text and list of stripped footers."""
    from pdf_chunker import text_cleaning
    from pdf_chunker.inline_styles import (
        build_index_map,
        build_index_remapper,
        normalize_spans,
    )

    text = block.get("text", "")
    page = block.get("page", 0)
    stripped_footers: list[str] = []

    # 1. Strip inline footers using explicit patterns (no confirmation needed)
    #    These patterns were provided by the user, so we strip automatically.
    if inline_patterns:
        from pdf_chunker.interactive import strip_inline_footers

        # Pass callback=None so patterns strip without prompting
        text, stripped_footers = strip_inline_footers(
            text,
            inline_patterns,
            callback=None,  # No prompting for explicit patterns
            cache=footer_cache,
            page=page,
        )

    # 2. If interactive mode enabled, also run heuristic detection to catch
    #    any additional footers not matched by explicit patterns
    #    For EPUB files, heuristic detection is skipped automatically
    if interactive_heuristic and footer_callback is not None:
        from pdf_chunker.interactive import (
            is_epub_source,
            is_standalone_footer_candidate,
            strip_inline_footers_interactive,
        )

        # Skip heuristic footer detection for EPUB (no positional footers)
        if not is_epub_source(source_path):
            # Check for additional inline footers (within the text) using heuristics
            text, additional_stripped = strip_inline_footers_interactive(
                text,
                callback=footer_callback,
                cache=footer_cache,
                page=page,
                source_path=source_path,
            )
            stripped_footers.extend(additional_stripped)

            # Also check if the entire block is a standalone footer
            standalone = is_standalone_footer_candidate(text, source_path=source_path)
            if standalone:
                title, page_num = standalone
                ctx = {
                    "inline": False,
                    "standalone": True,
                    "page": page,
                    "detected_page_num": page_num,
                    "heuristic_confidence": 0.8,
                    "heuristic": True,
                }
                is_footer = footer_callback(text, page, ctx)
                if is_footer:
                    stripped_footers.append(text)
                    text = ""  # Remove entire block

    # 2. Check if entire block matches a known footer pattern (block-level)
    #    This is for blocks that are entirely footers (no body content).
    for pat in footer_patterns:
        if pat.fullmatch(text.strip()):
            # Entire block is a footer
            return {**block, "text": "", "inline_styles": None}, stripped_footers

    cleaned_text = text_cleaning.clean_text(text)

    original_styles = block.get("inline_styles")
    remapped_styles = original_styles
    if original_styles:
        remapper = build_index_remapper(build_index_map(text, cleaned_text))
        normalized = normalize_spans(original_styles, len(cleaned_text), remapper)
        remapped_styles = list(normalized)

    return (
        {
            **block,
            "text": cleaned_text,
            "inline_styles": remapped_styles,
        },
        stripped_footers,
    )


def _clean_page(
    page: dict[str, Any],
    footer_patterns: tuple[re.Pattern[str], ...] = (),
    inline_patterns: tuple[re.Pattern[str], ...] = (),
    footer_callback: Any = None,
    footer_cache: Any = None,
    *,
    interactive_heuristic: bool = False,
    source_path: str | None = None,
) -> tuple[dict[str, Any], int, list[str]]:
    """Clean all blocks in ``page`` and return the block count and stripped footers."""
    page_num = page.get("page", 0)
    all_stripped: list[str] = []
    cleaned_blocks = []

    for b in page.get("blocks", []):
        # Pass page number to block
        b_with_page = {**b, "page": page_num}
        cleaned_block, stripped = _clean_block(
            b_with_page,
            footer_patterns,
            inline_patterns,
            footer_callback,
            footer_cache,
            interactive_heuristic=interactive_heuristic,
            source_path=source_path,
        )
        all_stripped.extend(stripped)
        cleaned_blocks.append(cleaned_block)

    # Filter out blocks that became empty after footer stripping
    non_empty = [b for b in cleaned_blocks if b.get("text", "").strip()]
    return {**page, "blocks": non_empty}, len(non_empty), all_stripped


def _clean_doc(
    doc: dict[str, Any],
    footer_patterns: tuple[re.Pattern[str], ...] = (),
    inline_patterns: tuple[re.Pattern[str], ...] = (),
    footer_callback: Any = None,
    footer_cache: Any = None,
    *,
    interactive_heuristic: bool = False,
    source_path: str | None = None,
) -> tuple[dict[str, Any], int, dict[str, Any], list[str]]:
    """Clean document pages and aggregate block metrics."""
    all_stripped: list[str] = []
    pages_with_counts = []

    for p in doc.get("pages", []):
        cleaned_page, count, stripped = _clean_page(
            p,
            footer_patterns,
            inline_patterns,
            footer_callback,
            footer_cache,
            interactive_heuristic=interactive_heuristic,
            source_path=source_path,
        )
        pages_with_counts.append((cleaned_page, count))
        all_stripped.extend(stripped)

    pages = [p for p, _ in pages_with_counts]
    blocks = sum(c for _, c in pages_with_counts)
    style_metrics = _inline_style_metrics(pages)
    return {**doc, "pages": pages}, blocks, style_metrics, all_stripped


@dataclass
class _TextCleanPass:
    """Pass to clean and normalize text in document blocks.

    Attributes:
        footer_patterns: Regex strings that identify footers to strip
        interactive_footers: Whether to prompt for footer confirmation (reserved)
    """

    name: str = field(default="text_clean", init=False)
    input_type: type = field(default=object, init=False)
    output_type: type = field(default=object, init=False)

    footer_patterns: tuple[str, ...] = ()
    interactive_footers: bool = False

    # Internal state for interactive mode
    _footer_callback: Any = field(default=None, init=False, repr=False)
    _footer_cache: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile footer patterns and set up interactive callback."""
        self._compiled_patterns: tuple[re.Pattern[str], ...] = tuple(
            re.compile(p, re.IGNORECASE) for p in self.footer_patterns if isinstance(p, str)
        )
        # Compile inline patterns (for mid-text footers with \n\n prefix and midtext)
        from pdf_chunker.interactive import compile_footer_patterns

        self._inline_patterns: tuple[re.Pattern[str], ...] = compile_footer_patterns(
            self.footer_patterns, inline=True, midtext=True
        )

        # Set up interactive callback if enabled
        if self.interactive_footers:
            from pdf_chunker.interactive import FooterDecisionCache, make_cli_footer_prompt

            self._footer_callback = make_cli_footer_prompt(default_is_footer=True)
            self._footer_cache = FooterDecisionCache()

    def __call__(self, a: Artifact) -> Artifact:
        payload = a.payload
        block_count: int | None = None
        cleaned: str | dict[str, Any]

        # Check for runtime options
        opts = (a.meta or {}).get("options", {}).get("text_clean", {})
        runtime_patterns = opts.get("footer_patterns", ())
        runtime_interactive = opts.get("interactive_footers", False)

        # Use runtime patterns if provided
        block_patterns = self._compiled_patterns
        inline_patterns = self._inline_patterns
        footer_callback = self._footer_callback
        footer_cache = self._footer_cache

        # Interactive mode: always enable heuristic detection when interactive is set
        # Patterns provide precision for known footers, interactive catches the rest
        interactive_heuristic = self.interactive_footers or runtime_interactive

        if runtime_patterns and runtime_patterns != self.footer_patterns:
            from pdf_chunker.interactive import compile_footer_patterns

            block_patterns = tuple(
                re.compile(p, re.IGNORECASE) for p in runtime_patterns if isinstance(p, str)
            )
            inline_patterns = compile_footer_patterns(runtime_patterns, inline=True, midtext=True)

        # Set up interactive callback if enabled at runtime
        if runtime_interactive and footer_callback is None:
            from pdf_chunker.interactive import FooterDecisionCache, make_cli_footer_prompt

            footer_callback = make_cli_footer_prompt(default_is_footer=True)
            footer_cache = FooterDecisionCache()

        # Extract source_path for format-aware footer detection
        # (EPUB files don't have positional footers like PDFs)
        source_path: str | None = None
        if isinstance(payload, dict):
            source_path = payload.get("source_path")

        if isinstance(payload, str):
            from pdf_chunker.text_cleaning import _clean_text_impl

            cleaned = _clean_text_impl(payload)
        elif isinstance(payload, dict) and payload.get("type") == "page_blocks":
            typed_payload = cast(dict[str, Any], payload)
            cleaned, block_count, style_metrics, stripped_footers = _clean_doc(
                typed_payload,
                block_patterns,
                inline_patterns,
                footer_callback,
                footer_cache,
                interactive_heuristic=interactive_heuristic,
                source_path=source_path,
            )
        else:
            return a

        meta = dict(a.meta or {})
        metrics = meta.setdefault("metrics", {})
        metrics["normalized"] = True
        if block_count is not None:
            text_metrics = metrics.setdefault("text_clean", {})
            text_metrics["blocks"] = block_count
            text_metrics.update(style_metrics)
            if block_patterns or inline_patterns:
                text_metrics["footer_patterns_applied"] = len(block_patterns)
                text_metrics["inline_patterns_applied"] = len(inline_patterns)
            if interactive_heuristic:
                text_metrics["interactive_heuristic"] = True
            if stripped_footers:
                text_metrics["inline_footers_stripped"] = len(stripped_footers)
        return Artifact(payload=cleaned, meta=meta)


text_clean = register(_TextCleanPass())
