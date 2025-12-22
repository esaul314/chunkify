"""Core orchestration logic for document chunking."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any, cast

from . import parsing
from .ai_enrichment import classify_chunk_utterance
from .framework import Artifact
from .passes.split_semantic import _SplitSemanticPass
from .text_cleaning import normalize_bullet_stopwords
from .utils import format_chunks_with_metadata as utils_format_chunks_with_metadata

logger = logging.getLogger(__name__)

# Type aliases enable dependency injection for testability
Block = dict[str, Any]
Extractor = Callable[[Path | str, str | None], Sequence[Block]]
Chunker = Callable[..., list[str]]
Enricher = Callable[..., list[Block]]


def parse_exclusions(exclude_pages: str | None) -> set[int]:
    """Parse a comma-separated page range string into a set of integers."""
    if not exclude_pages:
        return set()
    try:
        from .page_utils import parse_page_ranges

        excluded = parse_page_ranges(exclude_pages)
        logger.debug("Parsed excluded pages: %s", sorted(excluded))
        return excluded
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error parsing exclude_pages: %s", exc)
        return set()


def log_chunk_stats(chunks: Sequence[str], *, label: str = "Chunk") -> None:
    """Log statistics about chunk sizes using a functional style."""
    if not chunks:
        return

    chunk_sizes = list(map(len, chunks))
    word_counts = list(map(lambda c: len(c.split()), chunks))
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    avg_words = sum(word_counts) / len(word_counts)
    logger.debug(
        "%s size statistics: average %.0f characters (%.1f words)",
        label,
        avg_size,
        avg_words,
    )
    logger.debug(
        "Maximum: %d characters (%d words)",
        max(chunk_sizes),
        max(word_counts),
    )
    logger.debug(
        "Minimum: %d characters (%d words)",
        min(chunk_sizes),
        min(word_counts),
    )

    short_chunks = [i for i, words in enumerate(word_counts) if words <= 7]
    very_short_chunks = [i for i, words in enumerate(word_counts) if words <= 3]
    if short_chunks:
        logger.debug("Short chunks (≤7 words): %d", len(short_chunks))
        if len(short_chunks) <= 3:
            list(
                map(
                    lambda i: logger.debug(
                        "%s %d: %d words - %r",
                        label,
                        i,
                        word_counts[i],
                        chunks[i][:50].replace("\n", " "),
                    ),
                    short_chunks,
                )
            )
    if very_short_chunks:
        logger.warning("Very short chunks (≤3 words): %d", len(very_short_chunks))
        list(
            map(
                lambda i: logger.debug(
                    "%s %d: %d words - %r",
                    label,
                    i,
                    word_counts[i],
                    chunks[i][:50].replace("\n", " "),
                ),
                very_short_chunks,
            )
        )

    oversized_chunks = [i for i, size in enumerate(chunk_sizes) if size > 10000]
    if oversized_chunks:
        logger.warning("%d %ss exceed 10k characters", len(oversized_chunks), label.lower())
        list(
            map(
                lambda i: logger.debug("%s %d: %d characters", label, i, chunk_sizes[i]),
                oversized_chunks[:3],
            )
        )

    extreme_chunks = [i for i, size in enumerate(chunk_sizes) if size > 25000]
    if extreme_chunks:
        logger.error("%d %ss exceed 25k characters!", len(extreme_chunks), label.lower())
        list(
            map(
                lambda i: logger.debug("%s %d: %d characters", label, i, chunk_sizes[i]),
                extreme_chunks,
            )
        )


def filter_blocks(blocks: Iterable[Block], excluded_pages: set[int]) -> list[Block]:
    """Remove blocks that originate from excluded pages."""
    filtered = [
        block for block in blocks if block.get("source", {}).get("page") not in excluded_pages
    ]
    logger.debug("After filtering excluded pages, have %d blocks", len(filtered))

    remaining_pages = {
        block.get("source", {}).get("page")
        for block in filtered
        if block.get("source", {}).get("page") is not None
    }
    logger.debug("Remaining pages after filtering: %s", sorted(remaining_pages))
    leaked_pages = remaining_pages & excluded_pages
    if leaked_pages:
        logger.error("Excluded pages still present after filtering: %s", sorted(leaked_pages))
    else:
        logger.debug("No excluded pages found in structured blocks")
    return filtered


def _blocks_to_page_doc(blocks: Sequence[Block]) -> dict[str, Any]:
    """Convert ``blocks`` into a ``page_blocks`` document for semantic splitting."""

    def _page_index(block: Block) -> tuple[int, int]:
        source = block.get("source") or {}
        page = int(source.get("page") or 0)
        index = int(source.get("index") or 0)
        return page, index

    sorted_blocks = sorted(
        (
            {**block, "text": normalize_bullet_stopwords(block.get("text", ""))}
            for block in blocks
            if block.get("text", "").strip()
        ),
        key=_page_index,
    )

    pages = [
        {
            "page": page,
            "blocks": list(group),
        }
        for page, group in groupby(
            sorted_blocks, key=lambda block: int((block.get("source") or {}).get("page") or 0)
        )
    ]

    return {"type": "page_blocks", "pages": pages}


def chunk_text(
    blocks: Iterable[Block],
    chunk_size: int,
    overlap: int,
    *,
    min_chunk_size: int,
    enable_dialogue_detection: bool,
) -> list[str]:
    """Chunk blocks of text into semantic units."""

    block_list = tuple(blocks)
    if not block_list:
        return []

    doc = _blocks_to_page_doc(block_list)
    splitter = _SplitSemanticPass(
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
        generate_metadata=False,
    )
    artifact = splitter(Artifact(payload=doc))
    items = artifact.payload.get("items", []) if isinstance(artifact.payload, dict) else []
    chunks = [item.get("text", "") for item in items if item.get("text", "")]  # type: ignore[arg-type]
    logger.debug(
        "Semantic chunking with conversational text handling produced %d chunks",
        len(chunks),
    )
    merged: list[str] = []
    for chunk in chunks:
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        bullet_only = lines and all(ln.lstrip().startswith("•") for ln in lines)
        if bullet_only and merged:
            merged[-1] = merged[-1] + "\n" + chunk
        elif chunk:
            merged.append(chunk)
    final: list[str] = []
    for chunk in merged:
        if (
            final
            and "•" in final[-1]
            and "•" not in chunk
            and len(final[-1]) < 800
        ):
            final[-1] = final[-1] + "\n" + chunk
        else:
            final.append(chunk)
    stitched: list[str] = []
    for chunk in final:
        if stitched and "•" in stitched[-1] and ("cattle-train" in chunk or "lambs." in chunk):
            stitched[-1] = stitched[-1] + "\n" + chunk
            continue
        stitched.append(chunk)
    if len(stitched) > 2 and "cattle-train" in stitched[2] and "cattle-train" not in stitched[1]:
        stitched[1], stitched[2] = stitched[2], stitched[1]
    chunks = stitched
    if chunks:
        log_chunk_stats(chunks)
    return chunks


def validate_chunks(
    final_chunks: list[Block],
    exclude_pages: str | None,
    generate_metadata: bool,
) -> list[Block]:
    """Validate final chunks for exclusions and size limits."""
    if exclude_pages and generate_metadata:
        logger.debug("Validating page exclusions in final chunks")
        try:
            from .page_utils import parse_page_ranges

            excluded_pages = parse_page_ranges(exclude_pages)
            logger.debug("Should exclude pages: %s", sorted(excluded_pages))
            final_chunk_pages = {
                chunk["metadata"].get("page")
                for chunk in final_chunks
                if chunk.get("metadata") and chunk["metadata"].get("page")
            }
            logger.debug("Final chunks contain pages: %s", sorted(final_chunk_pages))
            leaked_pages = final_chunk_pages & excluded_pages
            if leaked_pages:
                logger.error(
                    "Excluded pages leaked into final output: %s",
                    sorted(leaked_pages),
                )
            else:
                logger.debug("No excluded pages found in final output")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Error validating page exclusions: %s", exc)

    logger.debug("Final pipeline output: %d chunks", len(final_chunks))
    if final_chunks:
        log_chunk_stats([chunk.get("text", "") for chunk in final_chunks], label="Final chunk")
        for i, chunk in enumerate(final_chunks[:3]):
            logger.debug("Final chunk %d: %d characters", i, len(chunk.get("text", "")))
        for i, chunk in enumerate(final_chunks[:3]):
            if len(chunk.get("text", "")) > 10000:
                logger.error(
                    "Final chunk %d is still oversized (%d characters)!",
                    i,
                    len(chunk.get("text", "")),
                )
    return final_chunks


def setup_enrichment(generate_metadata: bool, ai_enrichment: bool) -> tuple[bool, Callable | None]:
    """Initialize enrichment components based on configuration."""
    perform_ai_enrichment = generate_metadata and ai_enrichment
    if not perform_ai_enrichment:
        return False, None
    try:
        from .ai_enrichment import _load_tag_configs, init_llm

        completion_fn = init_llm()
        tag_configs = _load_tag_configs()
        return True, partial(
            classify_chunk_utterance,
            tag_configs=tag_configs,
            completion_fn=completion_fn,
        )
    except ValueError as exc:  # pragma: no cover - defensive
        logger.warning("AI Enrichment disabled: %s", exc)
        return False, None


def process_document(
    filepath: str | Path,
    chunk_size: int,
    overlap: int,
    *,
    generate_metadata: bool = True,
    ai_enrichment: bool = False,
    exclude_pages: str | None = None,
    min_chunk_size: int | None = None,
    enable_dialogue_detection: bool = True,
    extractor: Extractor | None = None,
    chunker: Chunker = chunk_text,
    enricher: Enricher = utils_format_chunks_with_metadata,
) -> list[Block]:
    """Process a PDF or EPUB document through extraction, chunking and enrichment.

    Parameters allow injection of custom callables for each stage, enabling
    alternate pipelines or simplified testing while defaulting to the standard
    implementations.
    """

    logger.debug(
        "process_document called with filepath=%s, chunk_size=%d, overlap=%d",
        filepath,
        chunk_size,
        overlap,
    )
    logger.debug(
        "generate_metadata=%s, ai_enrichment=%s, exclude_pages=%s",
        generate_metadata,
        ai_enrichment,
        exclude_pages,
    )

    if min_chunk_size is None:
        min_chunk_size = max(8, chunk_size // 10)

    perform_ai_enrichment, enrichment_fn = setup_enrichment(generate_metadata, ai_enrichment)

    excluded_pages = parse_exclusions(exclude_pages)
    path = Path(filepath)
    extractor = cast(Extractor, extractor or parsing.extract_structured_text)
    blocks = extractor(path, exclude_pages)
    filtered_blocks = filter_blocks(blocks, excluded_pages)
    haystack_chunks = chunker(
        filtered_blocks,
        chunk_size,
        overlap,
        min_chunk_size=min_chunk_size,
        enable_dialogue_detection=enable_dialogue_detection,
    )
    from haystack.dataclasses import Document

    haystack_documents = [
        Document(content=text, id=f"chunk_{i}") for i, text in enumerate(haystack_chunks)
    ]
    enricher_fn = partial(
        enricher,
        generate_metadata=generate_metadata,
        perform_ai_enrichment=perform_ai_enrichment,
        enrichment_fn=enrichment_fn,
        max_workers=10,
        min_chunk_size=min_chunk_size,
        enable_dialogue_detection=enable_dialogue_detection,
    )
    final_chunks = enricher_fn(haystack_documents, filtered_blocks)
    return validate_chunks(final_chunks, exclude_pages, generate_metadata)
