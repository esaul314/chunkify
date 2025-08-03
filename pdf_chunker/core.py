"""Core orchestration logic for PDF chunking."""

from __future__ import annotations

from typing import Iterable, List, Set

from haystack.dataclasses import Document

from .ai_enrichment import init_llm
from .splitter import semantic_chunker
from .utils import format_chunks_with_metadata as utils_format_chunks_with_metadata

import logging

logger = logging.getLogger(__name__)


def parse_exclusions(exclude_pages: str | None) -> Set[int]:
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


def extract_blocks(filepath: str, exclude_pages: str | None) -> List[dict]:
    """Extract structured text blocks from the PDF."""
    from .pdf_parsing import extract_text_blocks_from_pdf

    logger.debug("Starting PDF extraction for %s", filepath)
    blocks = extract_text_blocks_from_pdf(filepath, exclude_pages=exclude_pages)
    logger.debug("PDF extraction complete: %d blocks", len(blocks))
    for i, block in enumerate(blocks[:3]):
        text_preview = block.get("text", "")[:100].replace("\n", "\\n")
        logger.debug("Block %d text preview: %r", i, text_preview)
    return blocks


def filter_blocks(blocks: Iterable[dict], excluded_pages: Set[int]) -> List[dict]:
    """Remove blocks that originate from excluded pages."""
    filtered = [
        block
        for block in blocks
        if block.get("source", {}).get("page") not in excluded_pages
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
        logger.error(
            "Excluded pages still present after filtering: %s", sorted(leaked_pages)
        )
    else:
        logger.debug("No excluded pages found in structured blocks")
    return filtered


def chunk_text(
    blocks: Iterable[dict],
    chunk_size: int,
    overlap: int,
    *,
    min_chunk_size: int,
    enable_dialogue_detection: bool,
) -> List[str]:
    """Chunk blocks of text into semantic units."""
    full_text = "\n\n".join(
        block.get("text", "") for block in blocks if block.get("text", "")
    )
    chunks = semantic_chunker(
        full_text,
        chunk_size,
        overlap,
        min_chunk_size=min_chunk_size,
        enable_dialogue_detection=enable_dialogue_detection,
    )
    logger.debug(
        "Semantic chunking with conversational text handling produced %d chunks",
        len(chunks),
    )
    if chunks:
        chunk_sizes = [len(c) for c in chunks]
        word_counts = [len(c.split()) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        avg_words = sum(word_counts) / len(word_counts)
        logger.debug(
            "Chunk size statistics: average %.0f characters (%.1f words)",
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
                for i in short_chunks:
                    preview = chunks[i][:50].replace("\n", " ")
                    logger.debug("Chunk %d: %d words - %r", i, word_counts[i], preview)
        if very_short_chunks:
            logger.warning("Very short chunks (≤3 words): %d", len(very_short_chunks))
            for i in very_short_chunks:
                preview = chunks[i][:50].replace("\n", " ")
                logger.debug("Chunk %d: %d words - %r", i, word_counts[i], preview)
        oversized_chunks = [i for i, size in enumerate(chunk_sizes) if size > 10000]
        if oversized_chunks:
            logger.warning("%d chunks exceed 10k characters", len(oversized_chunks))
            for i in oversized_chunks[:3]:
                logger.debug("Chunk %d: %d characters", i, chunk_sizes[i])
        extreme_chunks = [i for i, size in enumerate(chunk_sizes) if size > 25000]
        if extreme_chunks:
            logger.error("%d chunks exceed 25k characters!", len(extreme_chunks))
            for i in extreme_chunks:
                logger.debug("Chunk %d: %d characters", i, chunk_sizes[i])
    return chunks


def validate_chunks(
    final_chunks: List[dict],
    exclude_pages: str | None,
    generate_metadata: bool,
) -> List[dict]:
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
        final_sizes = [len(chunk.get("text", "")) for chunk in final_chunks]
        final_avg = sum(final_sizes) / len(final_sizes)
        final_max = max(final_sizes)
        final_min = min(final_sizes)
        logger.debug("Final chunk size statistics: average %.0f characters", final_avg)
        logger.debug("Maximum: %d characters", final_max)
        logger.debug("Minimum: %d characters", final_min)
        oversized_final = [i for i, size in enumerate(final_sizes) if size > 10000]
        if oversized_final:
            logger.error("%d final chunks exceed 10k characters!", len(oversized_final))
            for i in oversized_final[:3]:
                logger.debug("Final chunk %d: %d characters", i, final_sizes[i])
        extreme_final = [i for i, size in enumerate(final_sizes) if size > 25000]
        if extreme_final:
            logger.critical(
                "%d final chunks exceed 25k characters!", len(extreme_final)
            )
            for i in extreme_final:
                logger.debug("Final chunk %d: %d characters", i, final_sizes[i])
        for i, chunk in enumerate(final_chunks[:3]):
            text_len = len(chunk.get("text", ""))
            logger.debug("Final chunk %d: %d characters", i, text_len)
            if text_len > 10000:
                logger.error(
                    "Final chunk %d is still oversized (%d characters)!", i, text_len
                )
    return final_chunks


def process_document(
    filepath: str,
    chunk_size: int,
    overlap: int,
    *,
    generate_metadata: bool = True,
    ai_enrichment: bool = True,
    exclude_pages: str | None = None,
    min_chunk_size: int | None = None,
    enable_dialogue_detection: bool = True,
) -> List[dict]:
    """Process a document through extraction, chunking and enrichment."""

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

    perform_ai_enrichment = generate_metadata and ai_enrichment
    if perform_ai_enrichment:
        try:
            init_llm()
        except ValueError as exc:
            logger.warning("AI Enrichment disabled: %s", exc)
            perform_ai_enrichment = False

    excluded_pages = parse_exclusions(exclude_pages)
    blocks = extract_blocks(filepath, exclude_pages)
    filtered_blocks = filter_blocks(blocks, excluded_pages)
    haystack_chunks = chunk_text(
        filtered_blocks,
        chunk_size,
        overlap,
        min_chunk_size=min_chunk_size,
        enable_dialogue_detection=enable_dialogue_detection,
    )
    haystack_documents = [
        Document(content=text, id=f"chunk_{i}")
        for i, text in enumerate(haystack_chunks)
    ]
    final_chunks = utils_format_chunks_with_metadata(
        haystack_documents,
        filtered_blocks,
        generate_metadata=generate_metadata,
        perform_ai_enrichment=perform_ai_enrichment,
        max_workers=10,
        min_chunk_size=min_chunk_size,
        enable_dialogue_detection=enable_dialogue_detection,
    )
    return validate_chunks(final_chunks, exclude_pages, generate_metadata)
