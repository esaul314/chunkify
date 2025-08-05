import logging
import textstat  # type: ignore[import]
from typing import Callable, Iterable
from itertools import accumulate
from haystack.dataclasses import Document
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import dataclass
from pdf_chunker.source_matchers import MATCHERS, Matcher


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int
    original_index: int


def _compute_readability(text: str) -> dict:
    """Computes readability scores and returns them as a dictionary matching canonical schema."""
    flesch_kincaid = textstat.flesch_kincaid_grade(text)

    # Map grade level to difficulty description
    if flesch_kincaid <= 6:
        difficulty = "elementary"
    elif flesch_kincaid <= 8:
        difficulty = "middle_school"
    elif flesch_kincaid <= 12:
        difficulty = "high_school"
    elif flesch_kincaid <= 16:
        difficulty = "college_level"
    else:
        difficulty = "graduate_level"

    return {"flesch_kincaid_grade": flesch_kincaid, "difficulty": difficulty}


def _generate_chunk_id(filename: str, page: int, chunk_index: int) -> str:
    """Generates a unique chunk ID using underscores as separators."""
    # Ensure filename does not contain underscores that would break the pattern
    # (but preserve the extension)
    # If page is None or 0, use 0
    page_part = page if page is not None else 0
    return f"{filename}_p{page_part}_c{chunk_index}"


def _truncate_chunk(text: str, max_chunk_size: int = 8000) -> str:
    """Truncate ``text`` to ``max_chunk_size`` using soft boundaries."""
    if len(text) <= max_chunk_size:
        return text

    logger.warning("chunk oversized (%s chars), applying strict truncation", len(text))

    truncate_point = max_chunk_size - 100
    sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]

    strategies = (
        (
            "sentence boundary",
            lambda: (
                (
                    best := max(
                        (
                            text.rfind(e, 0, truncate_point)
                            for e in sentence_endings
                            if text.rfind(e, 0, truncate_point) > truncate_point * 0.7
                        ),
                        default=-1,
                    )
                )
                > 0
                and text[: best + 1].strip()
            ),
        ),
        (
            "paragraph break",
            lambda: (
                (idx := text.rfind("\n\n", 0, truncate_point)) > truncate_point * 0.7
                and text[:idx].strip()
            ),
        ),
        (
            "word boundary",
            lambda: (
                (idx := text.rfind(" ", 0, truncate_point)) > truncate_point * 0.8
                and text[:idx].strip()
            ),
        ),
    )

    truncated, label = next(
        ((result, lbl) for lbl, fn in strategies if (result := fn())),
        (text[:truncate_point].strip(), "character limit"),
    )

    logger.debug("truncated at %s to %s chars", label, len(truncated))
    return truncated


def _enrich_chunk(
    text: str,
    perform_ai_enrichment: bool,
    enrichment_fn: Callable[[str], dict] | None,
) -> dict:
    """Perform optional AI enrichment using ``enrichment_fn``."""
    if not perform_ai_enrichment or enrichment_fn is None:
        return {"classification": "disabled", "tags": []}
    try:
        result = enrichment_fn(text)
        return {
            "classification": result.get("classification", "unclassified"),
            "tags": result.get("tags", []),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("AI enrichment failed: %s", exc)
        return {"classification": "error", "tags": []}


def _build_metadata(
    text: str,
    source_block: dict,
    chunk_index: int,
    utterance_type: dict,
) -> dict:
    """Construct metadata object for a chunk."""
    filename = source_block.get("source", {}).get("filename", "Unknown")
    page = source_block.get("source", {}).get("page", 0)
    location = source_block.get("source", {}).get("location")

    location_value = (
        None if location is None and filename.lower().endswith(".pdf") else location
    )
    page_value = page if isinstance(page, int) and page > 0 else None

    metadata = {
        "source": filename,
        "chunk_id": _generate_chunk_id(
            filename, page_value if page_value is not None else 0, chunk_index
        ),
        "page": page_value,
        "location": location_value,
        "block_type": source_block.get("type", "paragraph"),
        "language": source_block.get("language", "un"),
        "readability": _compute_readability(text),
        "utterance_type": utterance_type,
        "importance": "medium",
    }
    return {k: v for k, v in metadata.items() if k != "location" or v is None or v}


def process_chunk(
    chunk: Document,
    chunk_index: int,
    *,
    generate_metadata: bool,
    perform_ai_enrichment: bool,
    enrichment_fn: Callable[[str], dict] | None,
    char_map: dict,
    original_blocks: list[dict],
) -> dict | None:
    """Process a single chunk and return enriched metadata if requested."""
    logger.debug("process_chunk() ENTRY - chunk %s", chunk_index)

    final_text = _truncate_chunk((chunk.content or "").strip())
    if not final_text:
        logger.debug("process_chunk() EXIT - chunk %s - EMPTY CONTENT", chunk_index)
        return None

    if not generate_metadata:
        logger.debug(
            "process_chunk() EXIT - chunk %s - NO METADATA MODE",
            chunk_index,
        )
        return {"text": final_text}

    source_block = _find_source_block(chunk, char_map, original_blocks)
    if not source_block:
        logger.debug(
            "process_chunk() EXIT - chunk %s - NO SOURCE BLOCK FOUND",
            chunk_index,
        )
        return None

    utterance_type = _enrich_chunk(final_text, perform_ai_enrichment, enrichment_fn)
    metadata = _build_metadata(
        final_text,
        source_block,
        chunk_index,
        utterance_type,
    )

    result = {"text": final_text, "metadata": metadata}
    logger.debug(
        "process_chunk() EXIT - chunk %s SUCCESS - result has %s chars",
        chunk_index,
        len(result.get("text", "")),
    )
    return result


def format_chunks_with_metadata(
    haystack_chunks: list[Document],
    original_blocks: list[dict],
    generate_metadata: bool = True,
    perform_ai_enrichment: bool = False,
    enrichment_fn: Callable[[str], dict] | None = None,
    max_workers: int = 10,
    min_chunk_size: int | None = None,
    enable_dialogue_detection: bool = True,
) -> list[dict]:
    """
    Formats final chunks, enriching them in parallel with detailed metadata.
    Follows the canonical schema from README.ai with all required fields.
    """
    logger.debug(
        "format_chunks_with_metadata called with %s chunks and %s original blocks",
        len(haystack_chunks),
        len(original_blocks),
    )
    logger.debug(
        "min_chunk_size=%s, enable_dialogue_detection=%s",
        min_chunk_size,
        enable_dialogue_detection,
    )

    # Debug: Check what pages are in the original blocks
    original_pages = {
        block.get("source", {}).get("page")
        for block in original_blocks
        if block.get("source", {}).get("page")
    }
    logger.debug("Original blocks contain pages: %s", sorted(original_pages))

    char_map = _build_char_map(original_blocks)

    processor = partial(
        process_chunk,
        generate_metadata=generate_metadata,
        perform_ai_enrichment=perform_ai_enrichment,
        enrichment_fn=enrichment_fn,
        char_map=char_map,
        original_blocks=original_blocks,
    )

    if perform_ai_enrichment:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_chunks = [
                result
                for result in executor.map(
                    lambda p: processor(chunk=p[1], chunk_index=p[0]),
                    enumerate(haystack_chunks),
                )
                if result
            ]
    else:
        processed_chunks = [
            result
            for result in map(
                lambda p: processor(chunk=p[1], chunk_index=p[0]),
                enumerate(haystack_chunks),
            )
            if result
        ]

    return processed_chunks


def _build_char_map(blocks: list[dict]) -> dict:
    """Build a character position mapping for locating chunks in original blocks."""
    if not blocks:
        logger.debug("_build_char_map called with empty blocks list")
        return {"char_positions": ()}

    logger.debug("Building character map for %s blocks", len(blocks))

    block_pages = {
        block.get("source", {}).get("page")
        for block in blocks
        if block.get("source", {}).get("page")
    }
    logger.debug("Character map includes pages: %s", sorted(block_pages))

    lengths = (len(block["text"]) + 2 for block in blocks[:-1])
    starts = list(accumulate(lengths, initial=0))

    for i, (block, start) in enumerate(zip(blocks, starts)):
        text_len = len(block["text"])
        page = block.get("source", {}).get("page", "unknown")
        logger.debug(
            "Block %s (page %s): %s chars at position %s-%s",
            i,
            page,
            text_len,
            start,
            start + text_len,
        )

    char_positions = tuple(
        CharSpan(start, start + len(block["text"]), i)
        for i, (block, start) in enumerate(zip(blocks, starts))
    )

    logger.debug("Character map built with %s entries", len(char_positions))
    return {"char_positions": char_positions}


def _ensure_source(block: dict) -> dict:
    if "source" not in block or not isinstance(block["source"], dict):
        block["source"] = {}
    block["source"].setdefault("filename", "unknown")
    if not isinstance(block["source"].get("page"), int):
        block["source"]["page"] = None
    block["source"].setdefault("location", None)
    return block


def _match_source_block(
    chunk_start: str,
    original_blocks: list[dict],
    matchers: Iterable[tuple[str, Matcher]],
) -> tuple[dict | None, str | None]:
    return next(
        (
            (block, label)
            for label, predicate in matchers
            for block in original_blocks
            if predicate(chunk_start, block, original_blocks)
        ),
        (None, None),
    )


def _difflib_fallback(chunk_start: str, original_blocks: list[dict]) -> dict | None:
    try:
        import difflib

        block_starts = [block.get("text", "")[:50] for block in original_blocks]
        matches = difflib.get_close_matches(chunk_start, block_starts, n=1, cutoff=0.6)
        if matches:
            idx = block_starts.index(matches[0])
            block = original_blocks[idx]
            logger.debug(
                "Fallback: difflib matched block %s on page %s",
                idx,
                block.get("source", {}).get("page", None),
            )
            return block
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("difflib fallback failed: %s", exc)
    return None


def _find_source_block(
    chunk: Document,
    _char_map: dict,
    original_blocks: list[dict],
    *,
    matchers: Iterable[tuple[str, Matcher]] = MATCHERS,
) -> dict | None:
    """Find the original source block for a chunk using matcher strategies."""

    if not (chunk and chunk.content and original_blocks):
        logger.debug(
            "_find_source_block early return - chunk: %s, content: %s, blocks: %s",
            bool(chunk),
            bool(chunk.content if chunk else False),
            len(original_blocks) if original_blocks else 0,
        )
        return None

    chunk_text = chunk.content.strip()
    chunk_start = chunk_text[:50].replace("\n", " ").strip()
    logger.debug(
        "Looking for source block for chunk starting with: '%s...'", chunk_start
    )

    match, label = _match_source_block(chunk_start, original_blocks, matchers)
    if match:
        match = _ensure_source(match)
        logger.debug(
            "Found matching source block on page %s (%s)",
            match["source"].get("page", None),
            label,
        )
        return match

    block = _difflib_fallback(chunk_start, original_blocks)
    if block:
        return _ensure_source(block)

    block = _ensure_source(original_blocks[0])
    logger.warning(
        "No matching source block found for chunk starting with: '%s...'. Mapping to first block.",
        chunk_start,
    )
    return block
