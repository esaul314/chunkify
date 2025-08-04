import re
import textstat
from typing import Callable
from itertools import accumulate
from haystack.dataclasses import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


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
    """Truncate `text` to ``max_chunk_size`` using soft boundaries."""
    if len(text) <= max_chunk_size:
        return text

    import sys

    print(
        f"WARNING: chunk oversized ({len(text)} chars), applying strict truncation",
        file=sys.stderr,
    )

    truncate_point = max_chunk_size - 100
    sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]
    best_break = max(
        (
            text.rfind(ending, 0, truncate_point)
            for ending in sentence_endings
            if text.rfind(ending, 0, truncate_point) > truncate_point * 0.7
        ),
        default=-1,
    )
    if best_break > 0:
        truncated = text[: best_break + 1].strip()
        print(
            f"DEBUG: truncated at sentence boundary to {len(truncated)} chars",
            file=sys.stderr,
        )
        return truncated

    last_paragraph = text.rfind("\n\n", 0, truncate_point)
    if last_paragraph > truncate_point * 0.7:
        truncated = text[:last_paragraph].strip()
        print(
            f"DEBUG: truncated at paragraph break to {len(truncated)} chars",
            file=sys.stderr,
        )
        return truncated

    last_space = text.rfind(" ", 0, truncate_point)
    if last_space > truncate_point * 0.8:
        truncated = text[:last_space].strip()
        print(
            f"DEBUG: truncated at word boundary to {len(truncated)} chars",
            file=sys.stderr,
        )
        return truncated

    truncated = text[:truncate_point].strip()
    print(
        f"DEBUG: truncated at character limit to {len(truncated)} chars",
        file=sys.stderr,
    )
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
        import sys

        print(f"DEBUG: AI enrichment failed: {exc}", file=sys.stderr)
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


def format_chunks_with_metadata(
    haystack_chunks: list[Document],
    original_blocks: list[dict],
    generate_metadata: bool = True,
    perform_ai_enrichment: bool = False,
    enrichment_fn: Callable[[str], dict] | None = None,
    max_workers: int = 10,
    min_chunk_size: int = None,
    enable_dialogue_detection: bool = True,
) -> list[dict]:
    """
    Formats final chunks, enriching them in parallel with detailed metadata.
    Follows the canonical schema from README.ai with all required fields.
    """
    import sys

    print(
        f"DEBUG: format_chunks_with_metadata called with {len(haystack_chunks)} chunks and {len(original_blocks)} original blocks",
        file=sys.stderr,
    )
    print(
        f"DEBUG: min_chunk_size={min_chunk_size}, enable_dialogue_detection={enable_dialogue_detection}",
        file=sys.stderr,
    )

    # Debug: Check what pages are in the original blocks
    original_pages = {
        block.get("source", {}).get("page")
        for block in original_blocks
        if block.get("source", {}).get("page")
    }
    print(
        f"DEBUG: Original blocks contain pages: {sorted(original_pages)}",
        file=sys.stderr,
    )

    char_map = _build_char_map(original_blocks)

    def process_chunk(chunk, chunk_index):
        import sys

        print(f"DEBUG: process_chunk() ENTRY - chunk {chunk_index}", file=sys.stderr)

        final_text = _truncate_chunk(chunk.content.strip())
        if not final_text:
            print(
                f"DEBUG: process_chunk() EXIT - chunk {chunk_index} - EMPTY CONTENT",
                file=sys.stderr,
            )
            return None

        if not generate_metadata:
            print(
                f"DEBUG: process_chunk() EXIT - chunk {chunk_index} - NO METADATA MODE",
                file=sys.stderr,
            )
            return {"text": final_text}

        source_block = _find_source_block(chunk, char_map, original_blocks)
        if not source_block:
            print(
                f"DEBUG: process_chunk() EXIT - chunk {chunk_index} - NO SOURCE BLOCK FOUND",
                file=sys.stderr,
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
        print(
            f"DEBUG: process_chunk() EXIT - chunk {chunk_index} SUCCESS - result has {len(result.get('text', ''))} chars",
            file=sys.stderr,
        )
        return result

    if perform_ai_enrichment:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_chunk, chunk, i)
                for i, chunk in enumerate(haystack_chunks)
            ]
            processed_chunks = (future.result() for future in as_completed(futures))
    else:
        processed_chunks = (
            process_chunk(chunk, i) for i, chunk in enumerate(haystack_chunks)
        )

    return list(filter(None, processed_chunks))


def _build_char_map(blocks: list[dict]) -> dict:
    """Build a character position mapping for locating chunks in original blocks."""
    import sys

    if not blocks:
        print("DEBUG: _build_char_map called with empty blocks list", file=sys.stderr)
        return {"char_positions": ()}

    print(f"DEBUG: Building character map for {len(blocks)} blocks", file=sys.stderr)

    block_pages = {
        block.get("source", {}).get("page")
        for block in blocks
        if block.get("source", {}).get("page")
    }
    print(
        f"DEBUG: Character map includes pages: {sorted(block_pages)}",
        file=sys.stderr,
    )

    lengths = (len(block["text"]) + 2 for block in blocks[:-1])
    starts = list(accumulate(lengths, initial=0))

    for i, (block, start) in enumerate(zip(blocks, starts)):
        text_len = len(block["text"])
        page = block.get("source", {}).get("page", "unknown")
        print(
            f"DEBUG: Block {i} (page {page}): {text_len} chars at position {start}-{start + text_len}",
            file=sys.stderr,
        )

    char_positions = tuple(
        CharSpan(start, start + len(block["text"]), i)
        for i, (block, start) in enumerate(zip(blocks, starts))
    )

    print(
        f"DEBUG: Character map built with {len(char_positions)} entries",
        file=sys.stderr,
    )
    return {"char_positions": char_positions}


def _find_source_block(
    chunk: Document, _char_map: dict, original_blocks: list[dict]
) -> dict | None:
    """Find the original source block for a chunk using multiple heuristics."""
    import sys

    def ensure_source(block):
        if "source" not in block or not isinstance(block["source"], dict):
            block["source"] = {}
        block["source"].setdefault("filename", "unknown")
        if not isinstance(block["source"].get("page"), int):
            block["source"]["page"] = None
        block["source"].setdefault("location", None)
        return block

    if not (chunk and chunk.content and original_blocks):
        print(
            f"DEBUG: _find_source_block early return - chunk: {bool(chunk)}, content: {bool(chunk.content if chunk else False)}, blocks: {len(original_blocks) if original_blocks else 0}",
            file=sys.stderr,
        )
        return None

    chunk_text = chunk.content.strip()
    chunk_start = chunk_text[:50].replace("\n", " ").strip()
    print(
        f"DEBUG: Looking for source block for chunk starting with: '{chunk_start}...'",
        file=sys.stderr,
    )

    def substring_match(block: dict) -> bool:
        return chunk_start and chunk_start in block.get("text", "")

    def start_match(block: dict) -> bool:
        block_text = block.get("text", "").strip()
        if not block_text:
            return False
        block_start = block_text[: max(20, len(chunk_start))].replace("\n", " ").strip()
        lower_chunk = chunk_start.lower()
        lower_block = block_start.lower()
        return lower_chunk.startswith(lower_block) or lower_block.startswith(
            lower_chunk
        )

    def fuzzy_match(block: dict) -> bool:
        import re

        def normalize(s: str) -> str:
            return re.sub(r"[\W_]+", "", s).lower()

        block_start = block.get("text", "")[: max(20, len(chunk_start))]
        return normalize(block_start).startswith(normalize(chunk_start)[:15])

    def overlap_match(block: dict) -> bool:
        block_text = block.get("text", "")
        return any(chunk_start[:n] in block_text for n in range(30, 10, -5))

    def header_match(block: dict) -> bool:
        return (
            block is original_blocks[0]
            and chunk_start
            and (
                chunk_start.isupper()
                or chunk_start.startswith("CHAPTER")
                or chunk_start.startswith("SECTION")
            )
        )

    predicates = [
        ("substring match", substring_match),
        ("start match", start_match),
        ("fuzzy match", fuzzy_match),
        ("overlap match", overlap_match),
        ("header/special formatting", header_match),
    ]

    for label, predicate in predicates:
        match = next(
            (ensure_source(b) for b in original_blocks if predicate(b)),
            None,
        )
        if match:
            print(
                f"DEBUG: Found matching source block on page {match['source'].get('page', None)} ({label})",
                file=sys.stderr,
            )
            return match

    try:
        import difflib

        block_starts = [block.get("text", "")[:50] for block in original_blocks]
        matches = difflib.get_close_matches(chunk_start, block_starts, n=1, cutoff=0.6)
        if matches:
            idx = block_starts.index(matches[0])
            block = ensure_source(original_blocks[idx])
            print(
                f"DEBUG: Fallback: difflib matched block {idx} on page {block['source'].get('page', None)}",
                file=sys.stderr,
            )
            return block
    except Exception as e:
        print(f"DEBUG: difflib fallback failed: {e}", file=sys.stderr)

    block = ensure_source(original_blocks[0])
    print(
        f"WARNING: No matching source block found for chunk starting with: '{chunk_start}...'. Mapping to first block.",
        file=sys.stderr,
    )
    return block
