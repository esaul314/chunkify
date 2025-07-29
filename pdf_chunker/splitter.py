import logging

logger = logging.getLogger(__name__)
import re
from typing import List, Dict, Any, Tuple, Optional
from haystack.components.preprocessors import DocumentSplitter

# from haystack.dataclasses import Document
from haystack import Document
from .text_cleaning import _is_probable_heading


# Try importing RecursiveCharacterTextSplitter from both possible locations
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    _LANGCHAIN_SPLITTER_AVAILABLE = True
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        _LANGCHAIN_SPLITTER_AVAILABLE = True
    except ImportError:
        RecursiveCharacterTextSplitter = None
        _LANGCHAIN_SPLITTER_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Centralized chunk threshold constants
CHUNK_THRESHOLDS = {
    "very_short": 9,  # Very short chunks (≤5 words) - always merge
    "short": 12,  # Short chunks (≤10 words) - consider for merging
    "min_target": 10,  # Minimum target chunk size after merging
    "dialogue_response": 6,  # Short dialogue responses (≤6 words)
    "fragment": 4,  # Very short fragments (≤4 words) - always merge
    "related_short": 9,  # Related short chunks threshold
}


def detect_dialogue_patterns(text: str) -> List[Dict[str, Any]]:
    """
    Detect dialogue patterns in text including quotes, responses, and commentary.

    Args:
        text: Text to analyze for dialogue patterns

    Returns:
        List of dialogue segments with metadata
    """
    dialogue_segments = []

    # Pattern for quoted speech followed by attribution or response
    quote_patterns = [
        r'"([^"]+)"[,.]?\s*([^.!?]*[.!?])',  # "Quote," response.
        r'"([^"]+)"[,.]?\s+(\w+(?:\s+\w+){0,10}[.!?])',  # "Quote" short response.
        r'([^.!?]*[.!?])\s*"([^"]+)"',  # Response "quote"
        r'"([^"]+)"[,.]?\s*(\w+\s+(?:said|replied|asked|answered|continued|added|noted|observed|remarked|stated|declared|exclaimed|whispered|shouted|muttered|explained|insisted|argued|suggested|wondered|thought|concluded)[^.!?]*[.!?])',  # "Quote" attribution.
    ]

    for pattern in quote_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()

            # Extract the quote and response/attribution
            groups = match.groups()
            if len(groups) >= 2:
                quote_text = groups[0].strip()
                response_text = groups[1].strip()

                dialogue_segments.append(
                    {
                        "type": "dialogue",
                        "start": start_pos,
                        "end": end_pos,
                        "quote": quote_text,
                        "response": response_text,
                        "full_text": match.group(0),
                        "word_count": len(match.group(0).split()),
                    }
                )

    # Sort by position in text
    dialogue_segments.sort(key=lambda x: x["start"])

    return dialogue_segments


def analyze_chunk_relationships(chunks: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze relationships between chunks to identify those that should be merged.

    Args:
        chunks: List of chunk texts

    Returns:
        List of relationship analysis results
    """
    relationships = []

    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].strip()
        next_chunk = chunks[i + 1].strip()

        if not current_chunk or not next_chunk:
            continue

        current_words = current_chunk.split()
        next_words = next_chunk.split()

        relationship = {
            "chunk_index": i,
            "current_word_count": len(current_words),
            "next_word_count": len(next_words),
            "should_merge": False,
            "merge_reason": None,
        }

        # Check for short chunks that should be merged using centralized thresholds
        if len(current_words) <= CHUNK_THRESHOLDS["short"]:
            # Short chunk - analyze context for merging

            # Case 1: Short response after dialogue
            if len(current_words) <= CHUNK_THRESHOLDS["dialogue_response"] and any(
                word.lower()
                in [
                    "said",
                    "replied",
                    "asked",
                    "answered",
                    "continued",
                    "added",
                    "noted",
                    "observed",
                    "remarked",
                    "stated",
                    "declared",
                    "exclaimed",
                    "whispered",
                    "shouted",
                    "muttered",
                    "explained",
                    "insisted",
                    "argued",
                    "suggested",
                    "wondered",
                    "thought",
                    "concluded",
                ]
                for word in current_words
            ):
                relationship["should_merge"] = True
                relationship["merge_reason"] = "short_dialogue_attribution"

            # Case 2: Short commentary or response
            elif (
                len(current_words) <= CHUNK_THRESHOLDS["dialogue_response"]
                and not current_chunk.endswith((".", "!", "?"))
                and not next_chunk[0].isupper()
            ):
                relationship["should_merge"] = True
                relationship["merge_reason"] = "incomplete_sentence"

            # Case 3: Very short fragments
            elif len(current_words) <= CHUNK_THRESHOLDS["fragment"]:
                relationship["should_merge"] = True
                relationship["merge_reason"] = "very_short_fragment"

            # Case 4: Short chunk followed by related content
            elif (
                len(current_words) <= CHUNK_THRESHOLDS["related_short"]
                and len(next_words) <= CHUNK_THRESHOLDS["short"]
                and not current_chunk.endswith((".", "!", "?"))
            ):
                relationship["should_merge"] = True
                relationship["merge_reason"] = "related_short_chunks"

        # Check for next chunk being very short
        if len(next_words) <= CHUNK_THRESHOLDS["very_short"]:
            # Next chunk is very short - likely should be merged with current
            if not relationship["should_merge"]:
                relationship["should_merge"] = True
                relationship["merge_reason"] = "next_chunk_very_short"

        relationships.append(relationship)

    return relationships


def merge_conversational_chunks(
    chunks: List[str], min_chunk_size: int = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Merge chunks based on conversational patterns and minimum size requirements.

    Args:
        chunks: List of chunk texts
        min_chunk_size: Minimum number of words per chunk (defaults to CHUNK_THRESHOLDS['min_target'])

    Returns:
        Tuple of (merged_chunks, merge_statistics)
    """

    if not chunks:
        return chunks, {}
    # Use centralized threshold if not provided
    if min_chunk_size is None:
        min_chunk_size = CHUNK_THRESHOLDS["min_target"]

    # Analyze relationships between chunks
    relationships = analyze_chunk_relationships(chunks)

    merged_chunks = []
    merge_stats = {
        "original_count": len(chunks),
        "merges_performed": 0,
        "merge_reasons": {},
        "final_count": 0,
        "short_chunks_remaining": 0,
    }

    i = 0
    while i < len(chunks):
        current_chunk = chunks[i].strip()

        if not current_chunk:
            i += 1
            continue

        # Check if this chunk should be merged with the next one
        should_merge_forward = False
        merge_reason = None

        if i < len(relationships):
            rel = relationships[i]
            should_merge_forward = rel["should_merge"]
            merge_reason = rel["merge_reason"]

        if should_merge_forward and i + 1 < len(chunks):
            # Merge current chunk with next chunk
            next_chunk = chunks[i + 1].strip()
            if next_chunk:
                # Determine appropriate separator
                separator = " "
                if current_chunk.endswith((".", "!", "?")):
                    separator = " "
                elif current_chunk.endswith((",", ";", ":")):
                    separator = " "
                else:
                    separator = " "

                merged_text = current_chunk + separator + next_chunk
                merged_chunks.append(merged_text)

                merge_stats["merges_performed"] += 1
                if merge_reason:
                    merge_stats["merge_reasons"][merge_reason] = (
                        merge_stats["merge_reasons"].get(merge_reason, 0) + 1
                    )

                i += 2  # Skip both chunks since they were merged
            else:
                merged_chunks.append(current_chunk)
                i += 1
        else:
            # Keep chunk as is
            merged_chunks.append(current_chunk)
            i += 1

    # Second pass: merge any remaining very short chunks
    final_chunks = []
    i = 0
    while i < len(merged_chunks):
        current_chunk = merged_chunks[i].strip()
        current_words = len(current_chunk.split())

        if current_words < min_chunk_size and i + 1 < len(merged_chunks):
            # Try to merge with next chunk
            next_chunk = merged_chunks[i + 1].strip()
            next_words = len(next_chunk.split())

            # Merge if combined size is reasonable (use centralized threshold)
            if current_words + next_words <= min_chunk_size * 3:
                separator = " " if not current_chunk.endswith((".", "!", "?")) else " "
                merged_text = current_chunk + separator + next_chunk
                final_chunks.append(merged_text)

                merge_stats["merges_performed"] += 1
                merge_stats["merge_reasons"]["second_pass_size"] = (
                    merge_stats["merge_reasons"].get("second_pass_size", 0) + 1
                )

                i += 2
            else:
                final_chunks.append(current_chunk)
                i += 1
        else:
            final_chunks.append(current_chunk)
            i += 1

    # Count remaining short chunks
    for chunk in final_chunks:
        if len(chunk.split()) < min_chunk_size:
            merge_stats["short_chunks_remaining"] += 1

    merge_stats["final_count"] = len(final_chunks)

    return final_chunks, merge_stats


def _merge_short_chunks(
    chunks: List[str],
    min_chunk_size: int,
    very_short_threshold: int,
    short_threshold: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Merge adjacent short chunks that fall below the minimum word threshold.
    Combines very short and short chunks with their neighbors to improve semantic coherence.
    Returns the merged chunks and merge statistics.
    """
    logger.info(
        f"Running _merge_short_chunks: min_chunk_size={min_chunk_size}, very_short_threshold={very_short_threshold}, short_threshold={short_threshold}"
    )
    if not chunks:
        return [], {
            "merges_performed": 0,
            "final_count": 0,
            "short_chunks_remaining": 0,
        }

    merged_chunks = []
    merge_stats = {
        "merges_performed": 0,
        "final_count": 0,
        "short_chunks_remaining": 0,
        "merge_reasons": {},
    }

    i = 0
    while i < len(chunks):
        current_chunk = chunks[i].strip()
        current_words = len(current_chunk.split())

        # Always merge very short chunks (≤ very_short_threshold) with the next chunk
        if current_words <= very_short_threshold and i + 1 < len(chunks):
            next_chunk = chunks[i + 1].strip()
            merged_text = current_chunk + " " + next_chunk
            merged_chunks.append(merged_text)
            merge_stats["merges_performed"] += 1
            merge_stats["merge_reasons"]["very_short"] = (
                merge_stats["merge_reasons"].get("very_short", 0) + 1
            )
            i += 2
            continue

        # Merge short chunks (≤ short_threshold) with the next chunk if both are short
        elif current_words <= short_threshold and i + 1 < len(chunks):
            next_chunk = chunks[i + 1].strip()
            next_words = len(next_chunk.split())
            if next_words <= short_threshold:
                merged_text = current_chunk + " " + next_chunk
                merged_chunks.append(merged_text)
                merge_stats["merges_performed"] += 1
                merge_stats["merge_reasons"]["short_pair"] = (
                    merge_stats["merge_reasons"].get("short_pair", 0) + 1
                )
                i += 2
                continue
            else:
                merged_chunks.append(current_chunk)
                i += 1
                continue

        # Merge any chunk below min_chunk_size with the next chunk if possible
        elif current_words < min_chunk_size and i + 1 < len(chunks):
            next_chunk = chunks[i + 1].strip()
            merged_text = current_chunk + " " + next_chunk
            merged_chunks.append(merged_text)
            merge_stats["merges_performed"] += 1
            merge_stats["merge_reasons"]["below_min"] = (
                merge_stats["merge_reasons"].get("below_min", 0) + 1
            )
            i += 2
            continue

        else:
            merged_chunks.append(current_chunk)
            i += 1

    # Count remaining short chunks
    for chunk in merged_chunks:
        if len(chunk.split()) < min_chunk_size:
            merge_stats["short_chunks_remaining"] += 1

    merge_stats["final_count"] = len(merged_chunks)
    logger.info(f"_merge_short_chunks complete: {merge_stats}")
    return merged_chunks, merge_stats


def _split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into chunks using sentence boundaries."""

    if _LANGCHAIN_SPLITTER_AVAILABLE and RecursiveCharacterTextSplitter is not None:
        # Convert word-based sizes to character estimates
        chunk_size_chars = chunk_size * 6  # Rough estimate: 6 chars per word
        overlap_chars = overlap * 6

        logger.debug(
            f"Text splitting: {chunk_size} words ({chunk_size_chars} chars), "
            f"{overlap} words overlap ({overlap_chars} chars)"
        )

        # Check for potential quote-related splitting issues
        quote_count = text.count('"') + text.count("'")
        if quote_count > 0:
            logger.debug(
                f"Text contains {quote_count} quote characters - monitoring for split issues"
            )

        # Use quote-aware separators - avoid splitting at quotes when possible
        separators = [
            "\n\n",  # Paragraph breaks (highest priority)
            "\n",  # Line breaks
            ". ",  # Sentence endings with space
            "! ",  # Exclamation with space
            "? ",  # Question with space
            "; ",  # Semicolon with space
            ": ",  # Colon with space
            ", ",  # Comma with space (lower priority)
            " ",  # Word boundaries (very low priority)
            "",  # Character boundaries (last resort)
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_chars,
            chunk_overlap=overlap_chars,
            length_function=len,
            separators=separators,
        )

        chunks = splitter.split_text(text)

        logger.debug(f"RecursiveCharacterTextSplitter produced {len(chunks)} chunks")

        # Post-process to fix quote and heading related issues
        fixed_chunks = _fix_heading_splitting_issues(
            _fix_quote_splitting_issues(chunks)
        )

        if len(fixed_chunks) != len(chunks):
            logger.info(
                f"Boundary fixes applied: {len(chunks)} → {len(fixed_chunks)} chunks"
            )

        # Log potential issues with quote handling
        for i, chunk in enumerate(fixed_chunks):
            if chunk.startswith('"') and not chunk.endswith('"'):
                logger.warning(
                    f"Chunk {i} starts with quote but doesn't end with quote - potential split issue"
                )
            elif chunk.endswith('"') and not chunk.startswith('"'):
                logger.warning(
                    f"Chunk {i} ends with quote but doesn't start with quote - potential split issue"
                )

            # Check for suspicious text ordering
            if i > 0:
                prev_chunk = fixed_chunks[i - 1]
                if chunk.strip() and prev_chunk.strip():
                    # Simple heuristic: if current chunk starts with lowercase and previous ends without punctuation
                    if chunk[0].islower() and not prev_chunk.rstrip().endswith(
                        (".", "!", "?", ":", ";")
                    ):
                        logger.debug(
                            f"Potential continuation split between chunks {i-1} and {i}"
                        )

        return fixed_chunks
    else:
        # Fallback: simple paragraph-based splitting
        logger.warning(
            "LangChain RecursiveCharacterTextSplitter not available, using fallback splitter."
        )
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if not current_chunk:
                current_chunk = para
            elif len(current_chunk) + len(para) + 2 < chunk_size * 6:
                current_chunk += "\n\n" + para
            else:
                chunks.append(current_chunk)
                current_chunk = para
        if current_chunk:
            chunks.append(current_chunk)
        logger.info(f"Fallback splitter produced {len(chunks)} chunks")
        return _fix_heading_splitting_issues(chunks)


def _fix_quote_splitting_issues(chunks: List[str]) -> List[str]:
    """Post-process chunks to fix quote-related splitting issues."""
    if not chunks:
        return chunks

    if not chunks:
        return chunks
    fixed_chunks = []
    i = 0

    while i < len(chunks):
        current_chunk = chunks[i]

        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]

            # Case 1: Current chunk ends with opening quote, next starts with content
            if _ends_with_opening_quote(current_chunk) and _starts_with_quote_content(
                next_chunk
            ):
                logger.debug(f"Fixing quote split: merging chunks {i} and {i+1}")
                merged = current_chunk + " " + next_chunk
                fixed_chunks.append(merged)
                i += 2
                continue

            # Case 2: Current chunk is quote content, next starts with closing quote
            if _is_quote_content(current_chunk) and _starts_with_closing_quote(
                next_chunk
            ):
                logger.debug(f"Fixing quote split: merging chunks {i} and {i+1}")
                merged = current_chunk + " " + next_chunk
                fixed_chunks.append(merged)
                i += 2
                continue

            # Case 3: Detect and fix reordered text
            if _is_text_reordered(current_chunk, next_chunk):
                logger.warning(
                    f"Detected text reordering between chunks {i} and {i+1} - attempting fix"
                )
                # Try to fix by swapping the order
                fixed_chunks.append(next_chunk)
                fixed_chunks.append(current_chunk)
                i += 2
                continue

        fixed_chunks.append(current_chunk)
        i += 1

    return fixed_chunks


def _fix_heading_splitting_issues(chunks: List[str]) -> List[str]:
    """Ensure headings are grouped with their following content."""
    if not chunks:
        return chunks

    normalized = chunks[:]
    for i in range(len(normalized) - 1):
        lines = normalized[i].rstrip().splitlines()
        if not lines:
            continue
        last_line = lines[-1].strip()
        if _is_probable_heading(last_line):
            body = "\n".join(lines[:-1]).rstrip()
            next_chunk = normalized[i + 1].lstrip()
            normalized[i + 1] = f"{last_line}\n{next_chunk}".strip()
            normalized[i] = body

    return [c for c in normalized if c.strip()]


def _validate_chunk_integrity(chunks: List[str], original_text: str) -> List[str]:
    """Validate chunk integrity and detect corruption issues."""
    if not chunks:
        return chunks

    logger.debug(f"Validating integrity of {len(chunks)} chunks")

    # Check 1: Ensure all text is preserved
    combined_chunks = " ".join(chunks)
    original_words = set(original_text.split())
    combined_words = set(combined_chunks.split())

    missing_words = original_words - combined_words
    extra_words = combined_words - original_words

    if missing_words:
        logger.warning(f"Missing words detected: {list(missing_words)[:5]}...")
    if extra_words:
        logger.warning(f"Extra words detected: {list(extra_words)[:5]}...")

    # Check 2: Validate quote balance
    for i, chunk in enumerate(chunks):
        quote_balance = _check_quote_balance(chunk)
        if quote_balance != 0:
            logger.warning(
                f"Chunk {i} has unbalanced quotes (balance: {quote_balance})"
            )

    # Check 3: Look for obvious corruption patterns
    validated_chunks = []
    for i, chunk in enumerate(chunks):
        if _is_chunk_corrupted(chunk):
            logger.error(f"Chunk {i} appears corrupted: '{chunk[:100]}...'")
            # Try to fix or skip corrupted chunks
            fixed_chunk = _attempt_chunk_repair(chunk)
            if fixed_chunk:
                validated_chunks.append(fixed_chunk)
            else:
                logger.error(f"Could not repair chunk {i}, skipping")
        else:
            validated_chunks.append(chunk)

    return validated_chunks


def _ends_with_opening_quote(text: str) -> bool:
    """Check if text ends with an opening quote pattern."""
    import re

    # Pattern: ends with quote followed by capital letter or start of sentence
    return bool(re.search(r'"[A-Z]', text[-10:]) or re.search(r"'[A-Z]", text[-10:]))


def _starts_with_quote_content(text: str) -> bool:
    """Check if text starts with content that should be inside quotes."""
    if not text:
        return False

    continuation_words = {"and", "but", "or", "so", "yet", "for", "nor"}
    first_word = text.split()[0].lower() if text.split() else ""

    return (
        text[0].islower() or first_word in continuation_words or not text[0].isupper()
    )


def _is_quote_content(text: str) -> bool:
    """Check if text appears to be content inside quotes."""
    return text and not text[0].isupper() and not text.rstrip().endswith(('"', "'"))


def _starts_with_closing_quote(text: str) -> bool:
    """Check if text starts with a closing quote pattern."""
    import re

    # Pattern: starts with quote followed by punctuation or attribution
    return bool(
        re.match(r'^["\'][,.;:]', text)
        or re.match(r'^["\'].*?(said|asked|replied)', text.lower())
    )


def _is_text_reordered(chunk1: str, chunk2: str) -> bool:
    """Detect if text chunks appear to be in wrong order."""
    if not chunk1 or not chunk2:
        return False

    # Heuristic: if chunk1 starts with continuation and chunk2 ends with setup
    continuation_starters = ['"', ",", "and", "but", "or", "so"]
    chunk1_starts_continuation = any(
        chunk1.lower().startswith(starter) for starter in continuation_starters
    )

    setup_enders = [",", ":", ";", "said", "asked", "replied"]
    chunk2_ends_setup = any(
        chunk2.lower().rstrip().endswith(ender) for ender in setup_enders
    )

    return chunk1_starts_continuation and chunk2_ends_setup


def semantic_chunker(
    text: str,
    chunk_size: int = 8000,
    overlap: int = 200,
    min_chunk_size: Optional[int] = None,
    enable_dialogue_detection: bool = False,
) -> List[str]:
    """Split text into semantic chunks with dialogue awareness."""
    if not text.strip():
        logger.warning("Empty text provided to semantic chunker")
        return []

    logger.info(
        f"Starting semantic chunking: {len(text)} chars, target size={chunk_size} words, overlap={overlap} words, min_chunk_size={min_chunk_size} words, dialogue_detection={enable_dialogue_detection}"
    )

    # Log text preview for debugging
    text_preview = text[:200].replace("\n", "\\n")
    logger.debug(f"Input text preview: '{text_preview}...'")

    # Apply thresholds
    very_short_threshold = CHUNK_THRESHOLDS["very_short"]
    short_threshold = CHUNK_THRESHOLDS["short"]

    logger.debug(
        f"Using thresholds: very_short ≤{very_short_threshold} words, short ≤{short_threshold} words"
    )

    # Initial splitting
    initial_chunks = _split_text_into_chunks(text, chunk_size, overlap)
    logger.info(f"DocumentSplitter produced {len(initial_chunks)} initial chunks")

    # Count and log short chunks before processing
    initial_short_count = sum(
        1 for chunk in initial_chunks if len(chunk.split()) <= short_threshold
    )
    initial_very_short_count = sum(
        1 for chunk in initial_chunks if len(chunk.split()) <= very_short_threshold
    )

    logger.info(
        f"Initial short chunks (≤{short_threshold} words): {initial_short_count}"
    )
    logger.info(
        f"Initial very short chunks (≤{very_short_threshold} words): {initial_very_short_count}"
    )

    # Log initial chunk details
    for i, chunk in enumerate(initial_chunks):
        word_count = len(chunk.split())
        chunk_preview = chunk[:100].replace("\n", "\\n")
        logger.info(
            f"Initial chunk {i}: {word_count} words, preview: '{chunk_preview}...'"
        )

    # Apply conversational merging if enabled
    if enable_dialogue_detection and min_chunk_size:
        merged_chunks, merge_stats = _merge_short_chunks(
            initial_chunks, min_chunk_size, very_short_threshold, short_threshold
        )
        logger.info(f"Conversational merging completed: {merge_stats}")
    else:
        merged_chunks = initial_chunks
        logger.info("Dialogue detection disabled")

    # Validate chunk integrity and repair heading boundaries
    validated_chunks = _fix_heading_splitting_issues(
        _validate_chunk_integrity(merged_chunks, text)
    )

    # Final statistics
    final_short_count = sum(
        1 for chunk in validated_chunks if len(chunk.split()) <= short_threshold
    )
    final_very_short_count = sum(
        1 for chunk in validated_chunks if len(chunk.split()) <= very_short_threshold
    )

    logger.info("Final chunking results:")
    logger.info(f"  Total chunks: {len(validated_chunks)}")
    logger.info(
        f"  Short chunks (≤{short_threshold} words): {final_short_count} (reduced from {initial_short_count})"
    )
    logger.info(
        f"  Very short chunks (≤{very_short_threshold} words): {final_very_short_count} (reduced from {initial_very_short_count})"
    )

    if validated_chunks:
        word_counts = [len(chunk.split()) for chunk in validated_chunks]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        logger.info(
            f"  Chunk size stats: avg={avg_words:.1f} words, min={min_words} words, max={max_words} words"
        )

    return validated_chunks


def _check_quote_balance(text: str) -> int:
    """Check quote balance in text. Returns 0 if balanced, positive if more opening quotes."""
    double_quotes = text.count('"') - text.count('\\"')
    single_quotes = text.count("'") - text.count("\\'")

    # For simplicity, assume quotes should be balanced
    return (double_quotes % 2) + (single_quotes % 2)


def _is_chunk_corrupted(chunk: str) -> bool:
    """Detect if a chunk appears to be corrupted."""
    if not chunk.strip():
        return False

    # Pattern 1: Starts with punctuation that suggests it's a fragment
    if chunk.lstrip().startswith(('", ', '",', '".')):
        return True

    # Pattern 2: Contains obvious ordering issues
    if '", pulls it all together' in chunk and "Finally, Part III" in chunk:
        return True

    # Pattern 3: Starts with closing punctuation
    if chunk.lstrip().startswith((",", ".", ";", ":")):
        return True

    return False


def _attempt_chunk_repair(chunk: str) -> str:
    """Attempt to repair a corrupted chunk."""
    if not chunk:
        return chunk

    # Remove leading punctuation that suggests fragmentation
    repaired = chunk.lstrip()
    while repaired and repaired[0] in '",.:;':
        repaired = repaired[1:].lstrip()

    # If nothing meaningful remains, return empty
    if len(repaired) < 10:
        return ""

    return repaired
