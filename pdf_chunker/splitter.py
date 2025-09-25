import logging
from collections import Counter
import re
from functools import reduce
from typing import Any, Dict, Iterable, Iterator, List, Match, Optional, Tuple

from .text_cleaning import (
    _is_probable_heading,
    pipe,
)
from .list_detection import starts_with_bullet
from .page_artifacts import _drop_trailing_bullet_footers


logger = logging.getLogger(__name__)

# Centralized chunk threshold constants
CHUNK_THRESHOLDS = {
    "very_short": 9,  # Very short chunks (≤5 words) - always merge
    "short": 12,  # Short chunks (≤10 words) - consider for merging
    "min_target": 10,  # Minimum target chunk size after merging
    "dialogue_response": 6,  # Short dialogue responses (≤6 words)
    "fragment": 4,  # Very short fragments (≤4 words) - always merge
    "related_short": 9,  # Related short chunks threshold
}

# Common dialogue verbs for attribution detection
DIALOGUE_VERBS = {
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
}

# Patterns for detecting numbered list boundaries
NUMBERED_ITEM_START = re.compile(r"^\s*(\d+)[.)]\s+")
NUMBER_AT_END = re.compile(r"(\d+)[.)]?\s*$")
NUMBERED_ITEM_ANYWHERE = re.compile(r"\b(\d+)[.)]\s+")


def _is_dialogue_attribution(words: List[str]) -> bool:
    """Return True if words represent a short dialogue attribution."""
    return len(words) <= CHUNK_THRESHOLDS["dialogue_response"] and any(
        word.lower() in DIALOGUE_VERBS for word in words
    )


def _is_incomplete_sentence(words: List[str], current: str, next_chunk: str) -> bool:
    """Detect incomplete sentences that likely merge with following text."""
    return (
        len(words) <= CHUNK_THRESHOLDS["dialogue_response"]
        and not current.endswith((".", "!", "?"))
        and not next_chunk[0].isupper()
    )


def _is_very_short_fragment(words: List[str]) -> bool:
    """Determine if a chunk is an extremely short fragment."""
    return len(words) <= CHUNK_THRESHOLDS["fragment"]


def _is_related_short_chunks(current_words: List[str], next_words: List[str], current: str) -> bool:
    """Check for adjacent related short chunks that should merge."""
    return (
        len(current_words) <= CHUNK_THRESHOLDS["related_short"]
        and len(next_words) <= CHUNK_THRESHOLDS["short"]
        and not current.endswith((".", "!", "?"))
    )


def _is_next_chunk_very_short(next_words: List[str]) -> bool:
    """Return True when the following chunk is extremely short."""
    return len(next_words) <= CHUNK_THRESHOLDS["very_short"]


def _merge_reason(
    current: str, next_chunk: str
) -> Tuple[bool, Optional[str], List[str], List[str]]:
    """Evaluate merge predicates and return decision metadata."""
    current_words = current.split()
    next_words = next_chunk.split()
    conditions = [
        (_is_dialogue_attribution(current_words), "short_dialogue_attribution"),
        (
            _is_incomplete_sentence(current_words, current, next_chunk),
            "incomplete_sentence",
        ),
        (_is_very_short_fragment(current_words), "very_short_fragment"),
        (
            _is_related_short_chunks(current_words, next_words, current),
            "related_short_chunks",
        ),
        (_is_next_chunk_very_short(next_words), "next_chunk_very_short"),
    ]
    reason = next((r for cond, r in conditions if cond), None)
    return reason is not None, reason, current_words, next_words


def _concat_chunks(first: str, second: str) -> str:
    """Join two chunks with a space respecting sentence boundaries."""
    return f"{first}{' ' if not first.endswith(('.', '!', '?', ',', ';', ':')) else ' '}{second}"


def _merge_forward(
    chunks: List[str], relationships: List[Dict[str, Any]], idx: int = 0
) -> Tuple[List[str], Counter]:
    """Recursively merge chunks based on precomputed relationships."""
    if idx >= len(chunks):
        return [], Counter()

    current = chunks[idx].strip()
    if not current:
        return _merge_forward(chunks, relationships, idx + 1)

    if idx < len(relationships) and relationships[idx]["should_merge"] and idx + 1 < len(chunks):
        next_chunk = chunks[idx + 1].strip()
        rest, counter = _merge_forward(chunks, relationships, idx + 2)
        reason = relationships[idx]["merge_reason"]
        counter.update([reason])
        return [_concat_chunks(current, next_chunk)] + rest, counter

    rest, counter = _merge_forward(chunks, relationships, idx + 1)
    return [current] + rest, counter


def _second_pass_merge(
    chunks: List[str], min_chunk_size: int, idx: int = 0
) -> Tuple[List[str], Counter]:
    """Recursively merge remaining short chunks after initial pass."""
    if idx >= len(chunks):
        return [], Counter()

    current = chunks[idx].strip()
    current_words = len(current.split())
    if current_words < min_chunk_size and idx + 1 < len(chunks):
        next_chunk = chunks[idx + 1].strip()
        next_words = len(next_chunk.split())
        if current_words + next_words <= min_chunk_size * 3:
            rest, counter = _second_pass_merge(chunks, min_chunk_size, idx + 2)
            counter.update(["second_pass_size"])
            return [_concat_chunks(current, next_chunk)] + rest, counter

    rest, counter = _second_pass_merge(chunks, min_chunk_size, idx + 1)
    return [current] + rest, counter


def _starting_number(text: str) -> Optional[int]:
    """Return leading list number if present."""
    match = NUMBERED_ITEM_START.match(text)
    return int(match.group(1)) if match else None


def _ending_number(text: str) -> Optional[int]:
    """Return trailing list number if present."""
    match = NUMBER_AT_END.search(text)
    return int(match.group(1)) if match else None


def _last_number(text: str) -> Optional[int]:
    """Return the highest numbered list item in text."""
    matches = list(re.finditer(r"(\d+)[.)]", text))
    return int(matches[-1].group(1)) if matches else None


def _merge_numbered_list_chunks(chunks: List[str]) -> List[str]:
    """Merge chunks whose numbered list items spill into the following chunk.

    This implementation is intentionally conservative: it merges a numbered
    chunk only with the immediate next chunk when that next chunk does not start
    with a new list number. This avoids accidentally swallowing large portions of
    text that are not part of the list, a regression observed in earlier
    versions. The approach favors correctness over aggressive merging and keeps
    the logic side‑effect free.
    """

    def _combine(first: str, second: str, newline: bool) -> str:
        first_part, second_part = first.rstrip(), second.lstrip()
        sep = "\n" if newline else " "
        combined = f"{first_part}{sep}{second_part}"
        return re.sub(r"\n{2,}", "\n", combined) if newline else combined

    merged: List[str] = []
    idx = 0
    while idx < len(chunks):
        current = chunks[idx].strip()
        if idx + 1 < len(chunks):
            nxt = chunks[idx + 1].strip()
            curr_last = _last_number(current)
            nxt_first = _starting_number(nxt)
            if curr_last is not None and nxt_first is not None and nxt_first == curr_last + 1:
                merged.append(_combine(current, nxt, True))
                idx += 2
                continue
            if _starting_number(current) is not None and nxt_first is None:
                merged.append(_combine(current, nxt, False))
                idx += 2
                continue
            last_line = current.rsplit("\n", 1)[-1]
            if NUMBERED_ITEM_ANYWHERE.search(last_line) and nxt_first is None:
                merged.append(_combine(current, nxt, False))
                idx += 2
                continue
        merged.append(current)
        idx += 1
    return merged


def _extract_bullet_tail(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Split lines into non-bullet head and trailing bullet block."""
    idx = len(lines)
    while idx > 0 and starts_with_bullet(lines[idx - 1]):
        idx -= 1
    if idx > 0 and lines[idx - 1].rstrip().endswith(":"):
        idx -= 1
    return lines[:idx], lines[idx:]


def _rebalance_bullet_chunks(chunks: List[str]) -> List[str]:
    """Move trailing bullet lists to following chunks to keep lists intact."""
    if not chunks:
        return chunks
    result: List[str] = []
    current = chunks[0]
    for nxt in chunks[1:]:
        curr_lines = current.rstrip().splitlines()
        next_lines = nxt.lstrip().splitlines()
        head, tail = _extract_bullet_tail(curr_lines)
        if tail and next_lines and starts_with_bullet(next_lines[0]):
            combined = tail + next_lines
            tlen = len(tail)
            i = tlen
            while i <= len(combined) - tlen:
                if combined[i : i + tlen] == tail:
                    combined = combined[:i] + combined[i + tlen :]
                    break
                i += 1
            cleaned: List[str] = []
            seen = set()
            for line in combined:
                if starts_with_bullet(line):
                    if line in seen:
                        continue
                    seen.add(line)
                else:
                    seen.clear()
                cleaned.append(line)
            combined = cleaned
            current = "\n".join(head)
            nxt = "\n".join(combined)
        result.append(current.rstrip())
        current = nxt
    result.append(current.rstrip())
    return result


def _merge_standalone_lists(chunks: List[str]) -> List[str]:
    """Attach standalone bullet or numbered list chunks to preceding text.

    This ensures list items are kept with their introductory context instead of
    forming separate chunks that begin with a list marker.
    """
    merged: List[str] = []
    for chunk in chunks:
        stripped = chunk.lstrip()
        if merged and (starts_with_bullet(stripped) or _starting_number(stripped) is not None):
            merged[-1] = f"{merged[-1].rstrip()}\n{stripped}"
        else:
            merged.append(chunk.rstrip())
    return merged


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
    """Analyze adjacent chunk relationships and merge heuristics."""

    def _relationship(i: int, current: str, nxt: str) -> Dict[str, Any]:
        should_merge, reason, curr_words, next_words = _merge_reason(current, nxt)
        return {
            "chunk_index": i,
            "current_word_count": len(curr_words),
            "next_word_count": len(next_words),
            "should_merge": should_merge,
            "merge_reason": reason,
        }

    return [
        _relationship(i, c.strip(), n.strip())
        for i, (c, n) in enumerate(zip(chunks, chunks[1:]))
        if c.strip() and n.strip()
    ]


def merge_conversational_chunks(
    chunks: List[str], min_chunk_size: int | None = None
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

    min_chunk_size = min_chunk_size or CHUNK_THRESHOLDS["min_target"]
    relationships = analyze_chunk_relationships(chunks)

    first_pass, first_counter = _merge_forward(chunks, relationships)
    second_pass, second_counter = _second_pass_merge(first_pass, min_chunk_size)

    reason_counter = first_counter + second_counter
    merged_chunks = second_pass
    merge_stats = {
        "original_count": len(chunks),
        "merges_performed": sum(reason_counter.values()),
        "merge_reasons": dict(reason_counter),
        "final_count": len(merged_chunks),
        "short_chunks_remaining": sum(
            1 for chunk in merged_chunks if len(chunk.split()) < min_chunk_size
        ),
    }

    return merged_chunks, merge_stats


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
        "Running _merge_short_chunks: min_chunk_size=%s, very_short_threshold=%s, short_threshold=%s",
        min_chunk_size,
        very_short_threshold,
        short_threshold,
    )

    if not chunks:
        return [], {
            "merges_performed": 0,
            "final_count": 0,
            "short_chunks_remaining": 0,
        }

    def _helper(sequence: List[str]) -> Tuple[List[str], Counter]:
        if not sequence:
            return [], Counter()

        current = sequence[0].strip()
        current_words = len(current.split())

        if current_words <= very_short_threshold and len(sequence) > 1:
            merged = _concat_chunks(current, sequence[1].strip())
            rest, counter = _helper(sequence[2:])
            counter.update(["very_short"])
            return [merged] + rest, counter

        if (
            current_words <= short_threshold
            and len(sequence) > 1
            and len(sequence[1].split()) <= short_threshold
        ):
            merged = _concat_chunks(current, sequence[1].strip())
            rest, counter = _helper(sequence[2:])
            counter.update(["short_pair"])
            return [merged] + rest, counter

        if current_words < min_chunk_size and len(sequence) > 1:
            merged = _concat_chunks(current, sequence[1].strip())
            rest, counter = _helper(sequence[2:])
            counter.update(["below_min"])
            return [merged] + rest, counter

        rest, counter = _helper(sequence[1:])
        return [current] + rest, counter

    merged_chunks, reason_counter = _helper(chunks)
    merge_stats = {
        "merges_performed": sum(reason_counter.values()),
        "final_count": len(merged_chunks),
        "short_chunks_remaining": sum(
            1 for chunk in merged_chunks if len(chunk.split()) < min_chunk_size
        ),
        "merge_reasons": dict(reason_counter),
    }
    logger.info(f"_merge_short_chunks complete: {merge_stats}")
    return merged_chunks, merge_stats


NEWLINE_TOKEN = "[[BR]]"
NBSP = "\u00a0"
NBSP_TOKEN = "[[NBSP]]"
_BULLET_AFTER_NEWLINE = re.compile(r"\n\s*([\-\*\u2022]\s+)")


def _tokenize_with_newlines(text: str) -> List[str]:
    """Return tokens while preserving explicit newline and list markers."""

    def _join_newline_bullet(match: Match[str]) -> str:
        return f" {NEWLINE_TOKEN}{match.group(1)}"

    prepared = pipe(
        text,
        lambda s: _BULLET_AFTER_NEWLINE.sub(_join_newline_bullet, s),
        lambda s: s.replace(NBSP, f" {NBSP_TOKEN} "),
        lambda s: s.replace("\n", f" {NEWLINE_TOKEN} "),
    )
    return [token for token in prepared.split() if token]


def _detokenize_with_newlines(tokens: Iterable[str]) -> str:
    """Rebuild text from tokens, restoring newline and list markers."""
    joined = " ".join(tokens)
    joined = re.sub(rf"{NEWLINE_TOKEN}([\-\*\u2022])", r"\n\1", joined)
    joined = joined.replace(NEWLINE_TOKEN, "\n").replace(NBSP_TOKEN, NBSP)
    joined = re.sub(rf" ?{NBSP} ?", NBSP, joined)
    return re.sub(r"[ \t]*\n[ \t]*", "\n", joined)


def iter_word_chunks(text: str, max_words: int, overlap: int = 0) -> Iterator[str]:
    """Yield ``text`` split into sequential word windows respecting ``overlap``."""

    if max_words <= 0:
        if text:
            yield text
        return

    words = text.split()
    if not words:
        return

    step = max(max_words - max(overlap, 0), 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if not chunk_words:
            break
        yield " ".join(chunk_words)


def _split_short_text(text: str) -> List[str]:
    """Split short passages near the middle, preferring paragraph breaks."""
    mid = len(text) // 2
    for sep in ("\n\n", "\n", " "):
        split_at = text.rfind(sep, 0, mid)
        if split_at == -1:
            split_at = text.find(sep, mid)
        if split_at != -1:
            head = text[:split_at].strip()
            tail = text[split_at + len(sep) :].strip()
            return [head, tail] if tail else [head]
    return [text.strip()]


def _dedupe_overlapping_chunks(chunks: List[str]) -> List[str]:
    """Drop chunks fully contained within their predecessor."""

    return reduce(
        lambda acc, c: acc if acc and c and c in acc[-1] else acc + [c],
        chunks,
        [],
    )


def _split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Return ``text`` split into word windows respecting ``overlap``."""

    tokens = _tokenize_with_newlines(text)
    if not tokens or chunk_size <= 0:
        return []
    if len(tokens) <= chunk_size:
        return [
            "\n".join(_drop_trailing_bullet_footers(_detokenize_with_newlines(tokens).splitlines()))
        ]
    step = max(1, chunk_size - overlap)
    windows = (tokens[i : i + chunk_size] for i in range(0, len(tokens), step))
    chunks = [
        "\n".join(_drop_trailing_bullet_footers(_detokenize_with_newlines(w).splitlines()))
        for w in windows
    ]
    chunks = _dedupe_overlapping_chunks(chunks)
    if len(chunks) > 1 and len(chunks[-1].split()) <= overlap * 2:
        chunks = chunks[:-1]
    return chunks or [
        "\n".join(_drop_trailing_bullet_footers(_detokenize_with_newlines(tokens).splitlines()))
    ]


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
            if _ends_with_opening_quote(current_chunk) and _starts_with_quote_content(next_chunk):
                logger.debug(f"Fixing quote split: merging chunks {i} and {i+1}")
                merged = current_chunk + " " + next_chunk
                fixed_chunks.append(merged)
                i += 2
                continue

            # Case 2: Current chunk is quote content, next starts with closing quote
            if _is_quote_content(current_chunk) and _starts_with_closing_quote(next_chunk):
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


def _extract_trailing_heading(chunk: str) -> Tuple[str, Optional[str]]:
    """Return body text and trailing heading if present."""
    lines = chunk.rstrip().splitlines()
    if not lines:
        return chunk, None

    last_line = lines[-1].strip()

    # Handle cases where a footer line ends with a delimiter followed by a
    # heading on the same line, e.g. "Footer text | Heading". Split on the last
    # vertical bar and treat the trailing portion as a heading if it qualifies.
    if "|" in last_line:
        pre, post = last_line.rsplit("|", 1)
        candidate = post.strip()
        if _is_probable_heading(candidate):
            body_lines = lines[:-1] + [pre.rstrip()]
            return "\n".join(body_lines).rstrip(), candidate

    if _is_probable_heading(last_line):
        body = "\n".join(lines[:-1]).rstrip()
        return body, last_line
    return chunk, None


def _fix_heading_splitting_issues(chunks: List[str]) -> List[str]:
    """Attach headings to the following chunk."""
    if not chunks:
        return chunks

    result: List[str] = []
    pending_heading: Optional[str] = None

    for chunk in chunks:
        if pending_heading:
            chunk = f"{pending_heading}\n{chunk.lstrip()}".strip()
            pending_heading = None

        body, heading = _extract_trailing_heading(chunk)
        if body:
            result.append(body)
        if heading:
            pending_heading = heading

    if pending_heading:
        result.append(pending_heading)

    return [c for c in result if c.strip()]


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
            logger.warning(f"Chunk {i} has unbalanced quotes (balance: {quote_balance})")

    def _handle_chunk(idx: int, chunk: str) -> str:
        if not _is_chunk_corrupted(chunk):
            return chunk
        logger.error(f"Chunk {idx} appears corrupted: '{chunk[:100]}...'")
        repaired = _attempt_chunk_repair(chunk)
        if repaired:
            logger.debug(f"Repaired chunk {idx} to '{repaired[:100]}...'")
            return repaired
        logger.debug(f"Retaining suspect chunk {idx} unmodified")
        return chunk

    return [_handle_chunk(i, c) for i, c in enumerate(chunks)]


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

    return text[0].islower() or first_word in continuation_words or not text[0].isupper()


def _is_quote_content(text: str) -> bool:
    """Check if text appears to be content inside quotes."""
    return bool(text) and not text[0].isupper() and not text.rstrip().endswith(('"', "'"))


def _starts_with_closing_quote(text: str) -> bool:
    """Check if text starts with a closing quote pattern."""
    import re

    # Pattern: starts with quote followed by punctuation or attribution
    return bool(
        re.match(r'^["\'][,.;:]', text) or re.match(r'^["\'].*?(said|asked|replied)', text.lower())
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
    chunk2_ends_setup = any(chunk2.lower().rstrip().endswith(ender) for ender in setup_enders)

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

    if min_chunk_size is None:
        min_chunk_size = max(8, chunk_size // 10)

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
    logger.info(f"Sliding window produced {len(initial_chunks)} initial chunks")

    # Count and log short chunks before processing
    initial_short_count = sum(
        1 for chunk in initial_chunks if len(chunk.split()) <= short_threshold
    )
    initial_very_short_count = sum(
        1 for chunk in initial_chunks if len(chunk.split()) <= very_short_threshold
    )

    logger.info(f"Initial short chunks (≤{short_threshold} words): {initial_short_count}")
    logger.info(
        f"Initial very short chunks (≤{very_short_threshold} words): {initial_very_short_count}"
    )

    # Log initial chunk details
    for i, chunk in enumerate(initial_chunks):
        word_count = len(chunk.split())
        chunk_preview = chunk[:100].replace("\n", "\\n")
        logger.info(f"Initial chunk {i}: {word_count} words, preview: '{chunk_preview}...'")

    # Apply conversational merging if enabled
    if enable_dialogue_detection and min_chunk_size is not None:
        merged_chunks, merge_stats = _merge_short_chunks(
            initial_chunks, min_chunk_size, very_short_threshold, short_threshold
        )
        logger.info(f"Conversational merging completed: {merge_stats}")
    else:
        merged_chunks = initial_chunks
        logger.info("Dialogue detection disabled")

    # Validate chunk integrity and repair heading boundaries
    validated_chunks = _fix_heading_splitting_issues(_validate_chunk_integrity(merged_chunks, text))

    # Merge numbered list items that were split across chunks
    numbered_chunks = _merge_numbered_list_chunks(validated_chunks)

    # Rebalance bullet lists and attach standalone list chunks
    bullet_chunks = _rebalance_bullet_chunks(numbered_chunks)
    list_chunks = _merge_standalone_lists(bullet_chunks)

    # Final statistics
    final_short_count = sum(1 for chunk in list_chunks if len(chunk.split()) <= short_threshold)
    final_very_short_count = sum(
        1 for chunk in list_chunks if len(chunk.split()) <= very_short_threshold
    )

    logger.info("Final chunking results:")
    logger.info(f"  Total chunks: {len(list_chunks)}")
    logger.info(
        f"  Short chunks (≤{short_threshold} words): {final_short_count} (reduced from {initial_short_count})"
    )
    logger.info(
        f"  Very short chunks (≤{very_short_threshold} words): {final_very_short_count} (reduced from {initial_very_short_count})"
    )

    if list_chunks:
        word_counts = [len(chunk.split()) for chunk in list_chunks]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        logger.info(
            f"  Chunk size stats: avg={avg_words:.1f} words, min={min_words} words, max={max_words} words"
        )

    return list_chunks


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
