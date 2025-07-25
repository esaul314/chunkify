import re
import uuid
import textstat
from haystack.dataclasses import Document
from .ai_enrichment import classify_chunk_utterance
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    return {
        "flesch_kincaid_grade": flesch_kincaid,
        "difficulty": difficulty
    }
    
def _generate_chunk_id(filename: str, page: int, chunk_index: int) -> str:
    """Generates a unique chunk ID using underscores as separators."""
    # Ensure filename does not contain underscores that would break the pattern
    # (but preserve the extension)
    # If page is None or 0, use 0
    page_part = page if page is not None else 0
    return f"{filename}_p{page_part}_c{chunk_index}"
    
def format_chunks_with_metadata(
    haystack_chunks: list[Document],
    original_blocks: list[dict],
    generate_metadata: bool = True,
    perform_ai_enrichment: bool = False,
    max_workers: int = 10,
    min_chunk_size: int = None,
    enable_dialogue_detection: bool = True
) -> list[dict]:
    """
    Formats final chunks, enriching them in parallel with detailed metadata.
    Follows the canonical schema from README.ai with all required fields.
    """
    import sys
    import os

    print(f"DEBUG: format_chunks_with_metadata called with {len(haystack_chunks)} chunks and {len(original_blocks)} original blocks", file=sys.stderr)
    print(f"DEBUG: min_chunk_size={min_chunk_size}, enable_dialogue_detection={enable_dialogue_detection}", file=sys.stderr)
    
    # Debug: Check what pages are in the original blocks
    original_pages = set()
    for block in original_blocks:
        page = block.get('source', {}).get('page')
        if page:
            original_pages.add(page)
    print(f"DEBUG: Original blocks contain pages: {sorted(original_pages)}", file=sys.stderr)
    
    char_map = _build_char_map(original_blocks)
    
    def process_chunk(chunk, chunk_index):
        import sys
        
        print(f"DEBUG: process_chunk() ENTRY - chunk {chunk_index}", file=sys.stderr)
        
        final_text = chunk.content.strip()
        if not final_text:
            print(f"DEBUG: process_chunk() EXIT - chunk {chunk_index} - EMPTY CONTENT", file=sys.stderr)
            return None

        print(f"DEBUG: process_chunk() - chunk {chunk_index} has {len(final_text)} characters", file=sys.stderr)
        print(f"DEBUG: process_chunk() - chunk {chunk_index} preview: '{final_text[:50]}...'", file=sys.stderr)

        # Strict chunk size validation before processing
        max_chunk_size = 8000  # Strict 8k character limit
        if len(final_text) > max_chunk_size:
            print(f"WARNING: Chunk {chunk_index} is oversized ({len(final_text)} chars), applying strict truncation", file=sys.stderr)

            # Try to truncate at sentence boundary first
            truncate_point = max_chunk_size - 100  # Leave buffer for clean ending

            # Look for sentence endings within the truncation zone
            sentence_endings = ['. ', '.\n', '! ', '!\n', '? ', '?\n']
            best_break = -1

            for ending in sentence_endings:
                last_occurrence = final_text.rfind(ending, 0, truncate_point)
                if last_occurrence > truncate_point * 0.7:  # Don't truncate too early
                    best_break = max(best_break, last_occurrence + len(ending))

            if best_break > 0:
                final_text = final_text[:best_break].strip()
            else:
                # Look for paragraph breaks
                last_paragraph = final_text.rfind('\n\n', 0, truncate_point)
                if last_paragraph > truncate_point * 0.7:
                    final_text = final_text[:last_paragraph].strip()
                else:
                    # Last resort: word boundary
                    last_space = final_text.rfind(' ', 0, truncate_point)
                    if last_space > truncate_point * 0.8:
                        final_text = final_text[:last_space].strip()
                    else:
                        # Emergency character truncation
                        final_text = final_text[:truncate_point].strip()

            print(f"DEBUG: process_chunk() - chunk {chunk_index} truncated to {len(final_text)} characters", file=sys.stderr)

        print(f"DEBUG: process_chunk() - chunk {chunk_index} checking metadata generation flag", file=sys.stderr)

        if not generate_metadata:
            print(f"DEBUG: process_chunk() EXIT - chunk {chunk_index} - NO METADATA MODE", file=sys.stderr)
            return {"text": final_text}

        print(f"DEBUG: process_chunk() - chunk {chunk_index} finding source block", file=sys.stderr)
        source_block = _find_source_block(chunk, char_map, original_blocks)
        if not source_block:
            print(f"DEBUG: process_chunk() EXIT - chunk {chunk_index} - NO SOURCE BLOCK FOUND", file=sys.stderr)
            return None  # Or handle as an error

        filename = source_block.get("source", {}).get("filename", "Unknown")
        page = source_block.get("source", {}).get("page", 0)
        location = source_block.get("source", {}).get("location")  # For EPUBs, null for PDFs
        
        print(f"DEBUG: process_chunk() - chunk {chunk_index} mapped to page {page}, filename {filename}", file=sys.stderr)

        # AI classification is only done if the flag is set
        print(f"DEBUG: process_chunk() - chunk {chunk_index} checking AI enrichment flag: {perform_ai_enrichment}", file=sys.stderr)
        
        if perform_ai_enrichment:
            print(f"DEBUG: process_chunk() - chunk {chunk_index} CALLING AI ENRICHMENT", file=sys.stderr)
            try:
                utterance_result = classify_chunk_utterance(final_text)
                utterance_type = {
                    "classification": utterance_result.get("classification", "unclassified"),
                    "tags": utterance_result.get("tags", [])
                }
                print(f"DEBUG: process_chunk() - chunk {chunk_index} AI enrichment SUCCESS: {utterance_type}", file=sys.stderr)
            except Exception as e:
                print(f"DEBUG: process_chunk() - chunk {chunk_index} AI enrichment FAILED: {e}", file=sys.stderr)
                utterance_type = {
                    "classification": "error",
                    "tags": []
                }
        else:
            print(f"DEBUG: process_chunk() - chunk {chunk_index} AI enrichment DISABLED", file=sys.stderr)
            utterance_type = {
                "classification": "disabled",
                "tags": []
            }

        print(f"DEBUG: process_chunk() - chunk {chunk_index} building metadata", file=sys.stderr)

        # Determine location: null for PDFs, internal file path for EPUBs
        # If 'location' is missing, set to None
        if location is None and filename.lower().endswith('.pdf'):
            location_value = None
        else:
            location_value = location if location is not None else None

        # Ensure page is an integer or None
        page_value = page if isinstance(page, int) and page > 0 else None

        metadata = {
            "source": filename,
            "chunk_id": _generate_chunk_id(filename, page_value if page_value is not None else 0, chunk_index),
            "page": page_value,
            "location": location_value,  # null for PDFs, internal file path for EPUBs
            "block_type": source_block.get("type", "paragraph"),  # "heading" or "paragraph"
            "language": source_block.get("language", "un"),  # ISO language code, "un" for unknown
            "readability": _compute_readability(final_text),  # Object with flesch_kincaid_grade and difficulty
            "utterance_type": utterance_type,  # Object with classification and tags
            "importance": "medium"  # Currently defaults to "medium"
        }

        print(f"DEBUG: process_chunk() - chunk {chunk_index} building final result", file=sys.stderr)

        result = {
            "text": final_text,
            "metadata": {k: v for k, v in metadata.items() if k != "location" or v is None or v}  # Always include location, even if None
        }

        print(f"DEBUG: process_chunk() EXIT - chunk {chunk_index} SUCCESS - result has {len(result.get('text', ''))} chars", file=sys.stderr)
        return result

    # We only need parallel processing if AI enrichment is on
    if perform_ai_enrichment:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk, i): chunk
                for i, chunk in enumerate(haystack_chunks)
            }
            processed_chunks = [future.result() for future in as_completed(future_to_chunk)]
    else:
        # If no AI, process sequentially as it's very fast
        processed_chunks = [process_chunk(chunk, i) for i, chunk in enumerate(haystack_chunks)]

    return [chunk for chunk in processed_chunks if chunk]

def _build_char_map(blocks: list[dict]) -> dict:
    """
    Builds a character position mapping for locating chunks in original blocks.
    """
    import sys

    if not blocks:
        print(f"DEBUG: _build_char_map called with empty blocks list", file=sys.stderr)
        return {"char_positions": []}

    print(f"DEBUG: Building character map for {len(blocks)} blocks", file=sys.stderr)

    # Debug: Check what pages are in the blocks being mapped
    block_pages = set()
    for block in blocks:
        page = block.get('source', {}).get('page')
        if page:
            block_pages.add(page)
    print(f"DEBUG: Character map includes pages: {sorted(block_pages)}", file=sys.stderr)

    char_map = []
    current_pos = 0

    for i, block in enumerate(blocks):
        text_len = len(block["text"])
        page = block.get('source', {}).get('page', 'unknown')
        print(f"DEBUG: Block {i} (page {page}): {text_len} chars at position {current_pos}-{current_pos + text_len}", file=sys.stderr)
        
        char_entry = {
            "start": current_pos,
            "end": current_pos + text_len,
            "original_index": i
        }
        char_map.append(char_entry)
        current_pos += text_len + 2  # Account for '\n\n' separator

    print(f"DEBUG: Character map built with {len(char_map)} entries", file=sys.stderr)
    return {"char_positions": char_map}

def _find_source_block(chunk: Document, char_map: dict, original_blocks: list[dict]) -> dict | None:
    """
    Finds the original source block for a chunk using robust text matching.
    Handles cases where chunk text spans multiple source blocks, starts with headers,
    or has special formatting. Uses fallback logic to ensure all chunks are mapped.
    Ensures the returned block always has a valid 'source' dictionary with filename, page, and location.
    """
    import sys
    import os

    def ensure_source(block):
        """Ensure block has a valid source dictionary with filename, page, and location."""
        if "source" not in block or not isinstance(block["source"], dict):
            block["source"] = {}
        # Set filename
        if "filename" not in block["source"]:
            block["source"]["filename"] = "unknown"
        # Set page (None if missing or invalid)
        if "page" not in block["source"] or not isinstance(block["source"]["page"], int):
            block["source"]["page"] = None
        # Set location (None for PDFs)
        if "location" not in block["source"]:
            # Try to infer from filename extension
            filename = block["source"].get("filename", "")
            if filename.lower().endswith(".epub"):
                block["source"]["location"] = block["source"].get("location", None)
            else:
                block["source"]["location"] = None
        return block

    if not chunk or not chunk.content or not original_blocks:
        print(f"DEBUG: _find_source_block early return - chunk: {bool(chunk)}, content: {bool(chunk.content if chunk else False)}, blocks: {len(original_blocks) if original_blocks else 0}", file=sys.stderr)
        return None

    chunk_text = chunk.content.strip()
    chunk_start = chunk_text[:50].replace('\n', ' ').strip()
    print(f"DEBUG: Looking for source block for chunk starting with: '{chunk_start}...'", file=sys.stderr)

    # 1. Try exact substring match (as before)
    for i, block in enumerate(original_blocks):
        block_text = block.get("text", "")
        if chunk_start and chunk_start in block_text:
            block = ensure_source(block)
            print(f"DEBUG: Found matching source block {i} on page {block['source'].get('page', None)} (substring match)", file=sys.stderr)
            return block

    # 2. Try matching the start of the chunk to the start of any block (ignoring whitespace/case)
    for i, block in enumerate(original_blocks):
        block_text = block.get("text", "").strip()
        if not block_text:
            continue
        block_start = block_text[:max(20, len(chunk_start))].replace('\n', ' ').strip()
        if chunk_start.lower().startswith(block_start.lower()) or block_start.lower().startswith(chunk_start.lower()):
            block = ensure_source(block)
            print(f"DEBUG: Found matching source block {i} on page {block['source'].get('page', None)} (start match)", file=sys.stderr)
            return block

    # 3. Try fuzzy matching: ignore whitespace, punctuation, and case
    import re
    def normalize(s):
        return re.sub(r'[\W_]+', '', s).lower()

    norm_chunk_start = normalize(chunk_start)
    for i, block in enumerate(original_blocks):
        block_text = block.get("text", "")
        block_start = block_text[:max(20, len(chunk_start))]
        if norm_chunk_start and normalize(block_start).startswith(norm_chunk_start[:15]):
            block = ensure_source(block)
            print(f"DEBUG: Found matching source block {i} on page {block['source'].get('page', None)} (fuzzy match)", file=sys.stderr)
            return block

    # 4. Try overlap: find the block with the largest text overlap with the chunk start
    best_i = None
    best_overlap = 0
    for i, block in enumerate(original_blocks):
        block_text = block.get("text", "")
        overlap = 0
        for n in range(30, 10, -5):
            if block_text and chunk_start[:n] in block_text:
                overlap = n
                break
        if overlap > best_overlap:
            best_overlap = overlap
            best_i = i
    if best_i is not None and best_overlap > 0:
        block = ensure_source(original_blocks[best_i])
        print(f"DEBUG: Found best-overlap source block {best_i} on page {block['source'].get('page', None)} (overlap {best_overlap})", file=sys.stderr)
        return block

    # 5. As a last resort, map to the first block if chunk starts with a known header or special formatting
    if chunk_start and (chunk_start.isupper() or chunk_start.startswith("CHAPTER") or chunk_start.startswith("SECTION")):
        block = ensure_source(original_blocks[0])
        print(f"DEBUG: Fallback: mapping to first block due to header/special formatting", file=sys.stderr)
        return block

    # 6. If all else fails, map to the block with the most similar start (Levenshtein distance, if available)
    try:
        import difflib
        block_starts = [block.get("text", "")[:50] for block in original_blocks]
        matches = difflib.get_close_matches(chunk_start, block_starts, n=1, cutoff=0.6)
        if matches:
            idx = block_starts.index(matches[0])
            block = ensure_source(original_blocks[idx])
            print(f"DEBUG: Fallback: difflib matched block {idx} on page {block['source'].get('page', None)}", file=sys.stderr)
            return block
    except Exception as e:
        print(f"DEBUG: difflib fallback failed: {e}", file=sys.stderr)

    # 7. If still no match, log a warning and return the first block as a last resort
    block = ensure_source(original_blocks[0])
    print(f"WARNING: No matching source block found for chunk starting with: '{chunk_start}...'. Mapping to first block.", file=sys.stderr)
    return block
