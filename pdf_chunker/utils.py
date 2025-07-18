import re
import uuid
import textstat
from haystack.dataclasses import Document
from .ai_enrichment import classify_chunk_utterance
from concurrent.futures import ThreadPoolExecutor, as_completed

def _compute_readability(text: str) -> dict:
    """Computes readability scores and returns them as a dictionary."""
    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "difficulty": textstat.difficult_words(text)
    }

def _generate_chunk_id(filename: str, page: int, chunk_index: int) -> str:
    """Generates a unique chunk ID."""
    return f"{filename}-p{page}-c{chunk_index}"

def format_chunks_with_metadata(
    haystack_chunks: list[Document],
    original_blocks: list[dict],
    generate_metadata: bool = True,
    perform_ai_enrichment: bool = False, # New flag
    max_workers: int = 10
) -> list[dict]:
    """
    Formats final chunks, enriching them in parallel with detailed metadata.
    """
    import sys
    
    print(f"DEBUG: format_chunks_with_metadata called with {len(haystack_chunks)} chunks and {len(original_blocks)} original blocks", file=sys.stderr)
    
    # Debug: Check what pages are in the original blocks
    original_pages = set()
    for block in original_blocks:
        page = block.get('source', {}).get('page')
        if page:
            original_pages.add(page)
    print(f"DEBUG: Original blocks contain pages: {sorted(original_pages)}", file=sys.stderr)
    
    char_map = _build_char_map(original_blocks)
    
    def process_chunk(chunk, chunk_index):
        final_text = chunk.content.strip()
        if not final_text:
            return None

        print(f"DEBUG: Processing chunk {chunk_index} with {len(final_text)} characters", file=sys.stderr)

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

            print(f"Truncated chunk {chunk_index} to {len(final_text)} characters", file=sys.stderr)
    

        if not generate_metadata:
            return {"text": final_text}

        source_block = _find_source_block(chunk, char_map, original_blocks)
        if not source_block:
            return None  # Or handle as an error

        filename = source_block.get("source", {}).get("filename", "Unknown")
        page = source_block.get("source", {}).get("page", 0)

        # AI classification is only done if the flag is set
        utterance_type = classify_chunk_utterance(final_text) if perform_ai_enrichment else "disabled"

        metadata = {
            "source": filename,
            "chunk_id": _generate_chunk_id(filename, page, chunk_index),
            "page": page,
            "location": source_block.get("source", {}).get("location"),
            "block_type": source_block.get("type", "paragraph"),
            "language": source_block.get("language", "un"),
            "readability": _compute_readability(final_text),
            "utterance_type": utterance_type,
            "importance": "medium",
        }

        return {
            "text": final_text,
            "metadata": {k: v for k, v in metadata.items() if v is not None}
        }

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
    Finds the original source block for a chunk using simple text search.
    """
    import sys
    
    if not chunk or not chunk.content or not original_blocks:
        print(f"DEBUG: _find_source_block early return - chunk: {bool(chunk)}, content: {bool(chunk.content if chunk else False)}, blocks: {len(original_blocks) if original_blocks else 0}", file=sys.stderr)
        return None
    
    # Simple approach: find the block that contains the start of this chunk
    chunk_start = chunk.content.strip()[:30]
    print(f"DEBUG: Looking for source block for chunk starting with: '{chunk_start}...'", file=sys.stderr)
    
    for i, block in enumerate(original_blocks):
        if chunk_start in block["text"]:
            page = block.get('source', {}).get('page', 'unknown')
            print(f"DEBUG: Found matching source block {i} on page {page}", file=sys.stderr)
            return block
    
    print(f"DEBUG: No matching source block found for chunk starting with: '{chunk_start}...'", file=sys.stderr)
    return None
