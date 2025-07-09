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
    char_map = _build_char_map(original_blocks)
    
    def process_chunk(chunk, chunk_index):
        final_text = chunk.content.strip()
        if not final_text:
            return None


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
    if not blocks:
        return {"char_positions": []}
    
    char_map = []
    current_pos = 0
    
    for i, block in enumerate(blocks):
        text_len = len(block["text"])
        char_entry = {
            "start": current_pos, 
            "end": current_pos + text_len, 
            "original_index": i
        }
        char_map.append(char_entry)
        current_pos += text_len + 2  # Account for '\n\n' separator
    
    return {"char_positions": char_map}

def _find_source_block(chunk: Document, char_map: dict, original_blocks: list[dict]) -> dict | None:
    """
    Finds the original source block for a chunk using simple text search.
    """
    if not chunk or not chunk.content or not original_blocks:
        return None
    
    # Simple approach: find the block that contains the start of this chunk
    chunk_start = chunk.content.strip()[:30]
    
    for i, block in enumerate(original_blocks):
        if chunk_start in block["text"]:
            return block
    
    return None
