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

def _build_char_map(blocks: list[dict]) -> list[dict]:
    """Builds a character map to link chunks back to original blocks."""
    char_map, current_pos = [], 0
    for i, block in enumerate(blocks):
        text_len = len(block["text"])
        char_map.append({"start": current_pos, "end": current_pos + text_len, "original_index": i})
        current_pos += text_len + 2  # Account for '\n\n' separator
    return char_map

def _find_source_block(chunk: Document, char_map: list[dict], original_blocks: list[dict]) -> dict | None:
    """Finds the original source block for a given Haystack chunk."""
    # The new Haystack Document doesn't have the same char offset metadata.
    # We must find the source block by matching the start of the chunk content.
    # This is less precise but works for this implementation.
    chunk_content_start = chunk.content.strip()
    full_text = "\n\n".join(b["text"] for b in original_blocks)
    
    # Find the character position of our chunk in the full text
    chunk_start_char = full_text.find(chunk_content_start)

    if chunk_start_char == -1:
        return None # Chunk not found in original text

    for mapping in char_map:
        # If the chunk's starting position falls within a block's range
        if mapping["start"] <= chunk_start_char < mapping["end"]:
            return original_blocks[mapping["original_index"]]
            
    return None
