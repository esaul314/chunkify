import re
import uuid
import textstat
from haystack.schema import Document

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
    generate_metadata: bool = True
) -> list[dict]:
    """
    Formats the final chunks and enriches them with detailed, RAG-supportive metadata.
    This implementation is designed to be functional and declarative.
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
            return None # Or handle as an error

        filename = source_block.get("source", {}).get("filename", "Unknown")
        page = source_block.get("source", {}).get("page", 0)

        metadata = {
            "source": filename,
            "chunk_id": _generate_chunk_id(filename, page, chunk_index),
            "page": page,
            "location": source_block.get("source", {}).get("location"),
            "block_type": source_block.get("type", "paragraph"),
            "language": source_block.get("language", "un"),
            "readability": _compute_readability(final_text),
            # --- Placeholder metadata fields ---
            "utterance_type": "tbd",
            "importance": "medium", # Default value
        }

        return {
            "text": final_text,
            "metadata": {k: v for k, v in metadata.items() if v is not None}
        }

    # Process all chunks and filter out any None results
    final_chunks = [process_chunk(chunk, i) for i, chunk in enumerate(haystack_chunks)]
    return [chunk for chunk in final_chunks if chunk]

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
    chunk_start = chunk.meta.get("_split_start_char", 0)
    for mapping in char_map:
        if mapping["start"] <= chunk_start < mapping["end"]:
            return original_blocks[mapping["original_index"]]
    return None
