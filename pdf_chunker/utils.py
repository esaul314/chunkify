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

def _build_char_map(blocks: list[dict]) -> dict:
    """
    Builds an enhanced mapping structure for the new block-aware chunks.
    
    Since the new chunking algorithm preserves block metadata directly in chunks,
    this function creates a comprehensive mapping that supports both traditional
    character-based lookup and the new metadata-based approach.
    """
    if not blocks:
        return {"char_positions": [], "block_lookup": {}}
    
    char_map = []
    block_lookup = {}
    current_pos = 0
    
    for i, block in enumerate(blocks):
        text_len = len(block["text"])
        
        # Traditional character position mapping
        char_entry = {
            "start": current_pos, 
            "end": current_pos + text_len, 
            "original_index": i
        }
        char_map.append(char_entry)
        
        # Enhanced block lookup with multiple access patterns
        block_lookup[i] = {
            "block": block,
            "char_start": current_pos,
            "char_end": current_pos + text_len,
            "text_preview": block["text"][:100] + "..." if len(block["text"]) > 100 else block["text"],
            "page": block.get("page"),
            "bbox": block.get("bbox")
        }
        
        # Create text-based lookup for faster searching
        text_key = block["text"][:50].strip().lower()
        if text_key and text_key not in block_lookup:
            block_lookup[f"text_{text_key}"] = i
        
        current_pos += text_len + 2  # Account for '\n\n' separator
    
    return {
        "char_positions": char_map,
        "block_lookup": block_lookup,
        "total_blocks": len(blocks)
    }

def _find_source_block(chunk: Document, char_map: dict, original_blocks: list[dict]) -> dict | None:
    """
    Finds the original source block for a chunk using the enhanced metadata approach.
    
    The new paragraph-aware chunking algorithm stores source block information
    directly in chunk metadata, making this lookup much more reliable than
    the previous text-search approach.
    """
    if not chunk or not chunk.content or not original_blocks:
        return None
    
    # First, try to use the enhanced metadata from paragraph-aware chunking
    if hasattr(chunk, 'meta') and chunk.meta:
        source_blocks = chunk.meta.get("source_blocks", [])
        
        if source_blocks:
            # Use the first source block as the primary source
            primary_source = source_blocks[0]
            block_index = primary_source.get("block_index")
            
            # Validate the block index
            if block_index is not None and 0 <= block_index < len(original_blocks):
                return original_blocks[block_index]
            
            # If block_index is invalid, try to find by bbox or text preview
            for source_info in source_blocks:
                bbox = source_info.get("bbox")
                text_preview = source_info.get("text_preview", "")
                
                if bbox:
                    # Find block with matching bbox
                    for i, block in enumerate(original_blocks):
                        if block.get("bbox") == bbox:
                            return block
                
                if text_preview:
                    # Find block with matching text preview
                    preview_start = text_preview[:30].strip()
                    for i, block in enumerate(original_blocks):
                        if block["text"].strip().startswith(preview_start):
                            return block
    
    # Fallback to the enhanced character map approach
    block_lookup = char_map.get("block_lookup", {})
    chunk_start = chunk.content.strip()[:50].lower()
    
    # Try text-based lookup first
    text_key = f"text_{chunk_start}"
    if text_key in block_lookup:
        block_index = block_lookup[text_key]
        if isinstance(block_index, int) and 0 <= block_index < len(original_blocks):
            return original_blocks[block_index]
    
    # Fallback to the original text search method with improvements
    chunk_start_search = chunk.content.strip()[:30]
    candidates = []
    
    for i, block in enumerate(original_blocks):
        block_text = block["text"]
        if chunk_start_search in block_text:
            position = block_text.find(chunk_start_search)
            candidates.append({
                "block_index": i,
                "position": position,
                "confidence": len(chunk_start_search) / len(block_text)  # Simple confidence score
            })
    
    if candidates:
        # Sort by confidence (higher is better), then by block index, then by position
        candidates.sort(key=lambda c: (-c["confidence"], c["block_index"], c["position"]))
        best_match = candidates[0]
        return original_blocks[best_match["block_index"]]
    
    return None
