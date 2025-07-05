import logging
from typing import List
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def semantic_chunker(structured_blocks: list[dict], chunk_size: int, overlap: int) -> list[Document]:
    """
    Chunks the document using a paragraph-aware approach that respects block boundaries
    while maintaining optimal chunk sizes for LoRA training and RAG applications.
    
    This implementation:
    1. Preserves the natural paragraph/block structure identified during PDF parsing
    2. Merges small adjacent blocks when semantically appropriate
    3. Splits oversized blocks internally while respecting sentence boundaries
    4. Maintains target chunk sizes (typically 200-800 tokens for embedding models)
    """
    if not structured_blocks:
        return []
    
    logging.info(f"Starting paragraph-aware chunking: {len(structured_blocks)} blocks, target size={chunk_size} words, overlap={overlap} words")
    
    # Initialize the DocumentSplitter for internal splitting of oversized blocks
    splitter = DocumentSplitter(
        split_by="word",
        split_length=chunk_size,
        split_overlap=overlap,
        respect_sentence_boundary=True
    )
    splitter.warm_up()
    
    chunks = []
    current_chunk_blocks = []
    current_chunk_word_count = 0
    
    # Define size thresholds
    min_chunk_size = max(50, chunk_size // 8)  # Minimum viable chunk size
    max_chunk_size = chunk_size + (overlap // 2)  # Allow some flexibility above target
    oversized_threshold = chunk_size * 2  # Blocks larger than this need internal splitting
    
    def _count_words(text: str) -> int:
        """Count words in text, handling empty strings gracefully."""
        return len(text.split()) if text.strip() else 0
    
    def _create_chunk_from_blocks(blocks: list[dict]) -> Document:
        """Create a Document chunk from a list of blocks."""
        if not blocks:
            return None
        
        # Combine block texts with double newlines to preserve paragraph structure
        chunk_text = "\n\n".join(block["text"].strip() for block in blocks if block["text"].strip())
        
        if not chunk_text:
            return None
        
        # Store metadata about source blocks for improved mapping
        chunk_metadata = {
            "source_blocks": [
                {
                    "block_index": i,
                    "bbox": block.get("bbox"),
                    "page": block.get("page"),
                    "text_preview": block["text"][:50] + "..." if len(block["text"]) > 50 else block["text"]
                }
                for i, block in enumerate(blocks)
            ],
            "block_count": len(blocks),
            "chunk_type": "paragraph_aware"
        }
        
        return Document(content=chunk_text, meta=chunk_metadata)
    
    def _split_oversized_block(block: dict) -> List[Document]:
        """Split a single oversized block using sentence boundaries."""
        block_text = block["text"].strip()
        if not block_text:
            return []
        
        # Use the DocumentSplitter to split this block internally
        temp_doc = Document(content=block_text)
        result = splitter.run(documents=[temp_doc])
        split_docs = result.get("documents", [])
        
        # Add metadata to each split chunk indicating it came from a single block
        for doc in split_docs:
            doc.meta = {
                "source_blocks": [{
                    "block_index": 0,  # Will be updated by caller
                    "bbox": block.get("bbox"),
                    "page": block.get("page"),
                    "text_preview": block["text"][:50] + "..." if len(block["text"]) > 50 else block["text"],
                    "split_from_oversized": True
                }],
                "block_count": 1,
                "chunk_type": "split_oversized"
            }
        
        logging.info(f"Split oversized block ({_count_words(block_text)} words) into {len(split_docs)} chunks")
        return split_docs
    
    def _finalize_current_chunk():
        """Finalize the current chunk and add it to the chunks list."""
        if current_chunk_blocks:
            chunk = _create_chunk_from_blocks(current_chunk_blocks)
            if chunk:
                chunks.append(chunk)
                logging.debug(f"Created chunk with {len(current_chunk_blocks)} blocks, {_count_words(chunk.content)} words")
    
    # Process each block
    for block_idx, block in enumerate(structured_blocks):
        block_text = block["text"].strip()
        if not block_text:
            continue
        
        block_word_count = _count_words(block_text)
        
        # Handle oversized blocks by splitting them internally
        if block_word_count > oversized_threshold:
            # Finalize any current chunk first
            _finalize_current_chunk()
            current_chunk_blocks = []
            current_chunk_word_count = 0
            
            # Split the oversized block and add each piece as a separate chunk
            split_chunks = _split_oversized_block(block)
            for chunk in split_chunks:
                # Update the block index in metadata
                if chunk.meta and "source_blocks" in chunk.meta:
                    chunk.meta["source_blocks"][0]["block_index"] = block_idx
                chunks.append(chunk)
            
            continue
        
        # Check if adding this block would exceed our target size
        projected_size = current_chunk_word_count + block_word_count
        
        if current_chunk_blocks and projected_size > max_chunk_size:
            # Current chunk is full, finalize it
            _finalize_current_chunk()
            current_chunk_blocks = []
            current_chunk_word_count = 0
        
        # Add the block to the current chunk
        current_chunk_blocks.append(block)
        current_chunk_word_count += block_word_count
        
        # If this single block makes a good-sized chunk, finalize it
        if current_chunk_word_count >= chunk_size and len(current_chunk_blocks) == 1:
            _finalize_current_chunk()
            current_chunk_blocks = []
            current_chunk_word_count = 0
    
    # Finalize any remaining chunk
    _finalize_current_chunk()
    
    # Post-process: merge any very small chunks with adjacent chunks
    final_chunks = []
    i = 0
    while i < len(chunks):
        current_chunk = chunks[i]
        current_word_count = _count_words(current_chunk.content)
        
        # If this chunk is too small and there's a next chunk, try to merge
        if (current_word_count < min_chunk_size and 
            i + 1 < len(chunks) and 
            _count_words(chunks[i + 1].content) + current_word_count <= max_chunk_size):
            
            # Merge with next chunk
            next_chunk = chunks[i + 1]
            merged_content = current_chunk.content + "\n\n" + next_chunk.content
            
            # Combine metadata
            merged_meta = {
                "source_blocks": (current_chunk.meta.get("source_blocks", []) + 
                                next_chunk.meta.get("source_blocks", [])),
                "block_count": (current_chunk.meta.get("block_count", 0) + 
                              next_chunk.meta.get("block_count", 0)),
                "chunk_type": "merged_small"
            }
            
            merged_chunk = Document(content=merged_content, meta=merged_meta)
            final_chunks.append(merged_chunk)
            logging.debug(f"Merged small chunk ({current_word_count} words) with next chunk")
            i += 2  # Skip the next chunk since we merged it
        else:
            final_chunks.append(current_chunk)
            i += 1
    
    logging.info(f"Paragraph-aware chunking completed: {len(final_chunks)} chunks created from {len(structured_blocks)} blocks")
    
    # Debug: Log chunk size distribution
    chunk_sizes = [_count_words(chunk.content) for chunk in final_chunks]
    if chunk_sizes:
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        logging.info(f"Chunk size stats - Avg: {avg_size:.1f}, Min: {min_size}, Max: {max_size} words")
    
    return final_chunks
