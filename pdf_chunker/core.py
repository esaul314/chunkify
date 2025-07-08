from .parsing import extract_structured_text
from .splitter import semantic_chunker
from .utils import format_chunks_with_metadata
from .ai_enrichment import init_llm

import sys

def process_document(
    filepath: str,
    chunk_size: int,
    overlap: int,
    generate_metadata: bool = True,
    ai_enrichment: bool = True  # New flag to control AI calls
) -> list[dict]:
    """
    Core pipeline for processing a document with optional AI enrichment.
    """
    # Determine if AI enrichment should be performed
    perform_ai_enrichment = generate_metadata and ai_enrichment
    
    if perform_ai_enrichment:
        try:
            init_llm()
        except ValueError as e:
            print(f"AI Enrichment disabled: {e}", file=sys.stderr)
            perform_ai_enrichment = False

    # 1. Structural Pass: Extract text into structured blocks
    structured_blocks = extract_structured_text(filepath)
    
    # Debug: Show what we got from the structural pass
    print(f"Extracted {len(structured_blocks)} structured blocks", file=sys.stderr)
    total_block_chars = 0
    for i, block in enumerate(structured_blocks[:5]):  # First 5 blocks
        block_text = block.get("text", "")
        block_chars = len(block_text)
        total_block_chars += block_chars
        preview = block_text[:50] if block_text else "Empty text"
        page_info = f"page {block.get('source', {}).get('page', 'unknown')}" if "source" in block else "unknown page"
        print(f"Block {i} ({page_info}): {block_chars} chars - {preview}...", file=sys.stderr)
    
    # Calculate total characters in all blocks
    total_all_chars = sum(len(block.get("text", "")) for block in structured_blocks)
    print(f"Total characters in all blocks: {total_all_chars}", file=sys.stderr)
    
    # 2. Semantic Pass: Chunk the blocks into coherent documents
    haystack_chunks = semantic_chunker(structured_blocks, chunk_size, overlap)
    
    # Debug: Validate chunk sizes after semantic chunking
    print(f"Semantic chunking produced {len(haystack_chunks)} chunks", file=sys.stderr)
    for i, chunk in enumerate(haystack_chunks[:3]):  # First 3 chunks
        chunk_chars = len(chunk.content)
        chunk_words = len(chunk.content.split())
        print(f"Chunk {i}: {chunk_chars} chars, {chunk_words} words", file=sys.stderr)
        if chunk_chars > 10000:
            print(f"WARNING: Chunk {i} is very large ({chunk_chars} characters)!", file=sys.stderr)
    
    # 3. Final Formatting and AI Enrichment
    final_chunks = format_chunks_with_metadata(
        haystack_chunks,
        structured_blocks,
        generate_metadata=generate_metadata,
        perform_ai_enrichment=perform_ai_enrichment
    )
    
    # Final validation of chunk sizes
    print(f"Final pipeline output: {len(final_chunks)} chunks", file=sys.stderr)
    for i, chunk in enumerate(final_chunks[:3]):  # First 3 chunks
        text_len = len(chunk.get("text", ""))
        print(f"Final chunk {i}: {text_len} characters", file=sys.stderr)
        if text_len > 10000:
            print(f"ERROR: Final chunk {i} is still oversized ({text_len} characters)!", file=sys.stderr)
    
    return final_chunks
