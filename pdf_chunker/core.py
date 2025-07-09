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
    
    if haystack_chunks:
        chunk_sizes = [len(chunk.content) for chunk in haystack_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        
        print(f"Chunk size statistics after semantic chunking:", file=sys.stderr)
        print(f"  Average: {avg_size:.0f} characters", file=sys.stderr)
        print(f"  Maximum: {max_size} characters", file=sys.stderr)
        print(f"  Minimum: {min_size} characters", file=sys.stderr)
        
        # Check for problematic chunks
        oversized_chunks = [i for i, size in enumerate(chunk_sizes) if size > 10000]
        if oversized_chunks:
            print(f"  WARNING: {len(oversized_chunks)} chunks exceed 10k characters", file=sys.stderr)
            for i in oversized_chunks[:3]:  # Show first 3 oversized
                print(f"    Chunk {i}: {chunk_sizes[i]} characters", file=sys.stderr)
        
        extreme_chunks = [i for i, size in enumerate(chunk_sizes) if size > 25000]
        if extreme_chunks:
            print(f"  CRITICAL: {len(extreme_chunks)} chunks exceed 25k characters!", file=sys.stderr)
            for i in extreme_chunks:
                print(f"    Chunk {i}: {chunk_sizes[i]} characters", file=sys.stderr)
    
    # 3. Final Formatting and AI Enrichment
    final_chunks = format_chunks_with_metadata(
        haystack_chunks,
        structured_blocks,
        generate_metadata=generate_metadata,
        perform_ai_enrichment=perform_ai_enrichment
    )
    
    # Final validation of chunk sizes
    print(f"Final pipeline output: {len(final_chunks)} chunks", file=sys.stderr)
    
    if final_chunks:
        final_sizes = [len(chunk.get("text", "")) for chunk in final_chunks]
        final_avg = sum(final_sizes) / len(final_sizes)
        final_max = max(final_sizes)
        final_min = min(final_sizes)
        
        print(f"Final chunk size statistics:", file=sys.stderr)
        print(f"  Average: {final_avg:.0f} characters", file=sys.stderr)
        print(f"  Maximum: {final_max} characters", file=sys.stderr)
        print(f"  Minimum: {final_min} characters", file=sys.stderr)
        
        # Critical validation for JSONL output
        oversized_final = [i for i, size in enumerate(final_sizes) if size > 10000]
        if oversized_final:
            print(f"  ERROR: {len(oversized_final)} final chunks exceed 10k characters!", file=sys.stderr)
            for i in oversized_final[:3]:
                print(f"    Final chunk {i}: {final_sizes[i]} characters", file=sys.stderr)
        
        extreme_final = [i for i, size in enumerate(final_sizes) if size > 25000]
        if extreme_final:
            print(f"  CRITICAL ERROR: {len(extreme_final)} final chunks exceed 25k characters!", file=sys.stderr)
            print(f"  These will create extremely long JSONL lines!", file=sys.stderr)
            for i in extreme_final:
                print(f"    Final chunk {i}: {final_sizes[i]} characters", file=sys.stderr)
    
    for i, chunk in enumerate(final_chunks[:3]):  # First 3 chunks
        text_len = len(chunk.get("text", ""))
        print(f"Final chunk {i}: {text_len} characters", file=sys.stderr)
        if text_len > 10000:
            print(f"ERROR: Final chunk {i} is still oversized ({text_len} characters)!", file=sys.stderr)
    
    return final_chunks
