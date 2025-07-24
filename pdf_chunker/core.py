from .parsing import extract_structured_text
from .splitter import semantic_chunker
from .ai_enrichment import init_llm
from .utils import format_chunks_with_metadata as utils_format_chunks_with_metadata
from haystack.dataclasses import Document

import sys
import logging
logger = logging.getLogger(__name__)

def process_document(
    filepath: str,
    chunk_size: int,
    overlap: int,
    generate_metadata: bool = True,
    ai_enrichment: bool = True,  # New flag to control AI calls
    exclude_pages: str = None,  # New parameter for page exclusion
    min_chunk_size: int = None,  # New parameter for conversational text handling
    enable_dialogue_detection: bool = True  # New parameter to control dialogue pattern detection
) -> list[dict]:
    """
    Core pipeline for processing a document with optional AI enrichment and conversational text handling.

    Args:
        filepath: Path to the document to process
        chunk_size: Target chunk size in words
        overlap: Overlap size in words
        generate_metadata: Whether to generate metadata
        ai_enrichment: Whether to perform AI enrichment
        exclude_pages: Page ranges to exclude (e.g., "1,3,5-10,15-20")
        min_chunk_size: Minimum chunk size in words (defaults to max(8, chunk_size // 10))
        enable_dialogue_detection: Whether to enable dialogue pattern detection for conversational text
    """

    # Set default minimum chunk size if not provided
    if min_chunk_size is None:
        min_chunk_size = max(8, chunk_size // 10)  # Minimum 8 words or 10% of target size

    # Determine if AI enrichment should be performed
    perform_ai_enrichment = generate_metadata and ai_enrichment

    if perform_ai_enrichment:
        try:
            init_llm()
        except ValueError as e:
            print(f"AI Enrichment disabled: {e}", file=sys.stderr)
            perform_ai_enrichment = False


    # Parse excluded pages as a set of integers
    excluded_pages_set = set()
    if exclude_pages:
        try:
            from .page_utils import parse_page_ranges
            excluded_pages_set = parse_page_ranges(exclude_pages)
            print(f"DEBUG: Parsed excluded pages: {sorted(excluded_pages_set)}", file=sys.stderr)
        except Exception as e:
            print(f"Error parsing exclude_pages: {e}", file=sys.stderr)
            excluded_pages_set = set()

    # 1. Structural Pass: Extract text into structured blocks, passing excluded pages
    from .pdf_parsing import extract_text_blocks_from_pdf
    structured_blocks = extract_text_blocks_from_pdf(filepath, exclude_pages=exclude_pages)
    print(f"DEBUG: After PDF extraction, got {len(structured_blocks)} blocks", file=sys.stderr)

    # Filter out any blocks from excluded pages (defensive, in case enhancement/fallbacks leak them)
    filtered_blocks = []
    for block in structured_blocks:
        page = block.get("source", {}).get("page")
        if page is not None and page in excluded_pages_set:
            print(f"DEBUG: Filtering out block from excluded page {page}", file=sys.stderr)
            continue
        filtered_blocks.append(block)

    print(f"DEBUG: After filtering excluded pages, have {len(filtered_blocks)} blocks", file=sys.stderr)
    structured_blocks = filtered_blocks

    # Verify no excluded pages remain
    remaining_pages = set()
    for block in structured_blocks:
        page = block.get("source", {}).get("page")
        if page is not None:
            remaining_pages.add(page)

    print(f"DEBUG: Remaining pages after filtering: {sorted(remaining_pages)}", file=sys.stderr)

    # Check for any intersection with excluded pages
    leaked_pages = remaining_pages.intersection(excluded_pages_set)
    if leaked_pages:
        print(f"DEBUG: ERROR - Excluded pages still present after filtering: {sorted(leaked_pages)}", file=sys.stderr)
    else:
        print(f"DEBUG: SUCCESS - No excluded pages found in structured blocks", file=sys.stderr)
    

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

    # 2. Semantic Pass: Chunk the blocks into coherent documents with conversational text handling
    full_text = "\n\n".join(block.get("text", "") for block in structured_blocks if block.get("text", ""))
    haystack_chunks = semantic_chunker(
        full_text,
        chunk_size,
        overlap,
        min_chunk_size=min_chunk_size,
        enable_dialogue_detection=enable_dialogue_detection
    )

    # Debug: Validate chunk sizes after conversational text handling
    print(f"Semantic chunking with conversational text handling produced {len(haystack_chunks)} chunks", file=sys.stderr)
    
    if haystack_chunks:
        chunk_sizes = [len(chunk) for chunk in haystack_chunks]
        word_counts = [len(chunk.split()) for chunk in haystack_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        avg_words = sum(word_counts) / len(word_counts)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        max_words = max(word_counts)
        min_words = min(word_counts)

        print(f"Chunk size statistics after conversational text handling:", file=sys.stderr)
        print(f"  Average: {avg_size:.0f} characters ({avg_words:.1f} words)", file=sys.stderr)
        print(f"  Maximum: {max_size} characters ({max_words} words)", file=sys.stderr)
        print(f"  Minimum: {min_size} characters ({min_words} words)", file=sys.stderr)

        # Check for problematic chunks after conversational text handling
        short_chunks = [i for i, words in enumerate(word_counts) if words <= 7]
        very_short_chunks = [i for i, words in enumerate(word_counts) if words <= 3]

        if short_chunks:
            print(f"  Short chunks (≤7 words): {len(short_chunks)} chunks", file=sys.stderr)
            if len(short_chunks) <= 3:  # Show examples if not too many
                for i in short_chunks:
                    chunk_preview = haystack_chunks[i][:50].replace('\n', ' ')
                    print(f"    Chunk {i}: {word_counts[i]} words - '{chunk_preview}...'", file=sys.stderr)

        if very_short_chunks:
            print(f"  Very short chunks (≤3 words): {len(very_short_chunks)} chunks", file=sys.stderr)
            for i in very_short_chunks:
                chunk_preview = haystack_chunks[i][:50].replace('\n', ' ')
                print(f"    Chunk {i}: {word_counts[i]} words - '{chunk_preview}...'", file=sys.stderr)

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

    # 3. Convert text chunks to Haystack Document objects for utils compatibility
    haystack_documents = []
    for i, chunk_text in enumerate(haystack_chunks):
        doc = Document(content=chunk_text, id=f"chunk_{i}")
        haystack_documents.append(doc)

    # 4. Final Formatting and AI Enrichment using utils function with proper schema
    final_chunks = utils_format_chunks_with_metadata(
        haystack_documents,
        structured_blocks,
        generate_metadata=generate_metadata,
        perform_ai_enrichment=perform_ai_enrichment,
        max_workers=10,
        min_chunk_size=min_chunk_size,
        enable_dialogue_detection=enable_dialogue_detection
    )

    # Debug: Validate that excluded pages don't appear in final chunks
    if exclude_pages and generate_metadata:
        print(f"DEBUG: Validating page exclusions in final chunks...", file=sys.stderr)

        # Parse the excluded pages for validation
        try:
            from .page_utils import parse_page_ranges
            excluded_pages = parse_page_ranges(exclude_pages)
            print(f"DEBUG: Should exclude pages: {sorted(excluded_pages)}", file=sys.stderr)

            # Check what pages appear in final chunks
            final_chunk_pages = set()
            for i, chunk in enumerate(final_chunks):
                if chunk and 'metadata' in chunk:
                    page = chunk['metadata'].get('page')
                    if page:
                        final_chunk_pages.add(page)
                        if page in excluded_pages:
                            print(f"DEBUG: ERROR - Excluded page {page} found in final chunk {i}!", file=sys.stderr)

            print(f"DEBUG: Final chunks contain pages: {sorted(final_chunk_pages)}", file=sys.stderr)

            # Check for intersection
            leaked_pages = final_chunk_pages.intersection(excluded_pages)
            if leaked_pages:
                print(f"DEBUG: CRITICAL ERROR - Excluded pages leaked into final output: {sorted(leaked_pages)}", file=sys.stderr)
            else:
                print(f"DEBUG: SUCCESS - No excluded pages found in final output", file=sys.stderr)

        except Exception as e:
            print(f"DEBUG: Error validating page exclusions: {e}", file=sys.stderr)

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
