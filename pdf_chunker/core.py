from .parsing import extract_structured_text
from .splitter import semantic_chunker
from .ai_enrichment import init_llm

import sys
import logging
logger = logging.getLogger(__name__)

def format_chunks_with_metadata(
    chunks: list,
    original_blocks: list,
    filename: str = None,
    min_chunk_size: int = None,
    enable_dialogue_detection: bool = False
) -> list:
    """Format chunks with metadata and validate JSON serialization."""
    logger.debug(f"format_chunks_with_metadata called with {len(chunks)} chunks and {len(original_blocks)} original blocks")
    logger.debug(f"min_chunk_size={min_chunk_size}, enable_dialogue_detection={enable_dialogue_detection}")

    formatted_chunks = []

    for i, chunk_text in enumerate(chunks):
        logger.debug(f"process_chunk() ENTRY - chunk {i}")
        logger.debug(f"process_chunk() - chunk {i} has {len(chunk_text)} characters")

        chunk_preview = chunk_text[:100].replace('\n', '\\n')
        logger.debug(f"process_chunk() - chunk {i} preview: '{chunk_preview}...'")

        # Check for potential JSON escaping issues
        quote_count = chunk_text.count('"')
        if quote_count > 0:
            logger.debug(f"process_chunk() - chunk {i} contains {quote_count} quote characters")

        # Placeholder for metadata generation (should be replaced with actual logic)
        metadata = {
            "source": filename,
            "chunk_index": i,
            "min_chunk_size": min_chunk_size,
            "enable_dialogue_detection": enable_dialogue_detection
        }

        try:
            # Normalize quotes before creating chunk object
            normalized_text = _normalize_quotes_for_json(chunk_text)

            # Create the chunk object
            chunk_obj = {
                "text": normalized_text,
                "metadata": metadata
            }

            # Test JSON serialization to catch escaping issues early
            import json
            json_test = json.dumps(chunk_obj, ensure_ascii=False)
            logger.debug(f"process_chunk() - chunk {i} JSON serialization test passed")

            formatted_chunks.append(chunk_obj)
            logger.debug(f"process_chunk() EXIT - chunk {i} SUCCESS - result has {len(normalized_text)} chars")

        except (json.JSONEncodeError, UnicodeEncodeError) as e:
            logger.error(f"process_chunk() - chunk {i} JSON serialization FAILED: {e}")
            logger.error(f"Problematic text preview: '{chunk_text[:200]}'")

            # Try progressive repair strategies
            repaired_text = _repair_json_escaping_issues(chunk_text)

            try:
                chunk_obj = {
                    "text": repaired_text,
                    "metadata": metadata
                }
                json.dumps(chunk_obj, ensure_ascii=False)
                formatted_chunks.append(chunk_obj)
                logger.warning(f"process_chunk() - chunk {i} repaired with advanced escaping fixes")
            except Exception as e2:
                logger.error(f"process_chunk() - chunk {i} still failing after repair attempt: {e2}")
                # Create a safe fallback version
                safe_text = _create_safe_fallback_text(chunk_text)
                chunk_obj = {
                    "text": safe_text,
                    "metadata": metadata
                }
                formatted_chunks.append(chunk_obj)
                logger.warning(f"process_chunk() - chunk {i} using safe fallback text")

    logger.info(f"Final pipeline output: {len(formatted_chunks)} chunks")

    return formatted_chunks

def _normalize_quotes_for_json(text: str) -> str:
    """Normalize quotes in text to prevent JSON serialization issues."""
    if not text:
        return text

    import re

    # Step 1: Normalize different quote types to standard quotes
    # Convert smart quotes to standard quotes
    text = text.replace('“', '"').replace('”', '"')  # Smart double quotes
    text = text.replace("‘", "'").replace("’", "'")  # Smart single quotes

    # Step 2: Fix common quote escaping patterns that cause issues
    # Remove any existing backslash escaping that might be incorrect
    text = text.replace('\\\\\"', '\\"').replace("\\\\'", "\\'")

    # Step 3: Handle problematic quote sequences
    # Fix quotes at the beginning of text that might cause issues
    text = re.sub(r'^[\\"\\s]*\\"([^\\"])', r'"\1', text)

    # Fix quotes at the end of text
    text = re.sub(r'([^\\"])\"[\"\\s]*$', r'\1\"', text)

    # Step 4: Ensure balanced quotes where possible
    text = _balance_quotes_if_possible(text)

    return text

def _balance_quotes_if_possible(text: str) -> str:
    """Attempt to balance quotes in text without changing meaning."""
    if not text:
        return text

    # Count unescaped quotes
    double_quote_count = text.count('"')
    single_quote_count = text.count("'")

    # If we have an odd number of double quotes, try to fix
    if double_quote_count % 2 == 1:
        import logging
        logger = logging.getLogger(__name__)
        # Pattern 1: Quote at start but no closing quote
        if text.startswith('"') and not text.rstrip().endswith('"'):
            if len(text) > 10 and not text[1:].startswith('"'):
                text = text + '"'
        # Pattern 2: Quote at end but no opening quote
        elif text.rstrip().endswith('"') and not text.startswith('"'):
            if len(text) > 10:
                text = '"' + text
        # Pattern 3: Quote in middle - more complex, be conservative
        else:
            logger.debug(f"Unbalanced quotes detected but no simple fix available: '{text[:50]}...'")
    return text

def _repair_json_escaping_issues(text: str) -> str:
    """Apply progressive repair strategies for JSON escaping issues."""
    if not text:
        return text
    import re

    repaired = text

    # Strategy 1: Fix common problematic patterns
    # Pattern: '", ' at start
    if repaired.startswith('", '):
        repaired = repaired[3:]
    # Pattern: Unescaped quotes in the middle of text
    repaired = re.sub(r'([a-zA-Z])"([a-zA-Z])', r'\1\\\"\2', repaired)
    # Strategy 2: Handle control characters that might cause issues
    repaired = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', repaired)
    # Strategy 3: Fix newline handling
    repaired = repaired.replace('\r\n', '\\n').replace('\r', '\\n')
    # Strategy 4: Handle Unicode issues
    try:
        repaired = repaired.encode('utf-8').decode('utf-8')
    except UnicodeError:
        repaired = repaired.encode('utf-8', errors='replace').decode('utf-8')
    return repaired

def _create_safe_fallback_text(text: str) -> str:
    """Create a safe fallback version of text that will definitely serialize to JSON."""
    if not text:
        return ""
    import re

    safe_text = text
    # Remove all quotes to eliminate escaping issues
    safe_text = safe_text.replace('"', '').replace("'", "")
    # Remove problematic characters
    safe_text = re.sub(r'[^\w\s\.\,!\?\:\;\-\(\)]', ' ', safe_text)
    # Normalize whitespace
    safe_text = ' '.join(safe_text.split())
    # Truncate if too long (safety measure)
    if len(safe_text) > 1000:
        safe_text = safe_text[:997] + "..."
    # Add a note that this is a fallback
    safe_text = "[TEXT CLEANED] " + safe_text
    return safe_text

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

    # 1. Structural Pass: Extract text into structured blocks

    structured_blocks = extract_structured_text(filepath, exclude_pages=exclude_pages)

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
    # FIX: Convert list of block texts to a single string for semantic_chunker
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

    # 3. Final Formatting and AI Enrichment with conversational text metadata
    final_chunks = format_chunks_with_metadata(
        haystack_chunks,
        structured_blocks,
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
