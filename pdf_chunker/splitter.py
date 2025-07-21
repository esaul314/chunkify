import logging
import re
from typing import List, Dict, Any, Tuple
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Centralized chunk threshold constants
CHUNK_THRESHOLDS = {
    'very_short': 5,          # Very short chunks (≤5 words) - always merge
    'short': 10,              # Short chunks (≤10 words) - consider for merging
    'min_target': 8,          # Minimum target chunk size after merging
    'dialogue_response': 6,   # Short dialogue responses (≤6 words)
    'fragment': 4,            # Very short fragments (≤4 words) - always merge
    'related_short': 8        # Related short chunks threshold
}

def detect_dialogue_patterns(text: str) -> List[Dict[str, Any]]:
    """
    Detect dialogue patterns in text including quotes, responses, and commentary.

    Args:
        text: Text to analyze for dialogue patterns

    Returns:
        List of dialogue segments with metadata
    """
    dialogue_segments = []

    # Pattern for quoted speech followed by attribution or response
    quote_patterns = [
        r'"([^"]+)"[,.]?\s*([^.!?]*[.!?])',  # "Quote," response.
        r'"([^"]+)"[,.]?\s+(\w+(?:\s+\w+){0,10}[.!?])',  # "Quote" short response.
        r'([^.!?]*[.!?])\s*"([^"]+)"',  # Response "quote"
        r'"([^"]+)"[,.]?\s*(\w+\s+(?:said|replied|asked|answered|continued|added|noted|observed|remarked|stated|declared|exclaimed|whispered|shouted|muttered|explained|insisted|argued|suggested|wondered|thought|concluded)[^.!?]*[.!?])',  # "Quote" attribution.
    ]

    for pattern in quote_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()

            # Extract the quote and response/attribution
            groups = match.groups()
            if len(groups) >= 2:
                quote_text = groups[0].strip()
                response_text = groups[1].strip()

                dialogue_segments.append({
                    'type': 'dialogue',
                    'start': start_pos,
                    'end': end_pos,
                    'quote': quote_text,
                    'response': response_text,
                    'full_text': match.group(0),
                    'word_count': len(match.group(0).split())
                })

    # Sort by position in text
    dialogue_segments.sort(key=lambda x: x['start'])

    return dialogue_segments

def analyze_chunk_relationships(chunks: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze relationships between chunks to identify those that should be merged.

    Args:
        chunks: List of chunk texts

    Returns:
        List of relationship analysis results
    """
    relationships = []

    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].strip()
        next_chunk = chunks[i + 1].strip()

        if not current_chunk or not next_chunk:
            continue

        current_words = current_chunk.split()
        next_words = next_chunk.split()

        relationship = {
            'chunk_index': i,
            'current_word_count': len(current_words),
            'next_word_count': len(next_words),
            'should_merge': False,
            'merge_reason': None
        }


        # Check for short chunks that should be merged using centralized thresholds
        if len(current_words) <= CHUNK_THRESHOLDS['short']:
            # Short chunk - analyze context for merging

            # Case 1: Short response after dialogue
            if (len(current_words) <= CHUNK_THRESHOLDS['dialogue_response'] and 
                any(word.lower() in ['said', 'replied', 'asked', 'answered', 'continued', 'added', 'noted', 'observed', 'remarked', 'stated', 'declared', 'exclaimed', 'whispered', 'shouted', 'muttered', 'explained', 'insisted', 'argued', 'suggested', 'wondered', 'thought', 'concluded'] for word in current_words)):
                relationship['should_merge'] = True
                relationship['merge_reason'] = 'short_dialogue_attribution'

            # Case 2: Short commentary or response
            elif (len(current_words) <= CHUNK_THRESHOLDS['dialogue_response'] and 
                  not current_chunk.endswith(('.', '!', '?')) and
                  not next_chunk[0].isupper()):
                relationship['should_merge'] = True
                relationship['merge_reason'] = 'incomplete_sentence'

            # Case 3: Very short fragments
            elif len(current_words) <= CHUNK_THRESHOLDS['fragment']:
                relationship['should_merge'] = True
                relationship['merge_reason'] = 'very_short_fragment'

            # Case 4: Short chunk followed by related content
            elif (len(current_words) <= CHUNK_THRESHOLDS['related_short'] and 
                  len(next_words) <= CHUNK_THRESHOLDS['short'] and
                  not current_chunk.endswith(('.', '!', '?'))):
                relationship['should_merge'] = True
                relationship['merge_reason'] = 'related_short_chunks'

        # Check for next chunk being very short
        if len(next_words) <= CHUNK_THRESHOLDS['very_short']:
            # Next chunk is very short - likely should be merged with current
            if not relationship['should_merge']:
                relationship['should_merge'] = True
                relationship['merge_reason'] = 'next_chunk_very_short'

        relationships.append(relationship)

    return relationships

def merge_conversational_chunks(chunks: List[str], min_chunk_size: int = None) -> Tuple[List[str], Dict[str, Any]]:
    """
    Merge chunks based on conversational patterns and minimum size requirements.

    Args:
        chunks: List of chunk texts
        min_chunk_size: Minimum number of words per chunk (defaults to CHUNK_THRESHOLDS['min_target'])

    Returns:
        Tuple of (merged_chunks, merge_statistics)
    """

    if not chunks:
        return chunks, {}
    # Use centralized threshold if not provided
    if min_chunk_size is None:
        min_chunk_size = CHUNK_THRESHOLDS['min_target']


    # Analyze relationships between chunks
    relationships = analyze_chunk_relationships(chunks)

    merged_chunks = []
    merge_stats = {
        'original_count': len(chunks),
        'merges_performed': 0,
        'merge_reasons': {},
        'final_count': 0,
        'short_chunks_remaining': 0
    }

    i = 0
    while i < len(chunks):
        current_chunk = chunks[i].strip()

        if not current_chunk:
            i += 1
            continue

        # Check if this chunk should be merged with the next one
        should_merge_forward = False
        merge_reason = None

        if i < len(relationships):
            rel = relationships[i]
            should_merge_forward = rel['should_merge']
            merge_reason = rel['merge_reason']

        if should_merge_forward and i + 1 < len(chunks):
            # Merge current chunk with next chunk
            next_chunk = chunks[i + 1].strip()
            if next_chunk:
                # Determine appropriate separator
                separator = ' '
                if current_chunk.endswith(('.', '!', '?')):
                    separator = ' '
                elif current_chunk.endswith((',', ';', ':')):
                    separator = ' '
                else:
                    separator = ' '

                merged_text = current_chunk + separator + next_chunk
                merged_chunks.append(merged_text)

                merge_stats['merges_performed'] += 1
                if merge_reason:
                    merge_stats['merge_reasons'][merge_reason] = merge_stats['merge_reasons'].get(merge_reason, 0) + 1

                i += 2  # Skip both chunks since they were merged
            else:
                merged_chunks.append(current_chunk)
                i += 1
        else:
            # Keep chunk as is
            merged_chunks.append(current_chunk)
            i += 1

    # Second pass: merge any remaining very short chunks
    final_chunks = []
    i = 0
    while i < len(merged_chunks):
        current_chunk = merged_chunks[i].strip()
        current_words = len(current_chunk.split())

        if current_words < min_chunk_size and i + 1 < len(merged_chunks):
            # Try to merge with next chunk
            next_chunk = merged_chunks[i + 1].strip()
            next_words = len(next_chunk.split())

            # Merge if combined size is reasonable (use centralized threshold)

            if current_words + next_words <= min_chunk_size * 3:
                separator = ' ' if not current_chunk.endswith(('.', '!', '?')) else ' '
                merged_text = current_chunk + separator + next_chunk
                final_chunks.append(merged_text)

                merge_stats['merges_performed'] += 1
                merge_stats['merge_reasons']['second_pass_size'] = merge_stats['merge_reasons'].get('second_pass_size', 0) + 1

                i += 2
            else:
                final_chunks.append(current_chunk)
                i += 1
        else:
            final_chunks.append(current_chunk)
            i += 1

    # Count remaining short chunks
    for chunk in final_chunks:
        if len(chunk.split()) < min_chunk_size:
            merge_stats['short_chunks_remaining'] += 1

    merge_stats['final_count'] = len(final_chunks)

    return final_chunks, merge_stats

def semantic_chunker(
    structured_blocks: list[dict],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int = None,
    enable_dialogue_detection: bool = True
) -> list[Document]:

    """
    Chunks the document using Haystack's DocumentSplitter with enhanced conversational text handling.

    This enhanced version includes:
    - Dialogue pattern detection
    - Minimum chunk size enforcement
    - Intelligent merging of short fragments
    - Conversational flow preservation
    - Debugging output for chunk quality analysis

    Args:
        structured_blocks: List of text blocks from PDF parsing
        chunk_size: Target chunk size in words

        overlap: Overlap size in words
        min_chunk_size: Minimum chunk size in words (defaults to max(8, chunk_size // 10))
        enable_dialogue_detection: Whether to enable dialogue pattern detection


    Returns:
        List of Document objects representing chunks
    """
    if not structured_blocks:
        return []

    # Set default minimum chunk size using centralized threshold
    if min_chunk_size is None:
        min_chunk_size = max(CHUNK_THRESHOLDS['min_target'], chunk_size // 10)


    logging.info(f"Starting enhanced semantic chunking: {len(structured_blocks)} blocks, target size={chunk_size} words, overlap={overlap} words, min_chunk_size={min_chunk_size} words, dialogue_detection={enable_dialogue_detection}")

    # Debug: Analyze input blocks
    total_input_chars = 0
    dialogue_blocks = 0
    for i, block in enumerate(structured_blocks):
        block_text = block.get("text", "")
        block_chars = len(block_text)
        total_input_chars += block_chars

        # Check for dialogue patterns
        dialogue_segments = detect_dialogue_patterns(block_text)
        if dialogue_segments:
            dialogue_blocks += 1

        if i < 3:  # Log first 3 blocks for debugging
            logging.info(f"Block {i}: {block_chars} chars, {len(dialogue_segments)} dialogue segments, preview: '{block_text[:100]}...'")

    logging.info(f"Total input: {total_input_chars} characters, {dialogue_blocks} blocks with dialogue")

    # Join blocks with paragraph-aware spacing to better preserve structure
    full_text = "\n\n".join(
        block["text"].strip() for block in structured_blocks
        if block["text"].strip()
    )

    logging.info(f"Combined text length: {len(full_text)} characters")

    if not full_text.strip():
        return []

    # Detect dialogue patterns in full text (if enabled)
    dialogue_segments = []
    if enable_dialogue_detection:
        dialogue_segments = detect_dialogue_patterns(full_text)
        logging.info(f"Detected {len(dialogue_segments)} dialogue segments in full text")
    else:
        logging.info("Dialogue detection disabled")


    # Calculate maximum characters per chunk based on word target
    # Assume average 5 characters per word + spaces
    max_chars_per_chunk = chunk_size * 6  # Conservative estimate
    logging.info(f"Target max characters per chunk: {max_chars_per_chunk}")

    # Initialize the DocumentSplitter with conservative settings
    # Use smaller chunk size to ensure we don't exceed limits
    conservative_chunk_size = min(chunk_size, 800)  # Cap at 800 words max
    conservative_overlap = min(overlap, conservative_chunk_size // 4)  # Overlap no more than 25%

    logging.info(f"Using conservative settings: {conservative_chunk_size} words, {conservative_overlap} overlap")

    splitter = DocumentSplitter(
        split_by="word",
        split_length=conservative_chunk_size,
        split_overlap=conservative_overlap,
        respect_sentence_boundary=True
    )
    splitter.warm_up()

    # Create a single document from the full text
    document = Document(content=full_text)

    # Split the document into chunks
    result = splitter.run(documents=[document])
    initial_chunks = result.get("documents", [])

    # Extract chunk texts for analysis
    chunk_texts = [chunk.content.strip() for chunk in initial_chunks if chunk.content.strip()]

    # Debug: Analyze initial chunks
    logging.info(f"DocumentSplitter produced {len(chunk_texts)} initial chunks")

    short_chunks_initial = [chunk for chunk in chunk_texts if len(chunk.split()) <= CHUNK_THRESHOLDS['short']]
    very_short_chunks_initial = [chunk for chunk in chunk_texts if len(chunk.split()) <= CHUNK_THRESHOLDS['very_short']]

    logging.info(f"Initial short chunks (≤{CHUNK_THRESHOLDS['short']} words): {len(short_chunks_initial)}")
    logging.info(f"Initial very short chunks (≤{CHUNK_THRESHOLDS['very_short']} words): {len(very_short_chunks_initial)}")

    for i, chunk in enumerate(chunk_texts[:5]):  # Log first 5 chunks
        chunk_words = len(chunk.split())
        logging.info(f"Initial chunk {i}: {chunk_words} words, preview: '{chunk[:100]}...'")

    # Apply conversational text merging with configured minimum chunk size
    merged_chunk_texts, merge_stats = merge_conversational_chunks(chunk_texts, min_chunk_size)

    logging.info(f"Conversational merging completed: {merge_stats}")

    # Convert back to Document objects
    final_chunks = []
    max_allowed_chars = 8000  # Strict limit: 8k characters max per chunk

    for i, chunk_text in enumerate(merged_chunk_texts):
        if not chunk_text.strip():
            continue

        chunk_chars = len(chunk_text)
        chunk_words = len(chunk_text.split())

        if chunk_chars <= max_allowed_chars:
            # Chunk is acceptable size
            final_chunks.append(Document(content=chunk_text))
        else:
            # Force split oversized chunk (same logic as before)
            logging.warning(f"Force-splitting oversized chunk {i} ({chunk_chars} chars)")

            # Split by paragraphs first, then by sentences if needed
            paragraphs = chunk_text.split('\n\n')
            current_chunk = ""

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # If adding this paragraph would exceed limit, save current chunk
                if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_allowed_chars:
                    if current_chunk.strip():
                        final_chunks.append(Document(content=current_chunk.strip()))
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph

                # If single paragraph is too large, split by sentences
                if len(current_chunk) > max_allowed_chars:
                    sentences = current_chunk.split('. ')
                    sentence_chunk = ""

                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue

                        # Add period back if it was removed by split
                        if not sentence.endswith('.') and sentence != sentences[-1]:
                            sentence += '.'

                        if sentence_chunk and len(sentence_chunk) + len(sentence) + 2 > max_allowed_chars:
                            if sentence_chunk.strip():
                                final_chunks.append(Document(content=sentence_chunk.strip()))
                            sentence_chunk = sentence
                        else:
                            if sentence_chunk:
                                sentence_chunk += " " + sentence
                            else:
                                sentence_chunk = sentence

                        # If single sentence is still too large, force character split
                        if len(sentence_chunk) > max_allowed_chars:
                            # Character-based splitting as last resort
                            while len(sentence_chunk) > max_allowed_chars:
                                break_point = max_allowed_chars
                                # Try to break at word boundary
                                last_space = sentence_chunk.rfind(' ', 0, break_point)
                                if last_space > break_point // 2:
                                    break_point = last_space

                                final_chunks.append(Document(content=sentence_chunk[:break_point].strip()))
                                sentence_chunk = sentence_chunk[break_point:].strip()

                    if sentence_chunk.strip():
                        current_chunk = sentence_chunk
                    else:
                        current_chunk = ""

            # Add any remaining content
            if current_chunk.strip():
                final_chunks.append(Document(content=current_chunk.strip()))

    # Final analysis and logging
    final_chunk_sizes = [len(chunk.content.split()) for chunk in final_chunks]
    short_chunks_final = [size for size in final_chunk_sizes if size <= CHUNK_THRESHOLDS['short']]
    very_short_chunks_final = [size for size in final_chunk_sizes if size <= CHUNK_THRESHOLDS['very_short']]

    logging.info(f"Final chunking results:")
    logging.info(f"  Total chunks: {len(final_chunks)}")
    logging.info(f"  Short chunks (≤{CHUNK_THRESHOLDS['short']} words): {len(short_chunks_final)} (reduced from {len(short_chunks_initial)})")
    logging.info(f"  Very short chunks (≤{CHUNK_THRESHOLDS['very_short']} words): {len(very_short_chunks_final)} (reduced from {len(very_short_chunks_initial)})")

    if final_chunk_sizes:
        avg_size = sum(final_chunk_sizes) / len(final_chunk_sizes)
        max_size = max(final_chunk_sizes)
        min_size = min(final_chunk_sizes)
        logging.info(f"  Chunk size stats: avg={avg_size:.1f} words, min={min_size} words, max={max_size} words")

    # Log examples of remaining short chunks for debugging
    if short_chunks_final:
        logging.info("Examples of remaining short chunks:")
        for i, chunk in enumerate(final_chunks):
            chunk_words = len(chunk.content.split())
            if chunk_words <= CHUNK_THRESHOLDS['short'] and i < 5:  # Log first 5 short chunks
                logging.info(f"  Short chunk {i}: {chunk_words} words - '{chunk.content[:100]}...'")

    return final_chunks
