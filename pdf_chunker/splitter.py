import logging
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def semantic_chunker(structured_blocks: list[dict], chunk_size: int, overlap: int) -> list[Document]:
    """
    Chunks the document using Haystack's DocumentSplitter with strict size validation.
    
    Args:
        structured_blocks: List of text blocks from PDF parsing
        chunk_size: Target chunk size in words
        overlap: Overlap size in words
    
    Returns:
        List of Document objects representing chunks
    """
    if not structured_blocks:
        return []
    
    logging.info(f"Starting semantic chunking: {len(structured_blocks)} blocks, target size={chunk_size} words, overlap={overlap} words")
    
    # Debug: Analyze input blocks
    total_input_chars = 0
    for i, block in enumerate(structured_blocks):
        block_text = block.get("text", "")
        block_chars = len(block_text)
        total_input_chars += block_chars
        if i < 3:  # Log first 3 blocks for debugging
            logging.info(f"Block {i}: {block_chars} chars, preview: '{block_text[:100]}...'")
    
    logging.info(f"Total input text: {total_input_chars} characters")

    # Join blocks with paragraph-aware spacing to better preserve structure
    full_text = "\n\n".join(
        block["text"].strip() for block in structured_blocks
        if block["text"].strip()
    )
    
    logging.info(f"Combined text length: {len(full_text)} characters")
    
    if not full_text.strip():
        return []
    
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
    chunks = result.get("documents", [])
    
    # Debug: Analyze output chunks before validation
    logging.info(f"DocumentSplitter produced {len(chunks)} initial chunks")
    
    for i, chunk in enumerate(chunks[:5]):  # Log first 5 chunks
        chunk_chars = len(chunk.content)
        chunk_words = len(chunk.content.split())
        logging.info(f"Initial chunk {i}: {chunk_chars} chars, {chunk_words} words")
        
        if chunk_chars > max_chars_per_chunk:
            logging.warning(f"OVERSIZED INITIAL CHUNK: Chunk {i} has {chunk_chars} characters (target: {max_chars_per_chunk})")
    
    # Strict validation and force-splitting of oversized chunks
    validated_chunks = []
    max_allowed_chars = 8000  # Strict limit: 8k characters max per chunk
    
    for i, chunk in enumerate(chunks):
        chunk_content = chunk.content.strip()
        
        if len(chunk_content) <= max_allowed_chars:
            # Chunk is acceptable size
            validated_chunks.append(Document(content=chunk_content))
        else:
            # Force split oversized chunk
            logging.warning(f"Force-splitting oversized chunk {i} ({len(chunk_content)} chars)")
            
            # Split by paragraphs first, then by sentences if needed
            paragraphs = chunk_content.split('\n\n')
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If adding this paragraph would exceed limit, save current chunk
                if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_allowed_chars:
                    if current_chunk.strip():
                        validated_chunks.append(Document(content=current_chunk.strip()))
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
                                validated_chunks.append(Document(content=sentence_chunk.strip()))
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
                                
                                validated_chunks.append(Document(content=sentence_chunk[:break_point].strip()))
                                sentence_chunk = sentence_chunk[break_point:].strip()
                    
                    if sentence_chunk.strip():
                        current_chunk = sentence_chunk
                    else:
                        current_chunk = ""
            
            # Add any remaining content
            if current_chunk.strip():
                validated_chunks.append(Document(content=current_chunk.strip()))
    
    # Final validation - ensure no chunk exceeds the strict limit
    final_chunks = []
    for i, chunk in enumerate(validated_chunks):
        chunk_chars = len(chunk.content)
        if chunk_chars > max_allowed_chars:
            logging.error(f"CRITICAL: Chunk {i} still oversized after force-splitting ({chunk_chars} chars)")
            # Emergency character truncation
            truncated_content = chunk.content[:max_allowed_chars]
            # Try to end at sentence boundary
            last_period = truncated_content.rfind('. ')
            if last_period > max_allowed_chars * 0.8:
                truncated_content = truncated_content[:last_period + 1]
            final_chunks.append(Document(content=truncated_content))
        else:
            final_chunks.append(chunk)
    
    logging.info(f"Final validation: {len(final_chunks)} chunks after strict size validation")
    
    # Log final chunk size statistics
    if final_chunks:
        chunk_sizes = [len(chunk.content) for chunk in final_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        logging.info(f"Final chunk statistics: avg={avg_size:.0f} chars, max={max_size} chars")
        
        oversized_count = sum(1 for size in chunk_sizes if size > max_allowed_chars)
        if oversized_count > 0:
            logging.error(f"ERROR: {oversized_count} chunks still exceed {max_allowed_chars} character limit!")
    
    return final_chunks
