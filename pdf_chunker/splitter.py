import logging
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def semantic_chunker(structured_blocks: list[dict], chunk_size: int, overlap: int) -> list[Document]:
    """
    Chunks the document using Haystack's DocumentSplitter.
    
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
    
    # Initialize the DocumentSplitter with more conservative settings
    splitter = DocumentSplitter(
        split_by="word",
        split_length=chunk_size,
        split_overlap=overlap,
        respect_sentence_boundary=True
    )
    splitter.warm_up()
    
    # Create a single document from the full text
    document = Document(content=full_text)
    
    # Split the document into chunks
    result = splitter.run(documents=[document])
    chunks = result.get("documents", [])
    
    # Debug: Analyze output chunks
    logging.info(f"Semantic chunking completed: {len(chunks)} chunks created")
    
    for i, chunk in enumerate(chunks):
        chunk_chars = len(chunk.content)
        chunk_words = len(chunk.content.split())
        logging.info(f"Chunk {i}: {chunk_chars} chars, {chunk_words} words")
        
        # Warning for extremely large chunks
        if chunk_chars > 10000:  # More than 10k characters is suspicious
            logging.warning(f"LARGE CHUNK DETECTED: Chunk {i} has {chunk_chars} characters!")
            logging.warning(f"Chunk preview: '{chunk.content[:200]}...'")
    
    # Validate chunk sizes and split oversized chunks if needed
    validated_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk.content) > 50000:  # If chunk is larger than 50k chars, force split
            logging.warning(f"Force-splitting oversized chunk {i} ({len(chunk.content)} chars)")
            # Simple character-based splitting as fallback
            content = chunk.content
            max_chunk_chars = 5000  # Conservative chunk size
            
            while content:
                if len(content) <= max_chunk_chars:
                    validated_chunks.append(Document(content=content))
                    break
                
                # Find a good break point (sentence or paragraph boundary)
                break_point = max_chunk_chars
                for boundary in ['. ', '.\n', '\n\n']:
                    last_boundary = content.rfind(boundary, 0, max_chunk_chars)
                    if last_boundary > max_chunk_chars // 2:  # Don't break too early
                        break_point = last_boundary + len(boundary)
                        break
                
                validated_chunks.append(Document(content=content[:break_point]))
                content = content[break_point:].lstrip()
        else:
            validated_chunks.append(chunk)
    
    logging.info(f"Final validation: {len(validated_chunks)} chunks after size validation")
    
    return validated_chunks
