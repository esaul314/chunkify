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
    

    # Join blocks with paragraph-aware spacing to better preserve structure
    full_text = "\n\n".join(
        block["text"].strip() for block in structured_blocks
        if block["text"].strip()
    )
    
    
    if not full_text.strip():
        return []
    
    # Initialize the DocumentSplitter
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
    
    logging.info(f"Semantic chunking completed: {len(chunks)} chunks created from {len(structured_blocks)} blocks")
    
    return chunks
