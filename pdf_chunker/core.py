from .parsing import extract_structured_text
from .splitter import semantic_chunker
from .utils import format_chunks_with_metadata

def process_document(
    filepath: str, 
    chunk_size: int, 
    overlap: int, 
    generate_metadata: bool = True
) -> list[dict]:
    """
    Core pipeline for processing a document using a two-pass approach.
    """
    # 1. Structural Pass: Extract text into structured blocks
    structured_blocks = extract_structured_text(filepath)
    
    # 2. Semantic Pass: Chunk the blocks into coherent documents
    haystack_chunks = semantic_chunker(structured_blocks, chunk_size, overlap)
    
    # 3. Final Formatting: Map metadata back and format for output
    final_chunks = format_chunks_with_metadata(
        haystack_chunks, 
        structured_blocks, 
        generate_metadata=generate_metadata
    )
    
    return final_chunks
