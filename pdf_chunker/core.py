from .parsing import extract_structured_text
from .splitter import semantic_chunker
from .utils import clean_structured_text

def process_document(filepath: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    Core pipeline for processing a document.
    
    1. Performs a "Structural Pass" to extract text blocks with their type.
    2. Cleans the text within each structured block.
    3. Performs a "Semantic Pass" by combining the text and chunking it
       while respecting sentence boundaries.
    """
    # 1. Structural Pass
    structured_blocks = extract_structured_text(filepath)
    
    # 2. Clean the text within the blocks
    cleaned_blocks = clean_structured_text(structured_blocks)
    
    # 3. Semantic Pass (Chunking)
    chunks = semantic_chunker(cleaned_blocks, chunk_size, overlap)
    
    return chunks
