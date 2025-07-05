from .parsing import extract_structured_text
from .splitter import semantic_chunker
from .utils import format_chunks_with_metadata
from .ai_enrichment import init_llm

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
    
    # 2. Semantic Pass: Chunk the blocks into coherent documents
    haystack_chunks = semantic_chunker(structured_blocks, chunk_size, overlap)
    
    # 3. Final Formatting and AI Enrichment
    final_chunks = format_chunks_with_metadata(
        haystack_chunks,
        structured_blocks,
        generate_metadata=generate_metadata,
        perform_ai_enrichment=perform_ai_enrichment
    )
    
    return final_chunks
