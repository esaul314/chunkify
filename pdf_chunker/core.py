from .parsing import extract_text
from .splitter import semantic_split
from .utils import clean_text, enrich_metadata

def chunk_pdf(filepath, chunk_size=1000, overlap=100):
    """
    Core pipeline for processing a document.
    Extracts, cleans, chunks, and formats the text.
    """
    # Direct function calls to avoid any potential caching issues with `pipe`.
    
    # 1. Extract text from the source file
    raw_text = extract_text(filepath)

    # 2. Clean the extracted text
    cleaned_text = clean_text(raw_text)

    # 3. Split the cleaned text into chunks
    chunks = semantic_split(cleaned_text, chunk_size, overlap)

    # 4. Process metadata (currently removes it)
    final_chunks = enrich_metadata(filepath)(chunks)

    return final_chunks
