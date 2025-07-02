from haystack.nodes import PreProcessor
from haystack.schema import Document

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    split_by="word",
    split_length=400,
    split_overlap=50,
    split_respect_sentence_boundary=True,
)

def semantic_chunker(structured_blocks: list[dict], chunk_size: int, overlap: int) -> list[Document]:
    """
    Takes a list of structured text blocks, combines them, and splits them
    into semantically coherent chunks using Haystack.
    """
    full_text = "\n\n".join(block["text"] for block in structured_blocks)
    
    if not full_text:
        return []

    # Pass data as a list of dictionaries, which Haystack handles
    docs_to_process = [{"content": full_text}]
    
    split_docs = preprocessor.process(
        docs_to_process,
        split_length=chunk_size,
        split_overlap=overlap
    )
    
    return split_docs
