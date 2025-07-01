from haystack.nodes import PreProcessor

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False, # We do our own whitespace cleaning now
    split_by="word",
    split_length=400,
    split_overlap=50,
    split_respect_sentence_boundary=True,
)

def semantic_chunker(structured_blocks: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    Takes a list of structured text blocks, combines them, and splits them
    into semantically coherent chunks using Haystack.
    
    Returns a list of dictionaries, where each dict is a chunk of text.
    """
    # Combine the text from all blocks, separated by double newlines to mark
    # paragraphs for the splitter's context, even though we split by word.
    full_text = "\n\n".join(block["text"] for block in structured_blocks)
    
    if not full_text:
        return []

    # Haystack's PreProcessor works on a list of 'Document' objects or dicts
    haystack_docs = [{"content": full_text}]
    
    split_docs = preprocessor.process(
        haystack_docs,
        split_length=chunk_size,
        split_overlap=overlap
    )
    
    # Return a list of simple dictionaries with the text content
    return [{"text": doc.content} for doc in split_docs]
