from haystack.nodes import PreProcessor

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=400,
    split_overlap=50,
    split_respect_sentence_boundary=True,
    max_chars_check=20000
)

def semantic_split(text, chunk_size, overlap):
    """
    Splits a single block of text into chunks using the Haystack PreProcessor.
    The PreProcessor is configured to split by "passage" (i.e., by "\n\n").
    The chunk_size (in words) and overlap are applied to these passages.
    """
    # The preprocessor expects a list of dictionaries or Document objects.
    # We pass the entire text as a single document to be split.
    documents = [{"content": text}]
    
    processed_documents = preprocessor.process(
        documents=documents,
        split_length=chunk_size,
        split_overlap=overlap
    )
    return processed_documents
