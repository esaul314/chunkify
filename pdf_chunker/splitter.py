from haystack.nodes import PreProcessor

preprocessor = PreProcessor(
    split_by="word",
    split_length=300, 
    split_overlap=50,  
    split_respect_sentence_boundary=True,
    max_chars_check=20000  # Added this line
)

def semantic_split(text, chunk_size, overlap):
    processed_documents = preprocessor.process(
        documents=[{"content": text}], 
        split_length=chunk_size,  # Use the chunk_size passed to this function
        split_overlap=overlap     # Use the overlap passed to this function
    )
    return processed_documents

