import re
from pathlib import Path

def clean_text(text: str) -> str:
    """
    Performs minimal, safe pre-cleaning before splitting.
    """
    # Remove form feed characters, which can interfere with splitting.
    return text.replace("\x0c", "")

def post_process_chunk_text(text: str) -> str:
    """
    Cleans the text of an individual chunk after splitting.
    Normalizes whitespace and line breaks.
    """
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    return text.strip()

def remove_metadata_and_clean(chunks):
    """
    Removes metadata and applies final cleaning to the text of each chunk.
    """
    cleaned_chunks = []
    for chunk in chunks:
        if hasattr(chunk, 'content'):
            cleaned_text = post_process_chunk_text(chunk.content)
            cleaned_chunks.append({"text": cleaned_text})
    return cleaned_chunks

def enrich_metadata(filepath):
    """
    Creates a function that processes final chunks.
    Currently configured to strip metadata and apply post-cleaning.
    """
    def wrapper(chunks):
        processed_chunks = remove_metadata_and_clean(chunks)
        return processed_chunks
    return wrapper
