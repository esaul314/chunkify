import logging
from typing import List
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def semantic_chunker(structured_blocks: list[dict], chunk_size: int, overlap: int) -> list[Document]:
    """
    Chunks the document using Haystack's DocumentSplitter, configured to split by
    word while respecting sentence boundaries. This is the most robust method.
    """
    full_text = "\n\n".join(block["text"] for block in structured_blocks)
    if not full_text:
        return []

    logging.info(f"Starting chunking with DocumentSplitter: chunk_size={chunk_size} words, overlap={overlap} words.")

    # Configure the DocumentSplitter to function like the original PreProcessor
    splitter = DocumentSplitter(
        split_by="word",
        split_length=chunk_size,
        split_overlap=overlap,
        respect_sentence_boundary=True # Corrected parameter name
    )

    # The splitter needs to be warmed up before running
    splitter.warm_up()
    
    # Create a single document to be split
    doc_to_split = [Document(content=full_text)]
    
    result = splitter.run(documents=doc_to_split)
    
    split_docs = result.get("documents", [])
    logging.info(f"DocumentSplitter created {len(split_docs)} chunks.")

    return split_docs
